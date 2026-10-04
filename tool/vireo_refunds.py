#!/usr/bin/env python3
"""Vireo Audio refund summary: reconcile the export, classify every refund, build the board pack.

    python3 vireo_refunds.py                 # rules + cached LLM opinions, data in ../
    python3 vireo_refunds.py --llm           # also call local Ollama for uncached review tickets
    python3 vireo_refunds.py --data /path --out out

Outputs (in --out):
    board_pack.xlsx     tables for the board pack (see README)
    refund_tickets.csv  one row per refund ticket with every derived field
    summary.txt         headline numbers
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

import llm
import rules

HERE = os.path.dirname(os.path.abspath(__file__))

# Policy constants (support-policy.pdf v3.2)
REPL_HANDLING_INR = 340         # §5: replacement cost = unit cost + Rs 340 reverse pickup + forward shipping
GOODWILL_CAP_INR = 500          # §5: goodwill capped at Rs 500 per ticket
LEGACY_UNIT_DIVISOR = 100       # §9 + Sameer: Freshdesk stored money in paise
QUARTER_FROM = "2026-04"        # latest full quarter in the data (Apr-Jun 2026)


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# 1. Load and reconcile
# --------------------------------------------------------------------------- #
def load(data_dir: str):
    p = lambda f: os.path.join(data_dir, f)
    t = pd.read_csv(p("tickets.csv"), dtype=str, keep_default_na=False)
    agents = pd.read_csv(p("agents.csv"), dtype=str, keep_default_na=False)
    orders = pd.read_csv(p("orders.csv"), dtype=str, keep_default_na=False)
    products = pd.read_csv(p("products.csv"), dtype=str, keep_default_na=False)
    # customers.csv in the pack is a PNG image, not a CSV, and is not needed for refunds.
    return t, agents, orders, products


def reconcile(t: pd.DataFrame):
    """Return (clean ticket table, reconciliation table by quarter)."""
    t = t.copy()
    t["amt_export"] = pd.to_numeric(t["refund_amount_inr"], errors="coerce").fillna(0.0)
    t["is_legacy"] = t["source_system"].eq("legacy_fd")
    t["amt_inr"] = t["amt_export"].where(~t["is_legacy"], t["amt_export"] / LEGACY_UNIT_DIVISOR)
    t["month"] = t["created_at"].str[:7]          # created_at is IST in both systems (resolved_at is not)
    t["quarter"] = pd.PeriodIndex(pd.to_datetime(t["created_at"]), freq="Q").astype(str)

    # Migration re-import: the same ticket_id appears once as legacy_fd and once as helpdesk,
    # identical except source_system and the money unit. Keep the legacy original.
    t = t.sort_values(["ticket_id", "is_legacy"], ascending=[True, False])
    t["is_reimport_copy"] = t.duplicated("ticket_id", keep="first")

    q = t.groupby("quarter").apply(lambda g: pd.Series({
        "A_export_as_is": g["amt_export"].sum(),
        "B_less_reimport_duplicates": -g.loc[g.is_reimport_copy, "amt_export"].sum(),
        "C_less_paise_as_rupees": -(g.loc[g.is_legacy & ~g.is_reimport_copy, "amt_export"].sum()
                                    - g.loc[g.is_legacy & ~g.is_reimport_copy, "amt_inr"].sum()),
        "D_reconciled_refunds_inr": g.loc[~g.is_reimport_copy, "amt_inr"].sum(),
        "of_which_status_resolved": g.loc[~g.is_reimport_copy & g.status.eq("resolved"), "amt_inr"].sum(),
        "of_which_auto_closed": g.loc[~g.is_reimport_copy & g.status.eq("closed"), "amt_inr"].sum(),
        "of_which_open_or_pending": g.loc[~g.is_reimport_copy & g.status.isin(["open", "pending"]), "amt_inr"].sum(),
        "refund_tickets": int((g.loc[~g.is_reimport_copy, "amt_inr"] > 0).sum()),
        "all_tickets": int((~g.is_reimport_copy).sum()),
    }), include_groups=False)
    q["check_A+B+C=D"] = (q.A_export_as_is + q.B_less_reimport_duplicates + q.C_less_paise_as_rupees
                          - q.D_reconciled_refunds_inr).abs() < 0.5
    clean = t[~t.is_reimport_copy].drop(columns="is_reimport_copy")
    return clean, q.reset_index()


# --------------------------------------------------------------------------- #
# 2. Attribution and joins
# --------------------------------------------------------------------------- #
def attach_agents(df: pd.DataFrame, agents: pd.DataFrame) -> pd.DataFrame:
    """Roster is one row per assignment: pick the row valid on the ticket date (policy §7)."""
    a = agents.copy()
    a["from_date"] = pd.to_datetime(a["from_date"], errors="coerce")
    a["to_date"] = pd.to_datetime(a["to_date"], errors="coerce").fillna(pd.Timestamp("2099-12-31"))
    d = df[["ticket_id", "agent_id", "created_at"]].copy()
    d["ts"] = pd.to_datetime(d["created_at"])
    m = d.merge(a, on="agent_id", how="left")
    valid = m[(m.ts >= m.from_date) & (m.ts <= m.to_date)]
    # Fallback: if no assignment covers the date, use the agent's latest row.
    fallback = m.sort_values("from_date").drop_duplicates("ticket_id", keep="last")
    pick = pd.concat([valid, fallback]).drop_duplicates("ticket_id", keep="first")
    pick = pick[["ticket_id", "name", "site", "team", "shift", "tier"]].rename(
        columns={"name": "agent_name", "team": "agent_team", "site": "agent_site",
                 "shift": "agent_shift", "tier": "agent_tier"})
    return df.merge(pick, on="ticket_id", how="left")


def attach_products(df: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    p = products[["sku", "product_name", "family", "unit_cost_inr"]].rename(columns={"sku": "product_sku"})
    out = df.merge(p, on="product_sku", how="left")
    out["unit_cost_inr"] = pd.to_numeric(out["unit_cost_inr"], errors="coerce")
    return out


def order_key(df: pd.DataFrame) -> pd.Series:
    # README: order_id blank when not quoted; customer_id + product_sku is the fallback join.
    return df["order_id"].where(df["order_id"] != "", "C:" + df["customer_id"] + "|" + df["product_sku"])


# --------------------------------------------------------------------------- #
# 3. Classify refund tickets
# --------------------------------------------------------------------------- #
def classify(clean: pd.DataFrame, use_llm: bool, cache_dir: str) -> pd.DataFrame:
    clean = clean.copy()
    clean["order_key"] = order_key(clean)

    # Replacement evidence on every ticket (needed for the cross-ticket check too).
    ev = clean["agent_notes"].map(rules.replacement_evidence)
    clean["repl_text"] = ev.map(lambda e: e["verdict"])
    clean["repl_part_only"] = ev.map(lambda e: e["part_only"])
    clean["repl_evidence"] = ev.map(lambda e: e["evidence"])

    r = clean[clean["amt_inr"] > 0].copy()

    reason = r.apply(lambda x: rules.infer_reason(x.customer_message, x.agent_notes, x.category), axis=1)
    r["inferred_reason"] = reason.map(lambda z: z[0])
    r["reason_source"] = reason.map(lambda z: z[1])
    r["code_matches_text"] = [
        code in rules.COMPATIBLE.get(inf, set()) for code, inf in zip(r.refund_reason_code, r.inferred_reason)
    ]
    r["gw_over_cap"] = r["refund_reason_code"].eq("GW-OTHER") & (r["amt_inr"] > GOODWILL_CAP_INR)

    # Review queue: note contradicts itself, or flag and note disagree.
    flag_y = r["replacement_issued"].eq("Y")
    r["needs_review"] = r["repl_text"].eq("conflict") | (flag_y & r["repl_text"].eq("no"))
    rows = [(x.ticket_id, x.agent_notes, x.replacement_issued) for x in r[r.needs_review].itertuples()]
    opinions = llm.second_opinions(rows, cache_dir, use_model=use_llm, log=log)
    r["llm_verdict"] = r["ticket_id"].map(lambda k: opinions.get(k, {}).get("verdict", ""))
    r["llm_evidence"] = r["ticket_id"].map(lambda k: opinions.get(k, {}).get("evidence", ""))

    # Final same-ticket verdict. The note is the primary evidence; the flag is set by the
    # same agent and is wrong in both directions (see evaluation).
    def final(x):
        if x.repl_text == "yes":
            return "confirmed"                        # note says a unit went out
        if x.repl_text == "conflict" or (x.replacement_issued == "Y" and x.repl_text == "no"):
            return "review"                           # humans decide; LLM opinion shown alongside
        return "none"
    r["refund_plus_replacement"] = r.apply(final, axis=1)

    # Cross-ticket: another ticket on the same order shows a replacement going out.
    repl_tickets = clean[(clean.repl_text == "yes") | (clean.replacement_issued == "Y")]
    repl_orders = repl_tickets.groupby("order_key")["ticket_id"].apply(set)
    def other_ticket(x):
        s = repl_orders.get(x.order_key, set()) - {x.ticket_id}
        return ",".join(sorted(s))
    r["repl_on_other_ticket"] = r.apply(other_ticket, axis=1)

    # Money at stake on a confirmed double remedy: the replacement that should not have gone out.
    r["leakage_inr"] = 0.0
    conf = r.refund_plus_replacement.eq("confirmed")
    full_cost = r["unit_cost_inr"].fillna(0) + REPL_HANDLING_INR
    r.loc[conf, "leakage_inr"] = full_cost.where(~r.repl_part_only, REPL_HANDLING_INR)[conf]
    return r, clean


# --------------------------------------------------------------------------- #
# 4. Tables
# --------------------------------------------------------------------------- #
def monthly_pivot(r: pd.DataFrame, col: str) -> pd.DataFrame:
    p = r.pivot_table(index=col, columns="month", values="amt_inr", aggfunc="sum", fill_value=0)
    p["TOTAL"] = p.sum(axis=1)
    p = p.sort_values("TOTAL", ascending=False)
    p.loc["TOTAL"] = p.sum()
    return p.round(0)


def agent_table(r: pd.DataFrame, clean: pd.DataFrame, since: str | None) -> pd.DataFrame:
    rr = r if since is None else r[r.month >= since]
    cc = clean if since is None else clean[clean.month >= since]
    attended = cc[cc.status.isin(["resolved", "closed"])].groupby("agent_id").size()
    g = rr.groupby(["agent_team", "agent_id", "agent_name", "agent_site", "agent_tier"])
    t = g.agg(refunds=("ticket_id", "size"), refund_inr=("amt_inr", "sum"),
              gw_other_inr=("amt_inr", lambda s: s[rr.loc[s.index, "refund_reason_code"].eq("GW-OTHER")].sum()),
              refund_plus_replacement=("refund_plus_replacement", lambda s: (s == "confirmed").sum()),
              to_review=("refund_plus_replacement", lambda s: (s == "review").sum()),
              leakage_inr=("leakage_inr", "sum"),
              code_mismatch=("code_matches_text", lambda s: (~s).sum())).reset_index()
    t["tickets_attended"] = t["agent_id"].map(attended).fillna(0).astype(int)
    t["refunds_per_100_tickets"] = (100 * t.refunds / t.tickets_attended.where(t.tickets_attended > 0)).round(1)
    t["gw_other_share"] = (t.gw_other_inr / t.refund_inr).round(2)
    # Rank within team only: Tier 2 must not be compared with Tier 1 on volume (policy §6).
    t["rank_in_team_by_inr"] = t.groupby("agent_team")["refund_inr"].rank(ascending=False).astype(int)
    return t.sort_values(["agent_team", "refund_inr"], ascending=[True, False]).round(2)


def build(data_dir: str, out_dir: str, use_llm: bool) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    log(f"Reading data from {data_dir}")
    t, agents, orders, products = load(data_dir)
    log(f"  {len(t):,} ticket rows, {t.ticket_id.nunique():,} unique ticket ids")

    clean, recon = reconcile(t)
    clean = attach_agents(clean, agents)
    clean = attach_products(clean, products)
    r, clean = classify(clean, use_llm, HERE)

    # ---- tables
    by_code = monthly_pivot(r, "refund_reason_code")
    by_reason = monthly_pivot(r, "inferred_reason")
    by_team = monthly_pivot(r, "agent_team")
    r["agent_label"] = r["agent_id"] + " " + r["agent_name"].fillna("?") + " (" + r["agent_team"].fillna("?") + ")"
    by_agent_month = monthly_pivot(r, "agent_label")
    code_vs_text = pd.crosstab(r.refund_reason_code, r.inferred_reason, values=r.amt_inr,
                               aggfunc="sum", margins=True, margins_name="TOTAL").fillna(0).round(0)
    agents_all = agent_table(r, clean, None)
    agents_q = agent_table(r, clean, QUARTER_FROM)
    monthly = r.groupby("month").agg(refund_tickets=("ticket_id", "size"), refund_inr=("amt_inr", "sum"),
                                     confirmed_refund_plus_replacement=("refund_plus_replacement", lambda s: (s == "confirmed").sum()),
                                     leakage_inr=("leakage_inr", "sum"))
    monthly["all_tickets"] = clean.groupby("month").size()
    monthly["refund_rate"] = (monthly.refund_tickets / monthly.all_tickets).round(3)
    monthly["gw_other_inr"] = r[r.refund_reason_code == "GW-OTHER"].groupby("month").amt_inr.sum()
    monthly = monthly.fillna(0).reset_index()

    cols = ["ticket_id", "month", "created_at", "channel", "status", "agent_id", "agent_name", "agent_team",
            "assigned_team", "category", "product_sku", "order_id", "amt_inr", "refund_reason_code",
            "inferred_reason", "reason_source", "code_matches_text", "gw_over_cap", "replacement_issued",
            "repl_text", "refund_plus_replacement", "leakage_inr", "repl_evidence", "llm_verdict",
            "llm_evidence", "repl_on_other_ticket", "source_system", "customer_message", "agent_notes"]
    exceptions = r[r.refund_plus_replacement.eq("confirmed")][cols].sort_values("created_at")
    # Cross-ticket matches (repl_on_other_ticket) are kept as a column only: on inspection most
    # link unrelated issues on the same order (e.g. a cancellation and a later fault), so they
    # are too noisy for the review queue.
    review = r[r.refund_plus_replacement.eq("review")][cols]

    readme = pd.DataFrame({"How to read this workbook": [
        "All money in rupees. Months are by ticket creation date (IST).",
        "Reconciliation: export as-is -> minus migration duplicates -> minus legacy paise-as-rupees = reconciled refunds.",
        "Legacy Freshdesk rows (before 14 Sep 2025) store money in paise; they are divided by 100.",
        "By_dropdown_code: what agents picked. GW-OTHER is the first option in the dropdown, so it is over-used.",
        "By_actual_reason: what the customer message and agent note say the refund was for (rule-based reading).",
        "WRONG-ITEM and PRODUCT-FAULT have no dedicated refund code in policy section 5.",
        "Refund_plus_replacement: tickets where the note says the customer got money back AND a unit/reshipment. Policy section 5 forbids this.",
        "Leakage = product unit cost + Rs 340 handling (policy section 5 planning cost) for each confirmed case.",
        "Review_queue: notes that contradict themselves, or replacement flag = Y with nothing in the note. Not counted in leakage.",
        "Reconciliation 'of_which_status_resolved' is close to the helpdesk's own ~Rs 11 lakh/quarter report, which appears to count resolved tickets only.",
        "Sheets starting INTERNAL name individual agents or contain customer messages. Keep them out of the board pack.",
        "Agents: ranked within their own team only. Tier 2 (Escalations & Warranty) is not compared with Tier 1 (policy section 6).",
    ]})

    xlsx = os.path.join(out_dir, "board_pack.xlsx")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        readme.to_excel(w, sheet_name="Read_me", index=False)
        recon.to_excel(w, sheet_name="Reconciliation", index=False)
        monthly.to_excel(w, sheet_name="Monthly_overview", index=False)
        by_code.to_excel(w, sheet_name="By_dropdown_code")
        by_reason.to_excel(w, sheet_name="By_actual_reason")
        code_vs_text.to_excel(w, sheet_name="Code_vs_actual_reason")
        by_team.to_excel(w, sheet_name="By_team")
        # INTERNAL sheets name individual agents or carry customer text: for Finance / Ops, not the board.
        by_agent_month.to_excel(w, sheet_name="INTERNAL_Agent_by_month")
        agents_q.to_excel(w, sheet_name=f"INTERNAL_Agents_since_{QUARTER_FROM}", index=False)
        agents_all.to_excel(w, sheet_name="INTERNAL_Agents_18_months", index=False)
        exceptions.to_excel(w, sheet_name="INTERNAL_Refund+replacement", index=False)
        review.to_excel(w, sheet_name="INTERNAL_Review_queue", index=False)
        for ws in w.book.worksheets:
            ws.freeze_panes = "B2"
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(
                    60, max(10, max(len(str(c.value or "")) for c in col[:60]) + 2))
    r[cols].to_csv(os.path.join(out_dir, "refund_tickets.csv"), index=False)

    # ---- headline numbers
    last_q = r[r.month >= QUARTER_FROM]
    conf = r[r.refund_plus_replacement == "confirmed"]
    conf_q = last_q[last_q.refund_plus_replacement == "confirmed"]
    gw = r[r.refund_reason_code == "GW-OTHER"]
    h = {
        "export_rows": len(t), "reimport_duplicates": int(len(t) - len(clean)),
        "export_sum_inr": float(recon.A_export_as_is.sum()),
        "reconciled_sum_inr": float(recon.D_reconciled_refunds_inr.sum()),
        "reconciliation_balances": bool(recon["check_A+B+C=D"].all()),
        "refund_tickets": len(r),
        "last_quarter_refund_inr": float(last_q.amt_inr.sum()),
        "last_quarter_refund_tickets": len(last_q),
        "gw_other_share_of_inr": float(gw.amt_inr.sum() / r.amt_inr.sum()),
        "gw_other_over_cap_tickets": int(r.gw_over_cap.sum()),
        "gw_other_with_specific_reason_inr": float(gw[gw.inferred_reason.isin(
            ["CANCEL", "DUP-PAYMENT", "PRICE-ADJ", "RETURN-QC-OK", "LOST-TRANSIT", "WTY-BUYBACK", "DOA-REPL"])].amt_inr.sum()),
        "confirmed_refund_plus_replacement": len(conf),
        "confirmed_with_flag_N": int((conf.replacement_issued == "N").sum()),
        "leakage_18m_inr": float(conf.leakage_inr.sum()),
        "last_quarter_confirmed": len(conf_q),
        "last_quarter_rate": float(len(conf_q) / max(1, len(last_q))),
        "last_quarter_leakage_inr": float(conf_q.leakage_inr.sum()),
        "review_queue": len(review),
    }
    lines = [
        "VIREO REFUNDS - HEADLINE NUMBERS",
        f"Export rows {h['export_rows']:,}; migration duplicates removed {h['reimport_duplicates']:,}",
        f"Export summed as-is: Rs {h['export_sum_inr']:,.0f}  ->  reconciled: Rs {h['reconciled_sum_inr']:,.0f} "
        f"(bridge balances every quarter: {h['reconciliation_balances']})",
        f"Refund tickets (18 months): {h['refund_tickets']:,}",
        f"Last quarter (Apr-Jun 2026): Rs {h['last_quarter_refund_inr']:,.0f} on {h['last_quarter_refund_tickets']} tickets",
        f"GW-OTHER share of refund rupees: {h['gw_other_share_of_inr']:.0%}; GW-OTHER above the Rs 500 cap: {h['gw_other_over_cap_tickets']} tickets",
        f"GW-OTHER rupees whose note shows a specific coded reason: Rs {h['gw_other_with_specific_reason_inr']:,.0f}",
        f"Refund + replacement on the same ticket (confirmed by note): {h['confirmed_refund_plus_replacement']} "
        f"({h['confirmed_with_flag_N']} of them have replacement flag = N)",
        f"  cost of the extra replacement, 18 months: Rs {h['leakage_18m_inr']:,.0f}",
        f"  last quarter: {h['last_quarter_confirmed']} cases = {h['last_quarter_rate']:.1%} of refunds, Rs {h['last_quarter_leakage_inr']:,.0f}",
        f"Review queue (not counted): {h['review_queue']} tickets",
        f"Workbook: {xlsx}",
    ]
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log("\n".join(lines))
    return h


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.dirname(HERE), help="folder with tickets.csv etc. (default: parent of tool/)")
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--llm", action="store_true", help="call local Ollama for uncached review-queue tickets")
    a = ap.parse_args(argv)
    if not os.path.exists(os.path.join(a.data, "tickets.csv")):
        print(f"tickets.csv not found in {a.data}. Use --data to point at the data pack.", file=sys.stderr)
        return 2
    if a.llm and not llm.available():
        log("Ollama not reachable; continuing with rules + cache only.")
        a.llm = False
    build(a.data, a.out, a.llm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
