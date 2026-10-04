#!/usr/bin/env python3
"""How do we know the output is right? Three kinds of check.

1. Arithmetic checks on the reconciliation (must all pass, or the numbers are not used).
2. Text-reading accuracy against a hand-labelled sample (eval/gold_labels.csv):
     - 100 refund tickets drawn at random (seed 7)      -> unbiased error rate
     - 39 targeted hard cases (flag/note disagreements) -> where it breaks
   compared with the two things Vireo has today: the Y/N replacement flag and the dropdown code.
3. Optional: the same sample through the local LLM (--llm) to compare it with the rules.

    python3 evaluate.py            # uses out/refund_tickets.csv (run vireo_refunds.py first)
    python3 evaluate.py --llm      # also score llama3.1 via Ollama on the gold notes (slow, cached)
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

import llm
import rules
import vireo_refunds as vr

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL_TO_GOLD = {"confirmed": "yes", "none": "no", "review": "unclear"}
FAMILY = {"DOA-REPL": "PRODUCT", "PRODUCT-FAULT": "PRODUCT"}   # DOA vs fault is not decidable from text


def pct(a, b):
    return f"{a}/{b} = {100 * a / b:.0f}%" if b else "n/a"


def arithmetic_checks(data_dir: str) -> list[tuple[str, bool]]:
    t, *_ = vr.load(data_dir)
    clean, recon = vr.reconcile(t)
    out = []
    out.append(("Bridge export -> reconciled balances in every quarter", bool(recon["check_A+B+C=D"].all())))
    dup = t[t.ticket_id.duplicated(keep=False)].copy()
    dup["amt"] = pd.to_numeric(dup.refund_amount_inr, errors="coerce")
    p = dup.pivot_table(index="ticket_id", columns="source_system", values="amt", aggfunc="first").dropna()
    out.append((f"All {len(p)} duplicate pairs with a refund agree once legacy is /100",
                bool(((p.legacy_fd / 100 - p.helpdesk).abs() < 0.01).all())))
    other_cols = [c for c in t.columns if c not in ("source_system", "refund_amount_inr")]
    out.append(("Duplicate pairs are identical on every other column",
                bool((dup.groupby("ticket_id")[other_cols].nunique() <= 1).all().all())))
    legacy = pd.to_numeric(clean.loc[clean.source_system == "legacy_fd", "refund_amount_inr"], errors="coerce").dropna()
    out.append(("Every legacy refund is a whole multiple of 100 (consistent with paise)", bool((legacy % 100 == 0).all())))
    out.append(("No ticket left duplicated after cleaning", not clean.ticket_id.duplicated().any()))
    r = pd.read_csv(os.path.join(HERE, "out", "refund_tickets.csv"), keep_default_na=False)
    out.append(("Ticket-level detail sums to the reconciled total",
                abs(r.amt_inr.sum() - recon.D_reconciled_refunds_inr.sum()) < 0.5))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.dirname(HERE))
    ap.add_argument("--llm", action="store_true")
    a = ap.parse_args(argv)
    if not os.path.exists(os.path.join(HERE, "out", "refund_tickets.csv")):
        print("Run vireo_refunds.py first.", file=sys.stderr)
        return 2

    print("== 1. Arithmetic checks")
    checks = arithmetic_checks(a.data)
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    r = pd.read_csv(os.path.join(HERE, "out", "refund_tickets.csv"), keep_default_na=False)
    g = pd.read_csv(os.path.join(HERE, "eval", "gold_labels.csv")).merge(r, on="ticket_id", how="left")
    g["tool_repl"] = g.refund_plus_replacement.map(TOOL_TO_GOLD)
    g["flag_repl"] = g.replacement_issued.map({"Y": "yes", "N": "no"})
    rnd = g[g.stratum == "random"]

    print("\n== 2a. 'Refund AND replacement?' on 100 random refund tickets")
    print(pd.crosstab(rnd.gold_replacement, rnd.tool_repl, rownames=["gold"], colnames=["tool"]).to_string())
    decided = rnd[rnd.gold_replacement != "unclear"]
    tool_ok = (decided.tool_repl == decided.gold_replacement).sum()
    flag_ok = (decided.flag_repl == decided.gold_replacement).sum()
    gy = decided[decided.gold_replacement == "yes"]
    print(f"  tool agrees with gold:          {pct(tool_ok, len(decided))}")
    print(f"  helpdesk Y/N flag agrees:       {pct(flag_ok, len(decided))}")
    print(f"  true double-remedies found by tool: {pct((gy.tool_repl == 'yes').sum(), len(gy))};"
          f" by flag: {pct((gy.flag_repl == 'yes').sum(), len(gy))}")
    ty = rnd[rnd.tool_repl == "yes"]
    print(f"  tool 'confirmed' that are truly double: {pct((ty.gold_replacement == 'yes').sum(), len(ty))}")

    print("\n== 2b. Same question on 39 targeted hard cases")
    tg = g[g.stratum == "targeted"]
    print(pd.crosstab(tg.gold_replacement, tg.tool_repl, rownames=["gold"], colnames=["tool"]).to_string())
    bad = tg[(tg.tool_repl != tg.gold_replacement)]
    print(f"  disagreements: {len(bad)}; of which a wrong 'confirmed' (would overstate leakage): "
          f"{((bad.tool_repl == 'yes') & (bad.gold_replacement != 'yes')).sum()}")

    print("\n== 2c. 'What was the refund for?' on 100 random refund tickets")
    fam = lambda s: s.map(lambda x: FAMILY.get(x, x))
    rnd = rnd.assign(gold_f=fam(rnd.gold_reason), tool_f=fam(rnd.inferred_reason))
    reason_ok = (rnd.gold_f == rnd.tool_f).sum()
    code_ok = sum(code in rules.COMPATIBLE.get(gr, set()) for code, gr in zip(rnd.refund_reason_code, rnd.gold_reason))
    print(f"  tool's reading matches gold:            {pct(reason_ok, len(rnd))}")
    print(f"  agent's dropdown code fits gold reason: {pct(code_ok, len(rnd))}")
    print(f"  GW-OTHER in sample: {(rnd.refund_reason_code == 'GW-OTHER').sum()}, "
          f"of which gold has a specific coded reason: "
          f"{((rnd.refund_reason_code == 'GW-OTHER') & ~rnd.gold_reason.isin(['PRODUCT-FAULT', 'WRONG-ITEM'])).sum()}")
    wrong = rnd[rnd.gold_f != rnd.tool_f]
    for x in wrong.itertuples():
        print(f"   miss {x.ticket_id}: gold={x.gold_reason} tool={x.inferred_reason} (from {x.reason_source})")

    print("\n== 2d. All replacement disagreements (random + targeted)")
    for x in g[g.tool_repl != g.gold_replacement].itertuples():
        print(f"   {x.ticket_id} [{x.stratum}] gold={x.gold_replacement} tool={x.tool_repl} flag={x.replacement_issued}"
              f" | {x.agent_notes[:140]!r}")

    if a.llm:
        print("\n== 3. Local LLM (llama3.1 via Ollama) on the same notes")
        use = llm.available()
        if not use:
            print("  Ollama not reachable; scoring cached answers only.")
        rows = [(x.ticket_id, x.agent_notes, x.replacement_issued) for x in g.itertuples()]
        ops = llm.second_opinions(rows, HERE, use_model=use)
        g["llm_repl"] = g.ticket_id.map(lambda k: ops.get(k, {}).get("verdict"))
        s = g[g.llm_repl.notna()]
        secs = [ops[k]["seconds"] for k in s.ticket_id if ops[k].get("seconds")]
        print(f"  scored {len(s)} tickets; mean {sum(secs) / max(1, len(secs)):.1f} s per ticket")
        for name, part in [("random", s[s.stratum == "random"]), ("targeted", s[s.stratum == "targeted"])]:
            if len(part):
                print(f"  {name}: LLM agrees {pct((part.llm_repl == part.gold_replacement).sum(), len(part))}, "
                      f"rules agree {pct((part.tool_repl == part.gold_replacement).sum(), len(part))}")
                print(pd.crosstab(part.gold_replacement, part.llm_repl, rownames=["gold"], colnames=["llm"]).to_string())
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
