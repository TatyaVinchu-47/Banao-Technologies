# Vireo refund summary tool

Reconciles the helpdesk export, reads every refund ticket's notes, and writes a board-pack workbook:
monthly refunds by reason code, by actual reason, by team and by agent, plus the list of tickets where a
customer got a refund and a replacement.

## Run it (clean machine, about 2 minutes)

You need Python 3.10 or newer. No API keys, no network.

The data pack is not in this repo. Copy `tickets.csv`, `agents.csv`, `orders.csv` and `products.csv`
into the repo root (the folder above `tool/`) first.

```bash
cd tool
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python3 vireo_refunds.py           # reads ../tickets.csv etc., writes out/
python3 evaluate.py                # checks: arithmetic + accuracy on the labelled sample
```

If the data pack lives somewhere else: `python3 vireo_refunds.py --data /path/to/pack`.

Outputs in `tool/out/`:

| File | What it is |
|---|---|
| `board_pack.xlsx` | Read_me, Reconciliation, Monthly_overview, By_dropdown_code, By_actual_reason, Code_vs_actual_reason, By_team. Sheets prefixed `INTERNAL_` name agents or contain customer text and are not for the board. |
| `refund_tickets.csv` | One row per refund ticket with every derived field and the evidence text. |
| `summary.txt` | Headline numbers. |

## Optional: local LLM second opinion

`python3 vireo_refunds.py --llm` sends the small review queue (notes that contradict themselves or the
Y/N flag, about 12 tickets in 18 months) to a local model through [Ollama](https://ollama.com)
(`ollama pull llama3.1`). Ticket text stays on the machine. Answers are cached in `llm_cache.json`, so
runs without Ollama reproduce them. The LLM never changes a rupee figure; its opinion is shown next to
the ticket for a human to decide. `python3 evaluate.py --llm` scores it on the labelled sample.

## What it does, in order

1. **Reconcile** (`vireo_refunds.reconcile`)
   - 638 tickets appear twice (migration re-import, one `legacy_fd` + one `helpdesk` copy, identical
     except money unit). The legacy original is kept.
   - Legacy Freshdesk rows store money in paise (policy §9, Sameer's email). They are divided by 100.
     Checked: every legacy amount is a multiple of 100 and every duplicate pair agrees after dividing.
   - A bridge table goes from "export summed as-is" to the reconciled total, quarter by quarter.
2. **Attribute** refunds to the resolving agent (`agent_id`, never the name: two agents sign as "KS"),
   using the roster row valid on the ticket date.
3. **Read the text** (`rules.py`): normalise abbreviations and typos (`rplc`, `rfnd`, `refnud`...), split
   the note into clauses, then
   - decide whether a replacement / reshipment also went out (`yes` / `no` / `conflict`), handling
     negations such as "offered replacement, cx preferred refund" or "rejected as per policy";
   - infer what the refund was actually for, mapped to the policy §5 codes, plus two reasons that have
     no code (`WRONG-ITEM`, `PRODUCT-FAULT`).
4. **Cost** each confirmed refund + replacement at product unit cost + Rs 340 (policy §5 planning cost).

## Decisions (and why)

- **Month = ticket creation month (IST).** `resolved_at` is UTC for migrated tickets, `created_at` is IST in both systems.
- **All tickets with a refund raised count**, whatever their status. The helpdesk's own ~Rs 11 lakh/quarter matches counting `resolved` only; the reconciliation sheet shows the split so Finance can choose.
- **The agent's note beats the Y/N replacement flag.** In the labelled sample the flag missed 12 of 18 real cases.
- **Notes that contradict themselves are not counted**; they go to the review queue.
- **Cross-ticket matches** (refund on one ticket, replacement on another for the same order) are kept as a column only. Spot checks showed most link unrelated issues, so they are not in any total.
- **`customers.csv` is not used.** In the pack it is a 110 MB PNG image, not a CSV, and refunds do not need it.

## Files

`vireo_refunds.py` pipeline + CLI · `rules.py` text rules · `llm.py` optional Ollama layer ·
`evaluate.py` checks · `eval/gold_labels.csv` labelled sample · `llm_cache.json` cached LLM answers
