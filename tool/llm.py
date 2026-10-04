"""Optional LLM second opinion for the tickets the rules cannot settle.

Runs against a local Ollama model (default llama3.1), so ticket text never leaves
the machine and there is no per-call cost. Results are cached in llm_cache.json,
keyed by ticket_id + prompt version, so a re-run (or a machine without Ollama)
reproduces the same output.

The LLM is only asked about the review queue: notes where the rules found
contradictory statements, or where the helpdesk Y/N flag disagrees with the note.
It is never used to set money values.
"""
from __future__ import annotations

import json
import os
import urllib.request

PROMPT_VERSION = "v3"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.environ.get("VIREO_LLM_MODEL", "llama3.1")

PROMPT = """You audit customer-support tickets for an electronics brand.
Policy: a customer must never receive BOTH a refund AND a replacement/reshipment for the same order.
A refund was definitely issued on this ticket. Decide from the agent's note whether a replacement
unit, new unit, reshipment or replacement part was ALSO sent or raised for the customer.

Rules:
- "offered replacement, customer chose refund", "replacement declined", "rejected as per policy",
  "not applicable (out of stock)", "explained policy, refund only" => NOT sent.
- "new unit also going out", "fresh pair shipped", "refund + rplc", "sent as goodwill/one-time gesture",
  "dispatched a new one under RMA", "re-shipped from warehouse" => SENT.
- If the note says a unit was dispatched AND ALSO says the replacement was refused, answer "unclear".
- Abbreviations: rplc=replacement, rfnd=refund, cx=customer, rma=return authorisation.

Examples:
Note: "Full rfnd processed (1899). Replacement was offered earlier by chat team, cx opted for rfnd."
Answer: {{"replacement_sent": "no", "evidence": "cx opted for rfnd"}}
Note: "rfnd done & new unit sent as one-time gesture."
Answer: {{"replacement_sent": "yes", "evidence": "new unit sent as one-time gesture"}}
Note: "Replacement dispatched. Refunded. Replacement request rejected as per policy."
Answer: {{"replacement_sent": "unclear", "evidence": "dispatched vs rejected"}}

Now this ticket.
Agent note: "{note}"
Helpdesk replacement flag: {flag}
Answer with JSON only: {{"replacement_sent": "yes"|"no"|"unclear", "evidence": "<short quote from the note>"}}"""


def _cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "llm_cache.json")


def load_cache(cache_dir: str) -> dict:
    p = _cache_path(cache_dir)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache_dir: str, cache: dict) -> None:
    with open(_cache_path(cache_dir), "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=1, sort_keys=True)


def available() -> bool:
    try:
        urllib.request.urlopen(OLLAMA_URL.replace("/api/generate", "/api/tags"), timeout=2)
        return True
    except Exception:
        return False


def ask(note: str, flag: str, timeout: int = 180) -> dict:
    body = json.dumps({
        "model": MODEL, "stream": False, "format": "json",
        "options": {"temperature": 0, "num_predict": 80},
        "prompt": PROMPT.format(note=note.replace('"', "'").replace("\n", " / "), flag=flag),
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read())
    try:
        ans = json.loads(out["response"])
        verdict = str(ans.get("replacement_sent", "unclear")).lower()
        if verdict not in {"yes", "no", "unclear"}:
            verdict = "unclear"
        return {"verdict": verdict, "evidence": str(ans.get("evidence", ""))[:200],
                "seconds": round(out.get("total_duration", 0) / 1e9, 1)}
    except (ValueError, KeyError):
        return {"verdict": "unclear", "evidence": "unparseable model output", "seconds": 0}


def second_opinions(rows, cache_dir: str, use_model: bool, log=print) -> dict:
    """rows: iterable of (ticket_id, note, flag). Returns {ticket_id: result}."""
    cache = load_cache(cache_dir)
    results, todo = {}, []
    for tid, note, flag in rows:
        key = f"{PROMPT_VERSION}:{MODEL}:{tid}"
        if key in cache:
            results[tid] = cache[key]
        else:
            todo.append((key, tid, note, flag))
    if todo and use_model:
        log(f"  LLM: {len(todo)} uncached tickets -> {MODEL} via Ollama (about 10-20 s each on a laptop)")
        for i, (key, tid, note, flag) in enumerate(todo, 1):
            try:
                cache[key] = results[tid] = ask(note, flag)
            except Exception as e:  # network / model failure: leave for human review
                log(f"  LLM call failed on {tid}: {e}")
                break
            if i % 10 == 0:
                save_cache(cache_dir, cache)
                log(f"    {i}/{len(todo)}")
        save_cache(cache_dir, cache)
    elif todo:
        log(f"  LLM: {len(todo)} review tickets not in cache and Ollama not used -> left for human review")
    return results
