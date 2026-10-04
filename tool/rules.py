"""Deterministic text rules for refund tickets.

Two questions are answered for every refund ticket, from the customer's opening
message and the agent's closing note:

1. What was the refund actually for?  (inferred reason, mapped to policy §5 codes)
2. Did the customer ALSO get a replacement / reshipment?  (policy §5 says never both)

The helpdesk notes are short, templated, abbreviated and full of typos, so text is
normalised first (abbreviation expansion + fuzzy spelling correction), then matched
clause by clause. Negated clauses ("cx opted for refund", "replacement declined")
are recognised so they do not count as a replacement.
"""
from __future__ import annotations

import difflib
import re
from functools import lru_cache

# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
ABBREV = {
    "rplc": "replacement", "rpplc": "replacement", "rfnd": "refund", "rfd": "refund",
    "rfdn": "refund", "rnfd": "refund", "cx": "customer", "pkp": "pickup", "crr": "courier",
    "ord": "order", "dlvry": "delivery", "rcvd": "received", "conf": "confirmed",
    "chk": "check", "adv": "advised", "wty": "warranty", "fw": "firmware", "bt": "bluetooth",
    "rslvd": "resolved", "fr": "full refund", "tl": "team lead", "w": "with", "pg": "gateway",
}
# Canonical words that typo'd tokens are snapped to (only tokens of length >= 5).
VOCAB = [
    "refund", "refunded", "replacement", "reverse", "reversed", "reshipped", "shipped",
    "shipment", "dispatched", "dispatch", "delivered", "delivery", "cancel", "cancelled",
    "cancellation", "payment", "duplicate", "charged", "pickup", "warranty", "received",
    "damaged", "declined", "rejected", "preferred", "opted", "battery", "coupon", "discount",
    "processed", "initiated", "credited", "returned", "approved", "promo", "charge",
    "double", "deducted", "debited", "offer",
]


@lru_cache(maxsize=None)
def _fix_token(tok: str) -> str:
    if tok in ABBREV:
        return ABBREV[tok]
    if len(tok) < 5 or tok in VOCAB:
        return tok
    m = difflib.get_close_matches(tok, VOCAB, n=1, cutoff=0.84)
    return m[0] if m else tok


def normalise(text: str) -> str:
    text = (text or "").lower().replace("&", " and ").replace("w/", "with ").replace("re-ship", "reship")
    text = re.sub(r"[a-z]+", lambda m: _fix_token(m.group(0)), text)
    return re.sub(r"[ \t]+", " ", text)


def clauses(text: str) -> list[str]:
    """Split a normalised note into clauses."""
    parts = re.split(r"\n|\||--|//|->|;|(?<=[a-z0-9\)])\.\s+|\s/\s|\b\d\.\s", text)
    return [p.strip(" .,-~") for p in parts if p and p.strip(" .,-~")]


# --------------------------------------------------------------------------- #
# Q2: was a replacement / reshipment also given?
# --------------------------------------------------------------------------- #
# A clause that describes a unit going OUT to the customer.
REPL_ACTION = re.compile(
    r"replacement (unit |strap |case )?(raised|dispatched|approved|shipped|also shipped|sent)"
    r"|replacement raised|raised replacement|also raised replacement|replacement case raised"
    r"|replacement dispatched as goodwill|refund (\+|and) replacement|replacement (\+|and) refund both"
    r"|both refund and replacement|refund and replacement both|issued refund (\+|and) replacement"
    r"|new (unit|set|one|piece|pair)\b.*(ship|sent|dispatch|going out|sending)"
    r"|(sent|sending|dispatched|shipped) a new (unit|set|one|piece|pair)"
    r"|(fresh|new) (unit|pair|set|piece) (shipped|sent|dispatched|also)"
    r"|rma for new|reshipped|rto confirmed, reshipped|new unit shipped under doa"
)
# Phrases where a replacement is mentioned but NOT given.
REPL_NEGATION = re.compile(
    r"offered|opted for|preferred|declined|did not want|not applicable|rejected|refund only"
    r"|instead|explained policy|out of stock|no replacement|status query|check(ed)? (rma|replacement)"
)
PART_ONLY = re.compile(r"replacement (strap|case)|strap dispatched|for the bud|case raised")


MENTIONS_REPL = re.compile(r"replacement|new (unit|set|one|piece|pair)|reship|fresh (unit|pair)")


def replacement_evidence(note: str) -> dict:
    """Classify the note as 'yes' | 'no' | 'conflict' for "replacement also given".

    conflict = the note both says a unit went out AND that a replacement was
    refused/declined/out of stock. Those go to human (or LLM) review.
    """
    hits, negs = [], []
    for cl in clauses(normalise(note)):
        if REPL_ACTION.search(cl) and not REPL_NEGATION.search(cl):
            hits.append(cl)
        elif MENTIONS_REPL.search(cl) and REPL_NEGATION.search(cl):
            negs.append(cl)
    verdict = "conflict" if hits and negs else ("yes" if hits else "no")
    return {
        "verdict": verdict,
        "part_only": bool(hits) and all(PART_ONLY.search(h) for h in hits),
        "evidence": " | ".join(hits + negs)[:240],
    }


# --------------------------------------------------------------------------- #
# Q1: what was the refund actually for?
# --------------------------------------------------------------------------- #
# Order matters: first family that matches the agent note wins, then the customer
# message, then the intake category. Most specific / least ambiguous first.
REASONS: list[tuple[str, re.Pattern]] = [
    ("DUP-PAYMENT", re.compile(r"charged twice|double charge|duplicate payment|payment debited"
                               r"|deducted without order|failed order after payment|debited twice"
                               r"|went to you but|no orders?\b|payment failed|double payment|money debited|payment went through"
                               r"|shows success|app shows nothing|statement disagrees|no order id|deducted twice")),
    ("CANCEL", re.compile(r"cancel|ordered by mistake|stop the shipment|change of mind")),
    ("PRICE-ADJ", re.compile(r"coupon|promo|discount|price (drop|adjust|match)|price dropped|offer vanished"
                             r"|festive offer|\d+% off|full price|cart says")),
    ("LOST-TRANSIT", re.compile(r"\bdent\b|not delivered|not received|delivery delayed|undelivered|lost"
                                r"|\brto\b|damaged in transit|transit damage|never arrived"
                                r"|shows delivered|marked delivered|out for delivery|tracking")),
    ("WRONG-ITEM", re.compile(r"incorrect product|wrong (item|variant|product|colou?r)")),
    ("WTY-BUYBACK", re.compile(r"warranty claim|repair status|rma status|service cent")),
    ("RETURN-QC-OK", re.compile(r"refund (pending|not credited|delay|status)|reverse pickup"
                                r"|pickup (not done|missed|pending)|qc|money for the return"
                                r"|return (was )?accepted|no one showed up|nobody came for the pickup|picked up the item"
                                r"|money hasn.?t come back|refund was promised")),
    ("DOA-REPL", re.compile(r"\bdoa\b|dead on arrival|received damaged|out of the box|arrived dead"
                            r"|not turning on|won.?t turn on|bricked|arrived damaged|crack|before i even switched")),
    ("PRODUCT-FAULT", re.compile(r"pair|discoverable|battery|drain|disconnect|dropout|dropping"
                                 r"|no audio|silent|one side|crackl|distortion|static|mic\b|mic "
                                 r"|charg|led|wifi|display|touch|strap|firmware|connect|losing my phone|keeps losing"
                                 r"|buzz|white screen|app crash|no power|went dark|dead")),
]
CATEGORY_FALLBACK = {
    "Billing & Payments": "DUP-PAYMENT", "Returns & Refunds": "RETURN-QC-OK",
    "Delivery & Shipping": "LOST-TRANSIT", "Warranty & Repair": "WTY-BUYBACK",
    "Audio Quality": "PRODUCT-FAULT", "Connectivity": "PRODUCT-FAULT",
    "Charging & Battery": "PRODUCT-FAULT", "App & Firmware": "PRODUCT-FAULT",
}
# Which dropdown codes are a defensible choice for each inferred reason.
COMPATIBLE = {
    "DUP-PAYMENT": {"DUP-PAYMENT"},
    "CANCEL": {"CANCEL"},
    "PRICE-ADJ": {"PRICE-ADJ"},
    "RETURN-QC-OK": {"RETURN-QC-OK"},
    "LOST-TRANSIT": {"LOST-TRANSIT"},
    "WRONG-ITEM": {"RETURN-QC-OK", "GW-OTHER"},          # no dedicated code exists
    "WTY-BUYBACK": {"WTY-BUYBACK"},
    "DOA-REPL": {"DOA-REPL"},
    "PRODUCT-FAULT": {"DOA-REPL", "WTY-BUYBACK", "GW-OTHER"},
    "UNKNOWN": set(),
}


def _first_reason(text: str) -> str | None:
    t = normalise(text)
    for code, pat in REASONS:
        if pat.search(t):
            return code
    return None


def infer_reason(customer_message: str, agent_note: str, category: str) -> tuple[str, str]:
    """Return (reason, source) where source is note | message | category | none."""
    # Strip the refund/replacement outcome clauses from the note before reading the
    # *reason*, otherwise "refund processed to source" etc. would bias the match.
    note_reason_text = " . ".join(
        cl for cl in clauses(normalise(agent_note))
        if not REPL_ACTION.search(cl) and not re.search(r"refund (approved|initiated|processed|issued|done|of)|refunded|credited full|payment reversed|reversed the payment|money back|amount (back|returned)|released the money", cl)
    )
    r = _first_reason(note_reason_text)
    if r:
        return r, "note"
    r = _first_reason(customer_message)
    if r:
        return r, "message"
    if category in CATEGORY_FALLBACK:
        return CATEGORY_FALLBACK[category], "category"
    return "UNKNOWN", "none"
