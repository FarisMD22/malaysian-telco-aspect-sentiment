"""
Aspect extractor — self-contained copy for the HuggingFace Space.

This is a faithful copy of `src/aspect_sa.py` (the LOCKED aspect taxonomy + keyword extractor),
minus the data-processing `main()`. It is duplicated here so the Space is self-contained and does
not need the rest of the repo on PYTHONPATH. If the taxonomy changes in `src/aspect_sa.py`, update
this copy too (the keys are locked, so this should be rare).
"""
import re

ASPECTS = {
    "coverage": [
        "coverage", "signal", "reception", "no service", "no signal",
        "dropped call", "area", "indoor signal", "outdoor signal",
        "liputan", "kawasan",
    ],
    "speed": [
        "speed", "slow", "fast", "lag", "lagging", "buffer", "buffering",
        "ping", "latency", "mbps", "kbps", "bandwidth", "throughput",
        "download speed", "upload speed", "laju", "lembab",
    ],
    "billing": [
        "bill", "invoice", "price", "plan", "cost", "charge", "charged",
        "expensive", "cheap", "overcharge", "refund", "promotion", "promo",
        "discount", "rm", "mahal", "murah", "bayar",
    ],
    "customer_support": [
        "support", "customer service", "cs", "agent", "helpline",
        "ticket", "complaint", "representative", "call center",
        "live chat", "rude", "helpful", "responsive", "unresponsive",
    ],
    "app_usability": [
        "app", "interface", "ui", "ux", "bug", "crash", "crashes",
        "login", "update", "dashboard", "menu", "navigation",
        "glitch", "freeze", "freezes", "error",
    ],
}


def extract_aspects(text: str) -> list:
    """Return list of aspect keys matched in the text."""
    if not isinstance(text, str):
        return []
    t = text.lower()
    matched = []
    for aspect, kws in ASPECTS.items():
        for kw in kws:
            if re.search(r"\b" + re.escape(kw) + r"\b", t):
                matched.append(aspect)
                break
    return matched


def sentences_for_aspect(text: str, aspect: str) -> str:
    """Return the sentences of `text` that mention `aspect` (joined). Used for a light
    per-aspect sentiment read in the demo: classify only the sentences about that aspect."""
    kws = ASPECTS.get(aspect, [])
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    hits = [p for p in parts if any(re.search(r"\b" + re.escape(k) + r"\b", p.lower()) for k in kws)]
    return " ".join(hits).strip() or text
