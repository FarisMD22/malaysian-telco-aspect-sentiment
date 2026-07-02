"""
Aspect-based sentiment analysis — keyword-rule extractor.

Owner: Faris
Aspects are LOCKED per plan. Keyword lists curated by the group in Week 5.
Run: python src/aspect_sa.py
"""
import re
import pandas as pd
from pathlib import Path

# LOCKED aspect taxonomy. Keyword lists can be extended, but the aspect keys
# must not change without a group discussion.
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


def extract_aspects(text: str) -> list[str]:
    """Return list of aspect keys matched in the cleaned text."""
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


def main():
    df = pd.read_csv("data/labeled/labeled_main.csv")
    df["aspects"] = df["cleaned_text"].apply(extract_aspects)
    out = Path("data/labeled/labeled_main_with_aspects.csv")
    df.to_csv(out, index=False)
    flat = [a for row in df["aspects"] for a in row]
    print("Aspect distribution across labeled set:")
    print(pd.Series(flat).value_counts())
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
