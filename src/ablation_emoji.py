"""
Emoji-handling ablation (advanced feature #3).

Owner: Faris
Input:  data/labeled/labeled_main_train.csv + the three eval tiers
Output: models/ablation_emoji.csv + console table
Run:    python src/ablation_emoji.py

What it measures
----------------
`src/preprocess.py` does NOT demojize -- raw emoji survive into `cleaned_text` and reach both
models. This ablation quantifies what an emoji-handling step would buy by RE-TRAINING the cheap
TF-IDF + LogReg baseline under three treatments and scoring each on the three eval tiers:
  * raw      -- emoji left as-is (the current pipeline)
  * demojize -- emoji -> their description words (fire, smiling_face...) via `emoji.demojize`
  * strip    -- emoji removed entirely

We ablate on the baseline (not XLM-R) because it retrains in seconds; an XLM-R variant would need
a Colab retrain per treatment for very little expected signal -- see the prevalence note below.

Honest scope note (printed at runtime): emoji are sparse and concentrated in app-store text
(~8% in-domain train, ~7% in-domain test) and nearly absent cross-tier (Trustpilot ~3%, forum 0%),
so any measurable delta lives almost entirely in the in-domain tier. A near-zero cross-tier delta
is an expected, reportable result, not a null finding to hide.
"""
import sys
from pathlib import Path

import emoji
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
TRAIN_PATH = "data/labeled/labeled_main_train.csv"
TIERS = {
    "Play Store (in-domain)": "data/labeled/labeled_main_test.csv",
    "Trustpilot (cross-platform)": "data/labeled/trustpilot_eval.csv",
    "Reddit/Lowyat (cross-domain)": "data/labeled/forum_eval.csv",
}
MODES = ["raw", "demojize", "strip"]


def has_emoji(s: str) -> bool:
    return emoji.emoji_count(s) > 0


def transform(series: pd.Series, mode: str) -> pd.Series:
    s = series.fillna("").astype(str)
    if mode == "raw":
        return s
    if mode == "strip":
        return s.apply(lambda x: emoji.replace_emoji(x, ""))
    if mode == "demojize":
        # description words become separate tokens (drop the :_: scaffolding TF-IDF would fuse)
        return s.apply(lambda x: emoji.demojize(x, delimiters=(" ", " ")).replace("_", " "))
    raise ValueError(mode)


def train_baseline(X, y):
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(max_iter=2000)),
    ])
    pipe.fit(X, y)
    return pipe


def main():
    try:
        train = pd.read_csv(TRAIN_PATH)
        tiers = {n: pd.read_csv(p) for n, p in TIERS.items()}
    except FileNotFoundError as e:
        sys.exit(f"missing input: {e}")

    # Prevalence report (the honest-scope context).
    print("emoji prevalence (rows containing >=1 emoji):")
    print(f"  {'train':28s} {train['cleaned_text'].apply(has_emoji).mean()*100:5.1f}%  (n={len(train)})")
    for name, df in tiers.items():
        share = df["cleaned_text"].apply(has_emoji).mean() * 100
        print(f"  {name:28s} {share:5.1f}%  (n={len(df)})")
    print()

    y_train = train["sentiment_label"].map(LABEL_MAP).astype(int)
    rows = []
    for mode in MODES:
        model = train_baseline(transform(train["cleaned_text"], mode), y_train)
        for name, df in tiers.items():
            X = transform(df["cleaned_text"], mode)
            y_true = df["sentiment_label"].map(LABEL_MAP).astype(int)
            y_pred = model.predict(X)
            rows.append({
                "mode": mode, "tier": name,
                "accuracy": accuracy_score(y_true, y_pred),
                "macro_f1": f1_score(y_true, y_pred, average="macro"),
            })

    res = pd.DataFrame(rows)
    pivot = res.pivot(index="mode", columns="tier", values="macro_f1").reindex(MODES)
    print("=== baseline macro-F1 by emoji treatment x tier ===")
    print(pivot.round(4).to_string())

    # Delta vs the current pipeline (raw).
    delta = (pivot - pivot.loc["raw"]).drop(index="raw")
    print("\n=== delta vs raw (current pipeline) ===")
    print(delta.round(4).to_string())

    Path("models").mkdir(exist_ok=True)
    res.to_csv("models/ablation_emoji.csv", index=False)
    print("\nSaved models/ablation_emoji.csv")


if __name__ == "__main__":
    main()
