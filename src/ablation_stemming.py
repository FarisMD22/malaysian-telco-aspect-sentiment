"""
Stemming ablation.

Input:  data/labeled/labeled_main_train.csv + the three eval tiers
Output: models/ablation_stemming.csv + console table
Run:    python src/ablation_stemming.py

Stemming is kept out of the transformer pipeline (XLM-R works on SentencePiece sub-words, and no
validated Bahasa Melayu / Manglish stemmer exists; see METHODOLOGY sec 4.1). This applies Porter
stemming inside the TF-IDF + LogReg baseline only and reports stemmed vs unstemmed macro-F1 on all
three tiers, retraining each way. Porter is an English stemmer, so it stems the BM/Manglish portion
incorrectly; the ablation measures whether the vocabulary-shrinking benefit outweighs that cost.
"""
import re
import sys
from pathlib import Path

import pandas as pd
from nltk.stem import PorterStemmer
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
MODES = ["unstemmed", "stemmed"]
_TOKEN = re.compile(r"\b\w\w+\b")            # mirrors TfidfVectorizer's default token pattern
_stemmer = PorterStemmer()
_cache: dict[str, str] = {}                  # stem cache; Porter is the loop's hot spot


def stem_word(w: str) -> str:
    s = _cache.get(w)
    if s is None:
        s = _cache[w] = _stemmer.stem(w)
    return s


def transform(series: pd.Series, mode: str) -> pd.Series:
    s = series.fillna("").astype(str)
    if mode == "unstemmed":
        return s
    return s.apply(lambda x: " ".join(stem_word(t) for t in _TOKEN.findall(x.lower())))


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

    print("Porter stemming applied to the TF-IDF baseline only (English stemmer on mixed-language "
          "text; see METHODOLOGY sec 4.1). Reporting stemmed vs unstemmed macro-F1.\n")

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
    print("=== baseline macro-F1: unstemmed vs Porter-stemmed x tier ===")
    print(pivot.round(4).to_string())
    delta = (pivot.loc["stemmed"] - pivot.loc["unstemmed"]).rename("stemmed - unstemmed")
    print("\n=== delta (stemmed - unstemmed) ===")
    print(delta.round(4).to_string())

    Path("models").mkdir(exist_ok=True)
    res.to_csv("models/ablation_stemming.csv", index=False)
    print("\nSaved models/ablation_stemming.csv")


if __name__ == "__main__":
    main()
