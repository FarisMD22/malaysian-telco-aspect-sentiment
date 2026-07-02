"""
TF-IDF + Logistic Regression / Linear SVM baseline.

Input:  data/labeled/labeled_main.csv
Output: models/baseline_lr.pkl, models/baseline_svm.pkl
Run:    python src/baseline.py
"""
import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
LABEL_INV = {v: k for k, v in LABEL_MAP.items()}


def load(path="data/labeled/labeled_main.csv"):
    df = pd.read_csv(path)
    assert "sentiment_label" in df.columns, "labeled_main.csv must have a 'sentiment_label' column"
    df["label_id"] = df["sentiment_label"].map(LABEL_MAP)
    return df.dropna(subset=["label_id"])


def train(df, model_cls, name):
    X = df["cleaned_text"].fillna("")
    y = df["label_id"].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)),
        ("clf", model_cls(max_iter=2000)),
    ])
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    print(f"\n=== {name} ===")
    print(classification_report(y_te, y_pred, target_names=list(LABEL_MAP.keys()), zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_te, y_pred))
    print(f"Macro F1: {f1_score(y_te, y_pred, average='macro'):.3f}")
    return pipe


def main():
    df = load()
    print(f"Loaded {len(df)} labeled samples.")
    print(f"Class distribution:\n{df['sentiment_label'].value_counts()}")
    lr = train(df, LogisticRegression, "Logistic Regression")
    svm = train(df, LinearSVC, "Linear SVM")
    Path("models").mkdir(exist_ok=True)
    joblib.dump(lr, "models/baseline_lr.pkl")
    joblib.dump(svm, "models/baseline_svm.pkl")
    print("\nSaved: models/baseline_lr.pkl, models/baseline_svm.pkl")


if __name__ == "__main__":
    main()
