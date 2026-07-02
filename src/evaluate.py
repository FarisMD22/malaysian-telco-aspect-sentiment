"""
Three-way evaluation: in-domain, cross-platform, cross-domain.

Produces the core comparison table across the three eval tiers.
Run: python src/evaluate.py
"""
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}


def evaluate(preds_fn, df: pd.DataFrame, set_name: str, model_name: str) -> dict:
    y_true = df["sentiment_label"].map(LABEL_MAP).astype(int).tolist()
    X = df["cleaned_text"].fillna("").tolist()
    y_pred = preds_fn(X)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"[{model_name} | {set_name}]  acc={acc:.3f}  macro-F1={f1:.3f}")
    return {"set": set_name, "model": model_name, "accuracy": acc, "macro_f1": f1}


def baseline_preds_fn(pipeline):
    return lambda X: pipeline.predict(X).tolist()


def xlmr_preds_fn(model_path):
    from transformers import pipeline
    clf = pipeline("text-classification", model=model_path, tokenizer=model_path,
                   truncation=True, max_length=128, device=-1)

    def _f(X):
        out = clf(X)
        return [LABEL_MAP[r["label"]] for r in out]
    return _f


def main():
    Path("models").mkdir(exist_ok=True)

    # Load evaluation sets
    # In-domain = held-out split of labeled_main (assumed to be in labeled_main_test.csv)
    # If you didn't split separately, just re-split here with same seed as baseline.py
    sets = {}
    paths = {
        "Play Store (in-domain)": "data/labeled/labeled_main_test.csv",
        "Trustpilot (cross-platform)": "data/labeled/trustpilot_eval.csv",
        "Reddit/Lowyat (cross-domain)": "data/labeled/forum_eval.csv",
    }
    for name, path in paths.items():
        try:
            sets[name] = pd.read_csv(path)
        except FileNotFoundError:
            print(f"  [skip] {path} not found")

    if not sets:
        raise SystemExit("No eval sets found. Run the labeling sprint first.")

    results = []
    baseline = joblib.load("models/baseline_lr.pkl")
    for name, df in sets.items():
        results.append(evaluate(baseline_preds_fn(baseline), df, name, "LogReg"))

    try:
        xlmr_fn = xlmr_preds_fn("models/xlmr_final")
        for name, df in sets.items():
            results.append(evaluate(xlmr_fn, df, name, "XLM-R"))
    except Exception as e:
        print(f"  [skip XLM-R] {e}")

    out_df = pd.DataFrame(results)
    print("\n=== Final table ===")
    print(out_df.pivot(index="model", columns="set", values="macro_f1"))
    out_df.to_csv("models/eval_results.csv", index=False)
    print("\nSaved models/eval_results.csv")


if __name__ == "__main__":
    main()
