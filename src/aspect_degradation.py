"""
Aspect-conditioned cross-domain degradation (see EXPERIMENT_aspect_degradation.md).

A thin analysis layer over the evaluator and aspect rules; it duplicates no model logic.
For each (model, tier, aspect) it measures accuracy on the subset of that tier's rows that
mention the aspect, showing whether the in-domain -> cross-platform -> cross-domain degradation
is uniform across aspects or whether some transfer while others collapse. Always writes a valid
(header-at-minimum) CSV and does not crash when models or eval tiers are missing.

Run: python src/aspect_degradation.py   (or: python -m src.aspect_degradation)
"""
import math
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# Reuse locked aspect rules + the evaluator's label map and prediction wrappers.
# Support both `python src/aspect_degradation.py` and `python -m src.aspect_degradation`.
try:
    from src.aspect_sa import ASPECTS, extract_aspects
    from src.evaluate import LABEL_MAP, baseline_preds_fn, xlmr_preds_fn
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from aspect_sa import ASPECTS, extract_aspects
    from evaluate import LABEL_MAP, baseline_preds_fn, xlmr_preds_fn

# --- Configuration (shared constants; heatmap reuses TIERS for column ordering) ---
MIN_SUPPORT = 8  # aspect slices below this are flagged low_support and excluded from the headline claim
OVERALL = "__overall__"
CSV_COLS = ["model", "tier", "aspect", "n", "accuracy", "ci_low", "ci_high", "macro_f1", "low_support"]

# Eval tiers in degradation order: in-domain -> cross-platform -> cross-domain.
# Paths mirror src/evaluate.py.
TIERS = [
    ("Play Store (in-domain)", "data/labeled/labeled_main_test.csv"),
    ("Trustpilot (cross-platform)", "data/labeled/trustpilot_eval.csv"),
    ("Lowyat (cross-domain)", "data/labeled/forum_eval.csv"),
]

BASELINE_PATH = "models/baseline_lr.pkl"
XLMR_PATH = "models/xlmr_final"
OUT_CSV = "models/aspect_degradation.csv"


def wilson_ci(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% interval for a proportion. Pure arithmetic, no scipy."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = correct / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def tag_aspects(df: pd.DataFrame) -> pd.DataFrame:
    """Always (re)compute the aspects column from cleaned_text.

    Never trust a pre-existing `aspects` column: CSVs store Python lists as strings
    like "[]", which would silently break the membership masks below.
    """
    df = df.copy()
    df["aspects"] = df["cleaned_text"].fillna("").apply(extract_aspects)
    return df


def per_aspect_scores(df: pd.DataFrame, y_pred, tier: str, model: str) -> list[dict]:
    """Long-format rows: one __overall__ row + one row per locked aspect."""
    y_true = df["sentiment_label"].map(LABEL_MAP).astype(int).to_numpy()
    y_pred = pd.Series(y_pred, index=df.index).astype(int).to_numpy()
    rows = []

    # __overall__ row: accuracy + macro-F1. macro-F1 uses average="macro" to match
    # evaluate.py so the two scripts reconcile.
    n = len(df)
    correct = int((y_true == y_pred).sum())
    acc = accuracy_score(y_true, y_pred) if n else float("nan")
    macro_f1 = f1_score(y_true, y_pred, average="macro") if n else float("nan")
    lo, hi = wilson_ci(correct, n)
    rows.append({
        "model": model, "tier": tier, "aspect": OVERALL, "n": n,
        "accuracy": acc, "ci_low": lo, "ci_high": hi,
        "macro_f1": macro_f1, "low_support": n < MIN_SUPPORT,
    })

    # Per-aspect slices; subsets intentionally overlap (a row may match several aspects).
    # macro_f1 is NaN here: small slices rarely contain all three classes.
    for aspect in ASPECTS:
        mask = df["aspects"].apply(lambda a: aspect in a).to_numpy()
        an = int(mask.sum())
        if an == 0:
            acc_a, lo_a, hi_a = float("nan"), float("nan"), float("nan")
        else:
            correct_a = int((y_true[mask] == y_pred[mask]).sum())
            acc_a = correct_a / an
            lo_a, hi_a = wilson_ci(correct_a, an)
        rows.append({
            "model": model, "tier": tier, "aspect": aspect, "n": an,
            "accuracy": acc_a, "ci_low": lo_a, "ci_high": hi_a,
            "macro_f1": float("nan"), "low_support": an < MIN_SUPPORT,
        })
    return rows


def _safe_slug(model_name: str) -> str:
    """Filesystem-safe model slug for filenames (XLM-R -> XLM_R)."""
    return model_name.replace("-", "_").replace("/", "_").replace(" ", "_")


def make_heatmap(df_results: pd.DataFrame, model_name: str, out_path: str) -> None:
    """Aspect x tier accuracy heatmap for one model. __overall__ is excluded so it
    cannot dilute the per-aspect visual story. Best-effort: skips if matplotlib absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:  # matplotlib is an optional dependency
        print(f"  [heatmap skipped for {model_name}] matplotlib unavailable: {e}")
        return

    sub = df_results[(df_results["model"] == model_name) & (df_results["aspect"] != OVERALL)]
    if sub.empty:
        print(f"  [heatmap skipped for {model_name}] no aspect rows")
        return

    aspects = list(ASPECTS.keys())                       # fixed row order
    tiers = [t for t, _ in TIERS if t in set(sub["tier"])]  # column order from TIERS
    acc = np.full((len(aspects), len(tiers)), np.nan)
    note = [["" for _ in tiers] for _ in aspects]
    for i, asp in enumerate(aspects):
        for j, tier in enumerate(tiers):
            cell = sub[(sub["aspect"] == asp) & (sub["tier"] == tier)]
            if cell.empty:
                continue
            r = cell.iloc[0]
            if r["low_support"] or pd.isna(r["accuracy"]):
                # Leave acc[i, j] as NaN so the cell is masked and renders grey
                # (set_bad below); low-support cells should not carry a colour.
                note[i][j] = f"n/a\nn={int(r['n'])}"
            else:
                acc[i, j] = r["accuracy"]
                note[i][j] = f"{r['accuracy']:.2f}\nn={int(r['n'])}"

    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="lightgrey")               # masked low-support / no-data cells
    fig, ax = plt.subplots(figsize=(1.8 * len(tiers) + 2, 0.9 * len(aspects) + 2))
    im = ax.imshow(np.ma.masked_invalid(acc), cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(tiers)), [t.split(" (")[0] for t in tiers], rotation=20, ha="right")
    ax.set_yticks(range(len(aspects)), aspects)
    for i in range(len(aspects)):
        for j in range(len(tiers)):
            ax.text(j, i, note[i][j], ha="center", va="center", fontsize=8)
    ax.set_title(f"Aspect-conditioned accuracy by tier - {model_name}")
    fig.colorbar(im, ax=ax, label="accuracy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def main() -> None:
    Path("models").mkdir(exist_ok=True)

    # 1. Load whatever eval tiers exist; tag aspects on each.
    tiers = []
    for name, path in TIERS:
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            print(f"  [skip tier] {path} not found")
            continue
        tiers.append((name, tag_aspects(df)))

    # 2. Load whatever models exist. Graceful-skip is a property of THIS script only
    #    (evaluate.py itself still crashes on a missing baseline).
    models = []
    try:
        baseline = joblib.load(BASELINE_PATH)
        models.append(("LogReg", baseline_preds_fn(baseline)))
    except (FileNotFoundError, OSError) as e:
        print(f"  [skip model] LogReg ({BASELINE_PATH}): {e}")

    if Path(XLMR_PATH).exists():
        try:
            models.append(("XLM-R", xlmr_preds_fn(XLMR_PATH)))
        except Exception as e:  # transformers/model load is best-effort
            print(f"  [skip model] XLM-R ({XLMR_PATH}): {e}")
    else:
        print(f"  [skip model] XLM-R ({XLMR_PATH}): not found")

    # 3. Score every available (model x tier). Empty when nothing is available.
    results = []
    for model_name, preds_fn in models:
        for tier_name, df in tiers:
            X = df["cleaned_text"].fillna("").tolist()
            y_pred = preds_fn(X)
            results.extend(per_aspect_scores(df, y_pred, tier_name, model_name))

    # 4. Always write a valid CSV (header-only if nothing could be scored).
    out_df = pd.DataFrame(results, columns=CSV_COLS)
    out_df.to_csv(OUT_CSV, index=False)
    if out_df.empty:
        print(f"\nNo (model x tier) pairs available; wrote header-only {OUT_CSV}.")
        print("Train models/baseline_lr.pkl and/or add eval sets, then rerun.")
        return

    # 5. Console pivot + per-model heatmaps.
    for model_name in out_df["model"].unique():
        sub = out_df[out_df["model"] == model_name]
        print(f"\n=== {model_name}: accuracy by aspect x tier ===")
        print(sub.pivot(index="aspect", columns="tier", values="accuracy").round(3))
        make_heatmap(out_df, model_name, f"models/aspect_degradation_{_safe_slug(model_name)}.png")

    print(f"\nSaved {OUT_CSV}")


if __name__ == "__main__":
    main()
