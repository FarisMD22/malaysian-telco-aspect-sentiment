"""
Finalize eval sets + compute agreement.

Input:  labeling/trustpilot_eval_sheet.csv, labeling/forum_eval_sheet.csv  (label_1 filled)
Output: data/labeled/trustpilot_eval.csv, data/labeled/forum_eval.csv, data/labeled/kappa.md
Run:    python labeling/compute_kappa.py

Single-annotator setup:
- Trustpilot reports Cohen's kappa between the human label and the star-derived proxy label.
  This is a human-vs-proxy validity check, not inter-annotator agreement, and with only ~5/60
  non-negative rows it is a low-support, directional estimate.
- Lowyat has no stars and no second annotator, so kappa is N/A (single-label eval set).
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_eval_pools import star_to_label  # reuse the star rule for validation

VALID = {"negative", "neutral", "positive"}
KAPPA_REF = 0.6  # reference threshold only, not an inter-annotator pass/fail
CLEANED = "data/cleaned/all_sources.csv"  # source for columns hidden from the blind sheet
SHEETS = {
    "trustpilot": ("labeling/trustpilot_eval_sheet.csv", "data/labeled/trustpilot_eval.csv"),
    "forum": ("labeling/forum_eval_sheet.csv", "data/labeled/forum_eval.csv"),
}
FINAL_COLS = ["id", "source", "language", "cleaned_text", "sentiment_label"]


def norm(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()


def final_label(row) -> str:
    """final_label if present, else label_1."""
    fl = str(row.get("final_label", "") or "").strip().lower()
    return fl if fl in VALID else str(row.get("label_1", "") or "").strip().lower()


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Re-join columns hidden from the blind labeling sheet (rating/cleaned_text/source/
    language) from the cleaned corpus by `id`. Only pulls columns the sheet lacks, so a
    full (wide) sheet is left untouched."""
    need = ["rating", "cleaned_text", "source", "language"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        ref = pd.read_csv(CLEANED, usecols=["id", *missing]).drop_duplicates("id")
        df = df.merge(ref, on="id", how="left")
    return df


def main() -> None:
    Path("data/labeled").mkdir(parents=True, exist_ok=True)
    report = ["# Labeling agreement (kappa.md)", "",
              "**Single annotator.** True inter-annotator Cohen's kappa requires two "
              "independent human labelers (METHODOLOGY.md sec 3.4); that is not available here. "
              "Trustpilot reports a human-vs-star **proxy** kappa instead, from a **blind** "
              "re-label (the annotator did not see the star rating or any pre-label, so the "
              "agreement is genuine, not anchored), a validity check, *not* inter-annotator "
              "agreement. Lowyat has no proxy and no second annotator, so kappa is N/A.", ""]

    for tier, (sheet_path, out_path) in SHEETS.items():
        try:
            df = pd.read_csv(sheet_path)
        except FileNotFoundError:
            print(f"  [skip] {sheet_path} not found")
            report += [f"## {tier}", "", "_sheet not found_", ""]
            continue

        df = enrich(df)  # bring back rating/cleaned_text/source/language if the sheet is slim
        df["sentiment_label"] = df.apply(final_label, axis=1)
        labeled = df[df["sentiment_label"].isin(VALID)].copy()
        n_total, n_labeled = len(df), len(labeled)
        report += [f"## {tier}", "",
                   f"- rows: {n_total} | labeled: {n_labeled}"]

        if n_labeled == 0:
            print(f"  [{tier}] no labels yet; fill label_1 in {sheet_path}")
            report += ["- **status:** not yet labeled", ""]
            continue

        dist = labeled["sentiment_label"].value_counts().to_dict()
        report.append(f"- class balance: {dist}")

        if tier == "trustpilot":
            # Star proxy derived from `rating` (hidden during blind labeling).
            proxy = labeled["rating"].apply(star_to_label)
            mask = proxy.isin(VALID)
            human = norm(labeled["sentiment_label"])[mask.values]
            proxy = norm(proxy)[mask.values]
            agree = float((human.values == proxy.values).mean())
            k = cohen_kappa_score(human, proxy)
            print(f"  [trustpilot] n={int(mask.sum())} agreement={agree:.1%} proxy-kappa={k:.3f}")
            report += [
                "- labeling: **blind** (annotator did not see star rating or pre_label)",
                f"- raw agreement (blind human vs star-proxy): {agree:.1%}",
                f"- **proxy kappa = {k:.3f}** (blind human vs star; compared against the "
                f"{KAPPA_REF} reference threshold, but **not** counted as an inter-annotator "
                "pass/fail)",
                f"- caveat: only {dist.get('positive',0)+dist.get('neutral',0)}/{n_labeled} "
                "non-negative -> proxy kappa and macro-F1 are low-support, **directional only**.",
                "",
            ]
        else:
            print(f"  [forum] n={n_labeled} single-label (kappa N/A)")
            report += ["- **kappa: N/A** (single annotator, no star proxy)", ""]

        labeled[FINAL_COLS].to_csv(out_path, index=False)
        print(f"  -> {out_path}  (n={n_labeled})")
        report.append(f"- exported: `{out_path}`")
        report.append("")

    report += ["## Optional follow-up", "",
               "A genuine consistency number is still obtainable via **test-retest**: relabel a "
               "sample days apart and report intra-annotator kappa. Not done by default.", ""]

    Path("data/labeled/kappa.md").write_text("\n".join(report), encoding="utf-8")
    print("\nWrote data/labeled/kappa.md")


if __name__ == "__main__":
    main()
