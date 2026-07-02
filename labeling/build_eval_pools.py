"""
Build the two eval-set labeling sheets (B2).

Owner: Faris / Rain (labeling coordinator)
Input:  data/cleaned/all_sources.csv  (output of src/preprocess.py)
Output: labeling/trustpilot_eval_sheet.csv, labeling/forum_eval_sheet.csv
Run:    python labeling/build_eval_pools.py

Produces 60-row sheets in the labeling_template.csv schema PLUS appended `rating` and
`cleaned_text` columns (rating drives the Trustpilot proxy kappa, so it is kept visible).
A human then fills `label_1` per labeling/rubric.md; compute_kappa.py consumes the result.

Single-annotator reality: assigned_to_1 = Faris; the `_2` columns stay empty.
"""
import sys
from pathlib import Path

import pandas as pd

SEED = 42
POOL_SIZE = 60
CLEANED = "data/cleaned/all_sources.csv"
OUT_DIR = Path("labeling")

# labeling_template.csv schema, then explicit extras (rating needed for proxy kappa).
TEMPLATE_COLS = [
    "id", "text", "source", "language", "pre_label",
    "assigned_to_1", "label_1", "assigned_to_2", "label_2",
    "final_label", "disagreement", "notes",
]
EXTRA_COLS = ["rating", "cleaned_text"]
# Blind/slim sheet: only what a human needs. Hides pre_label + rating so labeling is unbiased
# (the star number is the same hint as pre_label). compute_kappa.py re-joins the rest by `id`.
SLIM_COLS = ["id", "text", "label_1", "notes"]


def star_to_label(rating) -> str:
    """Star auto-label rule (METHODOLOGY 3.2): 1-2 neg, 3 neu, 4-5 pos. '' if no star."""
    if pd.isna(rating):
        return ""
    r = int(rating)
    if r <= 2:
        return "negative"
    if r == 3:
        return "neutral"
    return "positive"


def make_sheet(pool: pd.DataFrame, pre_label: pd.Series, annotator: str = "Faris",
               blind: bool = False) -> pd.DataFrame:
    out = pd.DataFrame()
    out["id"] = pool["id"].values
    out["text"] = pool["text"].values
    out["source"] = pool["source"].values
    out["language"] = pool["language"].values
    out["pre_label"] = pre_label.values
    out["assigned_to_1"] = annotator
    out["label_1"] = ""
    out["assigned_to_2"] = ""
    out["label_2"] = ""
    out["final_label"] = ""
    out["disagreement"] = ""
    out["notes"] = ""
    out["rating"] = pool["rating"].values
    out["cleaned_text"] = pool["cleaned_text"].values
    return out[SLIM_COLS] if blind else out[TEMPLATE_COLS + EXTRA_COLS]


def build_trustpilot(df: pd.DataFrame, blind: bool = False) -> pd.DataFrame:
    """All non-negative rows (scarce) + random negatives to reach POOL_SIZE.

    NOT meaningful class balance -- the whole tier holds only ~5 non-negative items. This
    just avoids a 100%-negative pool; proxy kappa here is a low-support, directional estimate.
    """
    tp = df[df["source"] == "trustpilot"].copy()
    tp["star_label"] = tp["rating"].apply(star_to_label)
    nonneg = tp[tp["star_label"].isin(["neutral", "positive"])]
    neg = tp[tp["star_label"] == "negative"]
    n_neg = max(0, POOL_SIZE - len(nonneg))
    neg_sample = neg.sample(n=min(n_neg, len(neg)), random_state=SEED)
    pool = pd.concat([nonneg, neg_sample]).sample(frac=1, random_state=SEED)  # shuffle order
    print(f"  Trustpilot: {len(nonneg)} non-negative + {len(neg_sample)} negative = {len(pool)}")
    return make_sheet(pool, pool["star_label"], blind=blind)


def build_lowyat(df: pd.DataFrame, blind: bool = False) -> pd.DataFrame:
    """Mostly-random sample. We do NOT stratify on `language`: langdetect marks the vast
    majority `en` yet much of it is Manglish/BM (it also conflates BM with `id`), so the tag
    is unreliable -- random sampling already captures code-switching without biasing the
    distribution. pre_label stays blank: no model pre-labeling (would make the test circular).
    """
    lw = df[df["source"] == "lowyat"].copy()
    pool = lw.sample(n=min(POOL_SIZE, len(lw)), random_state=SEED)
    print(f"  Lowyat: {len(pool)} random rows (pre_label blank)")
    return make_sheet(pool, pd.Series([""] * len(pool)), blind=blind)


def main() -> None:
    try:
        df = pd.read_csv(CLEANED)
    except FileNotFoundError:
        sys.exit(f"{CLEANED} not found -- run `python src/preprocess.py` first.")

    OUT_DIR.mkdir(exist_ok=True)

    # Trustpilot: BLIND slim sheet -- annotator must not see star rating/pre_label, so the
    # proxy kappa reflects genuine agreement rather than anchoring on the hint.
    tp_path = OUT_DIR / "trustpilot_eval_sheet.csv"
    build_trustpilot(df, blind=True).to_csv(tp_path, index=False)
    print(f"  -> {tp_path}  (blind; columns = {SLIM_COLS})")

    # Forum: built once then hand-labeled -- never clobber an existing (possibly labeled) sheet.
    fm_path = OUT_DIR / "forum_eval_sheet.csv"
    if fm_path.exists():
        print(f"  [keep] {fm_path} exists -- not overwriting (preserves labels)")
    else:
        build_lowyat(df, blind=True).to_csv(fm_path, index=False)
        print(f"  -> {fm_path}  (blind; columns = {SLIM_COLS})")

    print("\nNext: fill `label_1` per labeling/rubric.md, then run "
          "`python labeling/compute_kappa.py`.")


if __name__ == "__main__":
    main()
