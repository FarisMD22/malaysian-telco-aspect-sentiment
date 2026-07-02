"""
Preprocessing pipeline — merge, clean, dedupe, language-detect.

Owner: Amgad
Input:  data/raw/*.csv  (output of all 4 scrapers)
Output: data/cleaned/all_sources.csv
Run:    python src/preprocess.py
"""
import glob
import re
import pandas as pd
from pathlib import Path

try:
    from langdetect import detect, DetectorFactory, LangDetectException
    DetectorFactory.seed = 42
except ImportError:
    raise SystemExit("pip install langdetect")

MIN_WORDS = 5


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"<[^>]+>", " ", text)                         # HTML tags
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)                  # repeated chars: sooooo -> soo
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)
    df["cleaned_text"] = df["text"].apply(clean_text)
    df["word_count"] = df["cleaned_text"].str.split().str.len().fillna(0).astype(int)
    df = df[df["word_count"] >= MIN_WORDS]
    df["language"] = df["cleaned_text"].apply(detect_language)
    df = df.drop_duplicates(subset="cleaned_text")
    return df.reset_index(drop=True)


def main():
    raw_files = glob.glob("data/raw/*.csv")
    if not raw_files:
        raise SystemExit("No files in data/raw/. Run scrapers first.")
    dfs = []
    for f in raw_files:
        print(f"Loading {f}")
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"  [!] {e}")
    full = pd.concat(dfs, ignore_index=True, sort=False)
    print(f"Raw total: {len(full):,}")

    clean = preprocess_df(full)
    print(f"Clean total: {len(clean):,}")
    print(f"By source:\n{clean['source'].value_counts()}")
    print(f"By language (top 5):\n{clean['language'].value_counts().head()}")

    out_dir = Path("data/cleaned")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "all_sources.csv"
    clean.to_csv(out_path, index=False)
    # Also save as HDF5 for KR rubric (Amgad: cite both in report)
    try:
        clean.to_hdf(out_dir / "all_sources.h5", key="data", mode="w")
    except Exception as e:
        print(f"  [!] HDF5 save skipped: {e} (pip install tables)")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
