"""
Google Play Store scraper for Malaysian telco & broadband apps.

Output: data/raw/play_store_<app>.csv (7 apps)
Run:    python scrapers/play_store.py
"""
import pandas as pd
import time
from pathlib import Path
from google_play_scraper import reviews, Sort

# App IDs for the 7 Malaysian telco/broadband self-care apps (country=my).
TELCO_APPS = {
    "MyCelcomDigi": "com.celcomdigi.selfcare",
    "MyMaxis":       "com.maxis.mymaxis",
    "Hotlink":       "my.com.maxis.hotlink.production",
    "MyUMobile":     "com.omesti.myumobile",
    "Yes5G":         "my.yes.yes4g",
    "MyUnifi":       "my.myunifi",
    "TIMESelfCare":  "my.com.time.homefibre.app",
}

UNIFIED_COLS = ["id", "text", "rating", "date", "source", "app"]


def scrape_app(app_id: str, app_name: str, max_reviews: int = 5000) -> pd.DataFrame:
    """Paginate through Play Store reviews for one app."""
    all_reviews = []
    token = None
    while len(all_reviews) < max_reviews:
        try:
            result, token = reviews(
                app_id, lang="en", country="my",
                sort=Sort.NEWEST, count=200,
                continuation_token=token,
            )
        except Exception as e:
            print(f"  [!] {e}. Retrying in 5s...")
            time.sleep(5)
            continue
        if not result:
            break
        all_reviews.extend(result)
        if token is None:
            break
        time.sleep(0.5)  # be polite

    if not all_reviews:
        return pd.DataFrame(columns=UNIFIED_COLS)

    df = pd.DataFrame(all_reviews)
    df = df.rename(columns={"reviewId": "id", "content": "text", "score": "rating", "at": "date"})
    df["source"] = "play_store"
    df["app"] = app_name
    return df[UNIFIED_COLS].copy()


def main():
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for app_name, app_id in TELCO_APPS.items():
        print(f"Scraping {app_name} ({app_id})...")
        df = scrape_app(app_id, app_name, max_reviews=5000)
        path = out_dir / f"play_store_{app_name}.csv"
        df.to_csv(path, index=False)
        print(f"  -> {len(df)} reviews saved to {path}")
        total += len(df)
    print(f"\nTOTAL: {total:,} reviews across {len(TELCO_APPS)} apps")


if __name__ == "__main__":
    main()
