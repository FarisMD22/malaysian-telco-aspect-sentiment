"""
Trustpilot scraper for Malaysian telco brand pages.

Output: data/raw/trustpilot_<brand>.csv
Run:    python scrapers/trustpilot.py
Setup:  pip install playwright && python -m playwright install chromium

Trustpilot pages sit behind an AWS WAF JavaScript challenge, so plain requests + BeautifulSoup
returns 403. We render with headless Chromium (Playwright), which clears the challenge, then read
reviews from the embedded `__NEXT_DATA__` JSON blob rather than HTML selectors. Only three brand
pages carry Malaysian-telco reviews (Digi, Maxis, Celcom); the others have empty pages.
"""
import json
import time
import pandas as pd
from pathlib import Path
from playwright.sync_api import sync_playwright

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

# Only these three brand pages carry Malaysian-telco reviews.
TRUSTPILOT_PAGES = {
    "Digi":   "https://www.trustpilot.com/review/digi.com.my",
    "Maxis":  "https://www.trustpilot.com/review/maxis.com.my",
    "Celcom": "https://www.trustpilot.com/review/celcom.com.my",
}

UNIFIED_COLS = ["id", "text", "rating", "date", "source", "app"]


def _get_next_data(page, url: str) -> dict:
    """Render the page (clearing the WAF JS challenge) and return parsed __NEXT_DATA__."""
    page.goto(url, wait_until="domcontentloaded", timeout=45000)
    page.wait_for_selector("script#__NEXT_DATA__", state="attached", timeout=25000)
    return json.loads(page.eval_on_selector("script#__NEXT_DATA__", "el => el.textContent"))


def scrape_brand(page, base_url: str, brand: str, max_pages: int = 30) -> pd.DataFrame:
    rows = []
    seen = set()
    for page_num in range(1, max_pages + 1):
        url = base_url if page_num == 1 else f"{base_url}?page={page_num}"
        try:
            data = _get_next_data(page, url)
        except Exception as e:
            print(f"  [!] page {page_num}: {type(e).__name__}")
            break
        reviews = data["props"]["pageProps"].get("reviews", [])
        if not reviews:
            break
        new_on_page = 0
        for rv in reviews:
            rid = rv.get("id")
            if rid in seen:
                continue
            seen.add(rid)
            new_on_page += 1
            dates = rv.get("dates") or {}
            rows.append({
                "id": rid or f"{brand}_{page_num}_{len(rows)}",
                "text": (rv.get("text") or "").strip(),
                "rating": rv.get("rating"),
                "date": dates.get("publishedDate"),
                "source": "trustpilot",
                "app": brand,
            })
        if new_on_page == 0:  # pagination exhausted / looping
            break
        time.sleep(2)  # politeness
    return pd.DataFrame(rows, columns=UNIFIED_COLS)


def main():
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_context(user_agent=UA, locale="en-US").new_page()
        for brand, url in TRUSTPILOT_PAGES.items():
            print(f"Scraping {brand}...")
            df = scrape_brand(page, url, brand)
            path = out_dir / f"trustpilot_{brand}.csv"
            df.to_csv(path, index=False)
            print(f"  -> {len(df)} reviews saved to {path}")
            total += len(df)
        browser.close()
    print(f"\nTOTAL: {total:,} reviews across {len(TRUSTPILOT_PAGES)} brands")


if __name__ == "__main__":
    main()
