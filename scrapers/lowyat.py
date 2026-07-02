"""
Lowyat.NET forum scraper — CROSS-DOMAIN eval source.

Owner: Rishima (original stub) · Rebuilt 2026-06-16 by Faris as Reddit insurance.
Target: ~1,000 forum posts across Malaysian telco/broadband sub-forums.
Run:   python scrapers/lowyat.py

WHY THIS EXISTS: Reddit funnelled app registration into Devvit (on-platform apps),
which gives no OAuth scraping credentials. Lowyat.NET is the cross-domain tier
instead — Malaysian users discussing the same telco/broadband brands in a forum
register (long-form, code-switched), which is exactly the domain shift we want to
measure against the Play Store training data.

ACCESS NOTE: Lowyat sits behind Cloudflare but serves plain 200 HTML to a normal
browser User-Agent (no JS challenge as of 2026-06-16), so requests + BeautifulSoup
is enough — no headless browser needed. Posts live in `div.postcolor`; quoted text
is wrapped in `div.quotetop` / `div.quotemain` and is stripped so we keep only each
author's own words. Threads paginate via `?st=<offset>` (20 posts/page).

Forum posts have no star rating, so `rating` is None (same as Reddit) — sentiment
for the eval sample comes from the manual labeling sprint, not auto-labels.
"""
import re
import time
import requests
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
}
BASE = "https://forum.lowyat.net"

# Telco/broadband sub-forums (verified live 2026-06-16).
SUBFORUMS = ["TelcoTalk", "Maxis", "NetworksandBroadband", "InternetRelated"]

UNIFIED_COLS = ["id", "text", "rating", "date", "source", "app"]

TARGET_POSTS = 1200        # stop once we have this many (need ~1,000 raw)
THREADS_PER_FORUM = 12     # top threads to crawl per sub-forum
MAX_PAGES_PER_THREAD = 8   # cap pagination so one mega-thread can't dominate
POSTS_PER_PAGE = 20        # Lowyat's fixed page size (?st= offset step)
SLEEP = 1.5                # politeness between requests

DATE_RE = re.compile(r"[A-Z][a-z]{2} \d{1,2} \d{4}, \d{1,2}:\d{2} [AP]M")


def fetch(url: str, session: requests.Session, retries: int = 3):
    for attempt in range(retries):
        try:
            r = session.get(url, headers=HEADERS, timeout=25)
            if r.status_code == 200 and "Just a moment" not in r.text:
                return BeautifulSoup(r.text, "html.parser")
            if "Just a moment" in r.text:
                print("  [!] Cloudflare JS challenge hit — needs Playwright fallback.")
                return None
        except Exception as e:
            print(f"  [!] {type(e).__name__} (attempt {attempt + 1})")
        time.sleep(SLEEP * (attempt + 1))
    return None


def thread_ids_in_forum(forum: str, session: requests.Session) -> list[str]:
    soup = fetch(f"{BASE}/{forum}", session)
    if soup is None:
        return []
    ids = []
    for a in soup.find_all("a", href=True):
        m = re.search(r"/topic/(\d+)", a["href"])
        if m and a.get_text(strip=True):
            ids.append(m.group(1))
    # preserve order, dedupe
    seen, out = set(), []
    for tid in ids:
        if tid not in seen:
            seen.add(tid)
            out.append(tid)
    return out


def posts_from_thread(topic_id: str, forum: str, session: requests.Session) -> list[dict]:
    rows = []
    for page in range(MAX_PAGES_PER_THREAD):
        st = page * POSTS_PER_PAGE
        url = f"{BASE}/topic/{topic_id}" + (f"?st={st}" if st else "")
        soup = fetch(url, session)
        if soup is None:
            break
        bodies = soup.select("div.postcolor")
        if not bodies:
            break
        # best-effort date alignment: only trust dates if they 1:1 match posts on the page
        page_dates = DATE_RE.findall(soup.get_text(" "))
        aligned = page_dates if len(page_dates) == len(bodies) else [None] * len(bodies)
        for i, body in enumerate(bodies):
            for q in body.select("div.quotetop, div.quotemain"):  # drop quoted text
                q.decompose()
            text = body.get_text(" ", strip=True)
            if not text:
                continue
            rows.append({
                "id": f"lowyat_{topic_id}_{st + i}",
                "text": text,
                "rating": None,
                "date": aligned[i],
                "source": "lowyat",
                "app": f"lowyat/{forum}",
            })
        if len(bodies) < POSTS_PER_PAGE:  # last page of the thread
            break
        time.sleep(SLEEP)
    return rows


def main():
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    all_rows = []

    for forum in SUBFORUMS:
        print(f"\n=== /{forum} ===")
        tids = thread_ids_in_forum(forum, session)[:THREADS_PER_FORUM]
        print(f"  {len(tids)} threads")
        for tid in tids:
            if len(all_rows) >= TARGET_POSTS:
                break
            rows = posts_from_thread(tid, forum, session)
            all_rows.extend(rows)
            print(f"  topic {tid}: +{len(rows)} posts (total {len(all_rows)})")
            time.sleep(SLEEP)
        if len(all_rows) >= TARGET_POSTS:
            print(f"\nReached target ({TARGET_POSTS}); stopping.")
            break

    df = pd.DataFrame(all_rows, columns=UNIFIED_COLS)
    df = df.drop_duplicates(subset="text").dropna(subset=["text"])
    path = out_dir / "lowyat.csv"
    df.to_csv(path, index=False)
    print(f"\nTOTAL: {len(df):,} unique posts saved to {path}")


if __name__ == "__main__":
    main()
