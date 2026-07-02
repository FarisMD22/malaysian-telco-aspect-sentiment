"""
Reddit scraper for Malaysian telco & broadband discussions.

Output: data/raw/reddit.csv
Run:    python scrapers/reddit.py

Setup before running:
1. Register a Reddit app: https://www.reddit.com/prefs/apps (type: "script")
2. Copy .env.example to .env and fill in credentials
"""
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

try:
    import praw
except ImportError:
    raise SystemExit("pip install praw python-dotenv")

SUBREDDITS = ["malaysia", "Bolehland"]
SEARCH_TERMS = [
    "unifi", "maxis", "celcom", "digi", "celcomdigi",
    "u mobile", "yes 5g", "yes 4g", "time fibre",
    "broadband", "fibre", "5g coverage", "4g coverage",
    "telco", "mobile data",
]

UNIFIED_COLS = ["id", "text", "rating", "date", "source", "app"]


def get_reddit():
    missing = [k for k in ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET") if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"Missing env vars: {missing}. See .env.example.")
    return praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ.get("REDDIT_USER_AGENT", "TNL6323/1.0"),
    )


def scrape_subreddit(reddit, subreddit_name: str, search_term: str, limit: int = 50) -> list[dict]:
    rows = []
    sub = reddit.subreddit(subreddit_name)
    for submission in sub.search(search_term, limit=limit):
        text = (submission.title + "\n\n" + (submission.selftext or "")).strip()
        rows.append({
            "id": submission.id, "text": text, "rating": None,
            "date": pd.to_datetime(submission.created_utc, unit="s"),
            "source": "reddit", "app": f"r/{subreddit_name}",
        })
        submission.comments.replace_more(limit=0)
        for comment in submission.comments.list()[:20]:
            rows.append({
                "id": comment.id, "text": comment.body, "rating": None,
                "date": pd.to_datetime(comment.created_utc, unit="s"),
                "source": "reddit", "app": f"r/{subreddit_name}",
            })
    return rows


def main():
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    reddit = get_reddit()

    all_rows = []
    for subreddit in SUBREDDITS:
        for term in SEARCH_TERMS:
            print(f"r/{subreddit} / '{term}' ...")
            try:
                rows = scrape_subreddit(reddit, subreddit, term, limit=50)
            except Exception as e:
                print(f"  [!] {e}")
                rows = []
            print(f"  -> {len(rows)} items")
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows, columns=UNIFIED_COLS)
    df = df.drop_duplicates(subset="id").dropna(subset=["text"])
    path = out_dir / "reddit.csv"
    df.to_csv(path, index=False)
    print(f"\nTOTAL: {len(df):,} unique items saved to {path}")


if __name__ == "__main__":
    main()
