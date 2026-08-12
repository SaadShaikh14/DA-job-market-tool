"""
fetch_jobs.py
Fetches Data Analyst job postings from the Adzuna API for Indian cities
and saves them to a CSV for further analysis.

This is a LIVE / ACCUMULATING fetch: every run pulls the latest postings
and MERGES them into the existing CSV (deduped by URL) instead of
overwriting it. That means:
- Re-running this regularly keeps the dataset fresh (new postings added)
- The dataset also keeps growing richer over time (more salary data,
  a real multi-week/month posting trend) instead of resetting each run
- Old postings are kept until they naturally fall out of Adzuna's
  max_days_old window; they are not deleted from your CSV automatically

BEFORE RUNNING:
1. Sign up free at https://developer.adzuna.com/ (instant, no approval wait)
2. Copy your app_id and app_key from the dashboard
3. pip install requests pandas python-dotenv
4. Add to your .env file:
   ADZUNA_APP_ID=your_id_here
   ADZUNA_APP_KEY=your_key_here
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

APP_ID = os.getenv("ADZUNA_APP_ID")
APP_KEY = os.getenv("ADZUNA_APP_KEY")

COUNTRY = "in"
BASE_URL = f"https://api.adzuna.com/v1/api/jobs/{COUNTRY}/search"

JOB_TITLES = ["data analyst", "business analyst", "junior data analyst"]
RESULTS_PER_PAGE = 50
PAGES_PER_QUERY = 10  # up to 500 postings per title search
MAX_DAYS_OLD = 30  # widened from 5 -> better one-shot coverage; re-running
                    # regularly (e.g. weekly) keeps things fresh regardless
RAW_CSV = "da_job_postings_raw.csv"


def fetch_jobs(title, page=1, max_retries=3):
    params = {
        "app_id": APP_ID,
        "app_key": APP_KEY,
        "results_per_page": RESULTS_PER_PAGE,
        "what": title,
        "max_days_old": MAX_DAYS_OLD,
        "sort_by": "date",
        "content-type": "application/json",
    }
    url = f"{BASE_URL}/{page}"
    for attempt in range(1, max_retries + 1):
        response = requests.get(url, params=params)
        if response.status_code == 429:
            wait = 15 * attempt
            print(f"    Rate limited, waiting {wait}s...")
            time.sleep(wait)
            continue
        response.raise_for_status()
        return response.json().get("results", [])
    response.raise_for_status()
    return response.json().get("results", [])


def main():
    if not APP_ID or not APP_KEY:
        print("ERROR: ADZUNA_APP_ID / ADZUNA_APP_KEY not found. Check your .env file.")
        return

    all_jobs = []

    for title in JOB_TITLES:
        for page in range(1, PAGES_PER_QUERY + 1):
            print(f"Fetching: {title} (page {page})")
            try:
                jobs = fetch_jobs(title, page)
            except requests.exceptions.HTTPError as e:
                print(f"  Skipped ({e})")
                continue

            if not jobs:
                print("  No more results.")
                break

            fetched_at = datetime.now(timezone.utc).isoformat()
            for job in jobs:
                all_jobs.append({
                    "title": job.get("title"),
                    "company": (job.get("company") or {}).get("display_name"),
                    "location": (job.get("location") or {}).get("display_name"),
                    "salary_min": job.get("salary_min"),
                    "salary_max": job.get("salary_max"),
                    "salary_is_predicted": job.get("salary_is_predicted"),
                    "description": job.get("description"),
                    "posted_date": job.get("created"),
                    "url": job.get("redirect_url"),
                    "search_title": title,
                    "first_fetched_at": fetched_at,
                })

            time.sleep(1)

    new_df = pd.DataFrame(all_jobs)
    new_df.drop_duplicates(subset=["url"], inplace=True)

    if len(new_df) == 0:
        print("\nNo postings fetched. Existing CSV (if any) was left untouched.")
        return

    # Merge with whatever is already on disk instead of overwriting, so the
    # dataset accumulates (more salary data, a real posting-date trend) run
    # over run instead of resetting to just "the last N days" every time.
    previous_count = 0
    if os.path.exists(RAW_CSV):
        existing_df = pd.read_csv(RAW_CSV)
        previous_count = len(existing_df)
        if "first_fetched_at" not in existing_df.columns:
            existing_df["first_fetched_at"] = None
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        # Keep the FIRST time we ever saw a URL, drop later duplicate fetches
        combined.sort_values("first_fetched_at", inplace=True, na_position="last")
        combined.drop_duplicates(subset=["url"], keep="first", inplace=True)
    else:
        combined = new_df

    combined.to_csv(RAW_CSV, index=False)
    print(f"\nFetched {len(new_df)} postings this run.")
    print(f"Added {len(combined) - previous_count} genuinely new postings.")
    print(f"Dataset now has {len(combined)} total postings in {RAW_CSV} (accumulated across runs).")


if __name__ == "__main__":
    main()
