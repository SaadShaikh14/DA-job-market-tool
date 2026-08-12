"""
fetch_full_descriptions.py
Fills skill-extraction gaps by scraping page text for jobs with no
skills matched yet. Uses parallel requests and stops automatically
after a fixed time budget (~3 minutes).
Safe to re-run: already-filled rows are skipped next time, so running
this again later will gradually cover more postings.
"""

import ast
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

SKILLS = [
    'Python', 'SQL', 'Excel', 'Power BI', 'Tableau', 'R', 'SAS', 'SPSS',
    'VBA', 'Alteryx', 'Looker', 'Qlik', 'MySQL', 'PostgreSQL', 'MongoDB',
    'NoSQL', 'Big Data', 'Hadoop', 'Spark', 'AWS', 'Azure', 'GCP',
    'Google Cloud', 'Machine Learning', 'Statistics', 'A/B Testing',
    'Data Visualization', 'ETL', 'Data Warehousing', 'Data Modeling',
    'Git', 'JIRA', 'Snowflake', 'DAX', 'Power Query', 'Google Sheets',
    'Google Analytics',
]

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0 Safari/537.36'
}

TIME_BUDGET_SECONDS = 165  # hard stop, safely under 3 minutes
MAX_WORKERS = 20
SAMPLE_SIZE = 300


def find_skills(text):
    found = []
    text_lower = text.lower()
    for skill in SKILLS:
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text_lower):
            found.append(skill)
    return found


def get_page_text(url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=6)
        soup = BeautifulSoup(resp.text, 'html.parser')
        for tag in soup(['script', 'style']):
            tag.decompose()
        return soup.get_text(separator=' ', strip=True)
    except Exception:
        return ''


def process_row(idx, url):
    text = get_page_text(url)
    return idx, find_skills(text)


def main():
    df = pd.read_csv('da_job_postings_clean.csv')
    df['skills_matched'] = df['skills_matched'].apply(ast.literal_eval)

    gap_rows = df[df['num_skills'] == 0].head(SAMPLE_SIZE)
    print(f"Attempting {len(gap_rows)} postings in parallel (~3 min budget)...")

    start = time.time()
    updated = 0
    processed = 0

    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    futures = {
        executor.submit(process_row, idx, row['url']): idx
        for idx, row in gap_rows.iterrows()
    }
    try:
        for future in as_completed(futures):
            if time.time() - start > TIME_BUDGET_SECONDS:
                print("Time budget reached, stopping early.")
                break
            idx = futures[future]
            try:
                _, new_skills = future.result()
            except Exception:
                new_skills = []
            processed += 1
            if new_skills:
                df.at[idx, 'skills_matched'] = new_skills
                df.at[idx, 'num_skills'] = len(new_skills)
                df.at[idx, 'skills_matched_str'] = ', '.join(new_skills)
                updated += 1
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    df.to_csv('da_job_postings_clean.csv', index=False)
    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.0f}s. Processed {processed}/{len(gap_rows)}, updated {updated} with new skills.")
    print("Rows with at least 1 skill matched (overall):", (df['num_skills'] > 0).sum())


if __name__ == "__main__":
    main()
