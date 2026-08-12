"""
test_api.py
Quick diagnostic: makes ONE call to JSearch's /search-v2 endpoint and
prints the raw response, so we can confirm the exact field names
(cursor, data, etc.) before rebuilding the full multi-page fetch script.
"""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

url = "https://jsearch.p.rapidapi.com/search-v2"
headers = {
    "X-RapidAPI-Key": RAPIDAPI_KEY,
    "X-RapidAPI-Host": "jsearch.p.rapidapi.com",
}
test_cases = [
    {"query": "developer jobs in chicago", "country": "us"},
    {"query": "data analyst jobs in Bangalore", "country": "in"},
]

for case in test_cases:
    print("=" * 50)
    print("QUERY:", case["query"], "| COUNTRY:", case["country"])
    response = requests.get(url, headers=headers, params=case)
    print("Status code:", response.status_code)
    body = response.json()
    num_jobs = len(body.get("data", {}).get("jobs", []))
    print("Jobs found:", num_jobs)
    print(json.dumps(body, indent=2)[:2000])
    print()
