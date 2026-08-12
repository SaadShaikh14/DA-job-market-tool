"""
test_active_jobs.py
Quick diagnostic: one call to the Active Jobs DB API to confirm the exact
response field names before building the full multi-city fetch script.
"""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

url = "https://active-jobs-db.p.rapidapi.com/active-ats"
headers = {
    "x-rapidapi-host": "active-jobs-db.p.rapidapi.com",
    "x-rapidapi-key": RAPIDAPI_KEY,
}
params = {
    "time_frame": "7d",
    "limit": "10",
    "offset": "0",
    "description_format": "text",
    "title": '"Data Analyst"',
    "location": '"India"',
}

response = requests.get(url, headers=headers, params=params)
print("Status code:", response.status_code)
print()
print(json.dumps(response.json(), indent=2)[:3000])
