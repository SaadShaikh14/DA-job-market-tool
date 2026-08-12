# Project: Data Analyst Job Market Intelligence Tool

## What this project is
A portfolio project for a Data Analyst job search. It's a Streamlit + RAG
application that analyzes live Data Analyst job postings across India to
surface skill demand trends and salary insights, and lets a user ask
natural-language questions about the job market (e.g. "what skills do
Mumbai companies want for DA freshers?") with answers grounded in real
postings.

## Why this project / positioning notes
- Built by a final-year BIT student targeting Data Analyst (not Data
  Scientist / ML Engineer) roles. Has received feedback that the profile
  can look "overqualified" for DA roles.
- IMPORTANT: the RAG/GenAI piece must stay a *feature*, not the headline.
  Resume framing should be analytics-first, e.g. "Built a job-market
  analytics tool... added a RAG-based Q&A layer" — not "Built an AI
  chatbot."
- The build was deliberately differentiated from an earlier portfolio
  project ("Review Insight" — scraping + CSV + Logistic Regression +
  Streamlit) so the two projects don't look like the same template:
  - Live API integration instead of web scraping (for data collection)
  - RAG (embeddings + retrieval + LLM) instead of a classic ML classifier
  - Considering PostgreSQL instead of flat CSV (not yet implemented —
    still using CSV at this stage)
  - Considering FastAPI/Gradio instead of pure Streamlit (not yet decided)

## Tech stack (so far)
Python, pandas, requests, BeautifulSoup, python-dotenv. Planned for later
steps: sentence-transformers, Chroma or FAISS, an LLM API, Streamlit,
Power BI (for a polished dashboard alongside the Python charts).

## Project folder
`C:\Users\ss496\OneDrive\Desktop\job-market-project` (Windows machine)

## Data source journey (context in case it comes up)
Several APIs were tried before settling on one:
1. **JSearch (RapidAPI)** — signed up, but the `/search` endpoint was
   deprecated (404), and the replacement `/search-v2` endpoint returned
   200 OK with zero results even for RapidAPI's own official example
   query, confirmed via their own playground. Concluded this was a
   provider-side bug/outage. Abandoned.
2. **Active Jobs DB (RapidAPI, by Fantastic Jobs)** — got this working
   (endpoint `/active-ats`), and it returned genuinely great data
   including AI-pre-extracted fields (`ai_key_skills`, `ai_experience_level`,
   `ai_work_arrangement`, salary fields). Got 190 real postings before
   hitting a 429. Checked the Pricing page and found the free Basic plan
   has a hard limit of only **25 requests/month** — already exhausted
   from testing, won't reset until next billing cycle. Shelved (not
   deleted — could be revisited next month for a richer data source since
   its AI-extracted fields are better quality than manual skill matching).
3. **Kaggle dataset (muhammetakkurt/naukri-jobs-dataset)** — downloaded
   as a fallback (13,691 rows, has a ready-made `tagsAndSkills` column),
   but the user preferred to keep pursuing live data instead, so this
   was not used in the end. Still sitting in the project folder if useful
   as a supplementary/backup dataset later.
4. **Adzuna API — this is what's actually being used.** Free tier: 1,000
   calls/month (self-serve signup at developer.adzuna.com, instant, no
   approval wait). Covers India (`country=in`). Successfully pulled
   **995 live job postings** across India for the search terms "data
   analyst", "business analyst", "junior data analyst" (broadened from
   an original 5-city plan to all-India per user preference for maximum
   coverage). Credentials are in `.env` as `ADZUNA_APP_ID` /
   `ADZUNA_APP_KEY`.

## Files currently in the project folder
- `.env` — API credentials (`ADZUNA_APP_ID`, `ADZUNA_APP_KEY`; may also
  have a leftover unused `RAPIDAPI_KEY` from the earlier attempts)
- `requirements.txt` — requests, pandas, python-dotenv, beautifulsoup4
- `fetch_jobs.py` — pulls live postings from the Adzuna API (loops over
  3 title searches, up to 10 pages each, ~50 results/page). Has
  retry-with-backoff on HTTP 429 and a safety check that never overwrites
  existing saved data with an empty result. **Run this to (re)collect
  raw data.** Output: `da_job_postings_raw.csv`.
- `da_job_postings_raw.csv` — 995 rows. Columns: `title`, `company`,
  `location`, `salary_min`, `salary_max`, `salary_is_predicted`,
  `description` (Adzuna only gives a **truncated excerpt**, not the full
  JD — this matters for the next file), `posted_date`, `url`,
  `search_title`.
- `clean_and_extract.py` — scans `title` + `description` for ~35
  predefined Data Analyst skills (Python, SQL, Excel, Power BI, Tableau,
  R, etc.) via regex word-boundary matching, and guesses an experience
  level (Fresher/Junior/Senior/Manager+/Not specified) from the title.
  Output: `da_job_postings_clean.csv`. First run only matched skills for
  231/995 rows (~23%) because of the truncated-description limitation
  above.
- `da_job_postings_clean.csv` — same as raw, plus: `skills_matched`
  (Python list stored as string — parse with `ast.literal_eval` when
  reloading), `num_skills`, `skills_matched_str` (comma-joined),
  `experience_level_guess`, `location_clean`.
- `fetch_full_descriptions.py` — gap-filling step: for rows where
  `num_skills == 0`, visits the job's original posting URL and scans the
  full page text for skills (since Adzuna's excerpt is often too short).
  Capped at a sample of 300 rows, parallelized (20 threads) with a hard
  ~165-second time budget (per user's "max 3 minutes" request). **Safe
  to re-run** — already-filled rows are skipped, so running it multiple
  times gradually improves coverage of the remaining zero-skill rows.
  Updates `da_job_postings_clean.csv` in place.
- Leftover/unused from earlier dead-end API attempts (safe to delete):
  `test_api.py` (JSearch diagnostic), `test_active_jobs.py` (Active Jobs
  DB diagnostic), the `naukri_jobs_raw.csv` folder (Kaggle fallback,
  contains `naukri_data_scientist.csv` and `naukri_software_engineer.csv`).

## Progress against the original 7-step build plan
1. Done — Scope & Data Plan
2. Done — Collect the Data (Adzuna, 995 postings)
3. In progress — Clean & Extract Skills: done, but coverage is being
   actively improved via `fetch_full_descriptions.py` (last run's exact
   results weren't confirmed yet as of this summary)
4. Not started — Build the Analytics Layer: EDA (skill frequency by
   city, experience-level breakdown, trends over posting dates, salary
   bands where available) in Python, then a polished Power BI dashboard
5. Not started — Set Up the RAG Layer: chunk descriptions, embed with a
   sentence-transformer model, store in Chroma or FAISS, wire up
   retrieval + an LLM for grounded natural-language answers
6. Not started — Combine into One App: Streamlit (or FastAPI/Gradio)
   with a "Dashboard" view and an "Ask the Market" chat view
7. Not started — Document, Deploy, Add to Resume: README in the
   established style, GitHub push, deploy (Streamlit Cloud or similar),
   analytics-first resume bullet

## Working style notes (for whichever AI picks this up)
The user prefers step-by-step guidance (one step at a time, confirm
before moving on), plain conversational explanations over jargon, and
tends to ask for the fastest/simplest path when things get complicated.
Screenshots of his actual screen are often the most reliable way to
confirm what a tool/UI is showing before giving the next instruction.
