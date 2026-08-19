# 📊 Data Analyst Job Market Intelligence Tool

A live job-market analytics tool that tracks Data Analyst postings across India, refreshes itself automatically every day — even when no one's machine is on — and answers natural-language questions about the market, grounded in real, current postings. Built with a colorful, job-portal-style interface: browse live postings, ask questions in plain English, and track market trends, all in one place.

**🔗 Live app:** https://da-job-market-tool-pquy4zwmepvwrotcd9g3fx.streamlit.app

**📦 Repo:** https://github.com/SaadShaikh14/DA-job-market-tool

## What it does

- **Dashboard** — key stats and charts: most in-demand skills, skill demand by city, experience-level breakdown, and posting trends over time
- **Ask the Market** — a chat interface for questions like *"What skills do Mumbai companies want for DA freshers?"*. Every answer follows a consistent structure: a table of the matching postings, a bullet-point breakdown of what they reveal (geography, skill clusters, seniority spread), and a one-line bottom-line takeaway — grounded only in real postings (RAG), with sources shown
- **Find Jobs — Any Industry** — a live search that isn't limited to Data Analyst roles: describe any job in your own words, optionally filter by city, and get back a ranked list of real, current postings with direct links to apply

## How it works

1. **Data collection** — pulls live postings from the [Adzuna API](https://developer.adzuna.com/) for India, accumulating into a growing dataset (deduped by posting URL) rather than resetting each run
2. **Cleaning** — dedupes postings, fixes invalid salary values, caps outliers
3. **Skill & experience extraction** — scans postings for ~35 known Data Analyst skills and guesses an experience level from the title, with a gap-filling pass that revisits the original posting page for postings where no skills were found
4. **Analytics layer** — generates charts (top skills, skills by city, experience breakdown, posting trends)
5. **RAG layer** — embeds postings locally (sentence-transformers), stores them in a Chroma vector database, and retrieves relevant postings to ground LLM answers (Groq / `openai/gpt-oss-120b`)
6. **App** — everything combined into one Streamlit app with three tabs, styled with a custom colorful theme

### Fully automated, cloud-side

The entire pipeline (collect → clean → extract → analyze → rebuild the vector store) runs **daily via GitHub Actions**, not a local scheduled task — so it keeps running even if no personal machine is powered on. Each run commits the refreshed data straight back to this repo, which the deployed Streamlit app then reflects. See [`.github/workflows/daily_pipeline.yml`](https://github.com/SaadShaikh14/DA-job-market-tool/blob/main/.github/workflows/daily_pipeline.yml).

## Tech stack

Python · pandas · requests · BeautifulSoup · sentence-transformers · Chroma · Groq (`openai/gpt-oss-120b`) · Streamlit (custom CSS theme) · GitHub Actions

## Running locally

```
pip install -r requirements.txt

# Collect and process data — or just run everything at once:
python run_pipeline.py

# ...which internally chains:
# fetch_jobs.py -> clean_data.py -> clean_and_extract.py ->
# fetch_full_descriptions.py -> eda.py -> build_vector_store.py

# Launch the app
streamlit run app.py
```

You'll need free API keys from [Adzuna](https://developer.adzuna.com/) and [Groq](https://console.groq.com/), set in a `.env` file:

```
ADZUNA_APP_ID=your_id_here
ADZUNA_APP_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

To run the daily automation yourself on a fork: add `ADZUNA_APP_ID` and `ADZUNA_APP_KEY` as repository secrets (Settings → Secrets and variables → Actions), and enable "Read and write permissions" for workflows (Settings → Actions → General).

## Project background

Built as a portfolio project for a Data Analyst job search — see [`PROJECT_NARRATIVE.md`](https://github.com/SaadShaikh14/DA-job-market-tool/blob/main/PROJECT_NARRATIVE.md) for the full story of what it does, why it's built the way it is, and a suggested resume framing.
