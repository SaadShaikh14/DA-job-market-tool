# 📊 Data Analyst Job Market Intelligence Tool

A live job-market analytics tool that tracks Data Analyst postings across India, refreshes itself daily, and answers natural-language questions about the market — grounded in real, current postings.

**🔗 Live app:** https://da-job-market-tool-pquy4zwmepvwrotcd9g3fx.streamlit.app

## What it does

- **Dashboard** — key stats and charts: most in-demand skills, skill demand by city, experience-level breakdown, and posting trends over time
- **Ask the Market** — a chat interface for questions like *"What skills do Mumbai companies want for DA freshers?"*, answered using only real postings as grounding (RAG), with sources shown
- **Find Jobs** — a personal search: describe the job you want, optionally filter by city/experience level, and get back a ranked list of real matching postings

## How it works

1. **Data collection** — pulls live postings from the [Adzuna API](https://developer.adzuna.com/) for India
2. **Cleaning** — dedupes postings, fixes invalid salary values, caps outliers
3. **Skill & experience extraction** — scans postings for ~35 known Data Analyst skills and guesses an experience level from the title
4. **Analytics layer** — generates charts (top skills, skills by city, experience breakdown, posting trends)
5. **RAG layer** — embeds postings locally (sentence-transformers), stores them in a Chroma vector database, and retrieves relevant postings to ground LLM answers (Groq / Llama 3.3)
6. **App** — everything combined into one Streamlit app

The full pipeline (steps 1–4) is scheduled to run automatically once a day, and new postings are merged into the existing dataset rather than overwriting it — so the tool stays current and keeps getting richer over time.

## Tech stack

Python · pandas · requests · BeautifulSoup · sentence-transformers · Chroma · Groq (Llama 3.3) · Streamlit · Windows Task Scheduler

## Running locally

```bash
pip install -r requirements.txt

# Collect and process data (run once, or use run_pipeline.py to chain all steps)
python fetch_jobs.py
python clean_data.py
python clean_and_extract.py
python fetch_full_descriptions.py
python eda.py

# Build the RAG vector store
python build_vector_store.py

# Launch the app
streamlit run app.py
```

You'll need free API keys from [Adzuna](https://developer.adzuna.com/) and [Groq](https://console.groq.com/), set in a `.env` file:

```
ADZUNA_APP_ID=your_id_here
ADZUNA_APP_KEY=your_key_here
GROQ_API_KEY=your_key_here
```
