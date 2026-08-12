# Data Analyst Job Market Intelligence Tool — The Story

## The one-line version
*"I built a tool that tracks live Data Analyst job postings across India, automatically refreshes itself every day, and lets you ask it questions about the job market in plain English — grounded in real, current postings, not guesses."*

## The problem I was solving
As a final-year student targeting Data Analyst roles, I kept asking the same questions manually: *What skills are companies actually asking for right now? Which cities are hiring? What does a "junior" DA posting even look like versus a "senior" one?* Job portals don't answer this in aggregate — you can only look at one posting at a time. I wanted a system that could look at hundreds of postings at once and answer that kind of question directly.

I also wanted this project to clearly demonstrate **analytics skills**, not just show off AI — the RAG/chatbot layer is a feature on top of a real data pipeline, not the whole point.

## What it actually does
The tool has three parts a user can interact with:

1. **A Dashboard** — live stats and charts: the most in-demand skills (SQL, Python, Power BI, Excel top the list), how skill demand differs by city, a breakdown of experience levels being hired for, and posting trends over time.
2. **"Ask the Market"** — a chat interface where you can ask something like *"What do Mumbai companies want for a DA fresher?"* and get an answer built directly from real postings, with the source postings shown so you can verify it.
3. **"Find Jobs"** — a personal search: describe the kind of job you want in your own words, optionally filter by city or experience level, and get back a ranked list of actual current postings that match — not a generic market summary, but jobs *for you* to apply to.

## How it's built (the pipeline)
**Step 1 — Data collection.** A Python script pulls live postings from the Adzuna API (free tier, covers India) for "data analyst", "business analyst", and "junior data analyst" searches. This replaced an earlier plan to scrape job sites directly, since a proper API is more reliable and more defensible in an interview.

**Step 2 — Cleaning.** Duplicate postings are removed, invalid salary values are fixed, and outliers are capped so a few garbage data points don't skew the charts.

**Step 3 — Skill & experience extraction.** Each posting's title and description are scanned for ~35 known Data Analyst skills (SQL, Python, Power BI, Tableau, etc.) using pattern matching, and a rough experience level (Fresher / Junior / Senior / Manager+) is guessed from the title. A gap-filling step also visits postings that didn't yield any skills on the first pass and re-scans the full page, since the API only returns a truncated description.

**Step 4 — Analytics layer.** The cleaned data feeds a set of charts: top skills overall, top skills by city, experience-level breakdown, and posting volume over time.

**Step 5 — RAG layer (the AI feature).** Every posting is turned into a text chunk, embedded with a small local embedding model (no API cost), and stored in a vector database (Chroma). When someone asks a question, the system retrieves the most relevant postings and asks an LLM (Groq, free tier) to answer *using only those postings* — so the answers are grounded in real, current data instead of the model's general knowledge.

**Step 6 — The app.** Everything is combined into a single Streamlit app with the three tabs described above.

**The self-updating piece.** All five collection/cleaning/analysis steps are chained into one script and scheduled to run automatically every day via Windows Task Scheduler. New postings get merged into the existing dataset (not overwritten), so the tool keeps itself current — and also keeps growing richer over time (more historical trend data, more salary data) rather than resetting itself each day.

## A decision worth mentioning if asked
Early on, a script that refetched data accidentally **overwrote** the existing dataset instead of adding to it — losing some accumulated history. That's exactly why the daily-refresh script was rebuilt to *merge and deduplicate* rather than overwrite: a good learning moment about designing data pipelines that are safe to re-run.

## Tech stack
Python, pandas, requests — Adzuna API for data · sentence-transformers + Chroma for embeddings/retrieval · Groq (Llama 3.3) for the LLM · Streamlit for the app · Windows Task Scheduler for automation.

## How to describe it on a resume (analytics-first framing)
> *Built a live job-market analytics tool tracking Data Analyst postings across India via a real-time API pipeline, with automated daily refresh; added a RAG-based Q&A and job-search layer for natural-language queries grounded in current postings.*

## Where it stands right now
Steps 1–6 are done and working end-to-end. What's left: Step 7 — writing up a proper README, pushing to GitHub, deploying it somewhere public (e.g. Streamlit Cloud), and finalizing the resume bullet.
