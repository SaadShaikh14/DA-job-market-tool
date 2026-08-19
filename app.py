"""
app.py
Step 6: combines everything into one Streamlit app with two views:
  - Dashboard: key stats + the charts from eda.py
  - Ask the Market: chat interface over the RAG layer (ask_market.py logic)

Run with:
    streamlit run app.py

Assumes you've already run (in order): run_pipeline.py (or the
individual fetch/clean/extract/eda scripts) and build_vector_store.py,
so da_job_postings_clean.csv, the charts/ folder, and chroma_db/ all
exist.
"""

import os
import ast
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

CSV_PATH = "da_job_postings_clean.csv"
CHARTS_DIR = "charts"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "job_postings"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-120b"  # llama-3.3-70b-versatile was deprecated by Groq in June 2026
TOP_K = 8

# Find Jobs searches ALL industries live via Adzuna, separate from the
# Dashboard/Ask the Market data (which stays Data-Analyst-focused, from
# the pre-built dataset/vector store). A general "any job, any industry"
# search can't be pre-embedded for every possible query, so this tab
# just calls Adzuna's search directly with whatever the user typed.
ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY")
ADZUNA_SEARCH_URL = "https://api.adzuna.com/v1/api/jobs/in/search/1"

SYSTEM_PROMPT = """You are a job-market research assistant. You answer \
questions ONLY using the job postings provided as context below — do \
not use outside knowledge about the job market. If none of the \
postings are relevant to the question, say so plainly instead of \
forcing the structure below onto irrelevant data.

When relevant postings ARE found, ALWAYS structure your answer exactly \
like this, regardless of which job role or skill was asked about:

1. A markdown table of the most relevant postings with these exact \
columns: Company / Role | Location | Seniority (as stated) | Core \
skills mentioned. Use "Not specified" where a posting doesn't state it \
— never invent a value.
2. A short section titled "What the postings tell you about \
<the role/topic asked about>", with bullet points covering: geography \
(which cities/regions show up), skill clusters (which skills appear \
together often), seniority spread, and typical responsibilities if \
the postings mention them.
3. A one-line "Bottom line:" takeaway giving practical advice for \
someone targeting this kind of role.

Keep the table and bullets grounded strictly in the retrieved \
postings — don't pad with generic career advice not supported by the \
data."""

st.set_page_config(page_title="DA Job Market Intelligence", layout="wide")


# ---------- Cached resources (loaded once per server session) ----------

@st.cache_resource
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(COLLECTION_NAME)


@st.cache_resource
def load_groq_client():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None
    return Groq(api_key=key)


@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    return df


# ---------- Dashboard view ----------

def render_dashboard():
    df = load_data()

    total = len(df)
    matched = (df["num_skills"] > 0).sum()
    has_salary = (df["salary_min"].notna() | df["salary_max"].notna()).sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total postings", f"{total:,}")
    c2.metric("Postings with skills identified", f"{matched:,}", f"{matched/total*100:.0f}%")
    c3.metric("Postings with salary data", f"{has_salary:,}", f"{has_salary/total*100:.0f}%")

    st.caption(
        "Data refreshes daily via an automated pipeline (Adzuna API → cleaning → "
        "skill extraction). Salary coverage is inherently low — most Indian job "
        "postings on Adzuna don't disclose salary."
    )

    st.divider()

    chart_files = {
        "top_skills.png": "Most In-Demand Skills",
        "skills_by_city.png": "Top Skills by City",
        "experience_level.png": "Experience Level Breakdown",
        "postings_over_time.png": "Postings Over Time",
    }

    cols = st.columns(2)
    for i, (fname, title) in enumerate(chart_files.items()):
        path = Path(CHARTS_DIR) / fname
        with cols[i % 2]:
            st.subheader(title)
            if path.exists():
                st.image(str(path), use_container_width=True)
            else:
                st.info(f"Chart not found — run eda.py to generate {fname}")


# ---------- Ask the Market view ----------

def format_context(results):
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    blocks = []
    for doc, meta in zip(docs, metas):
        blocks.append(
            f"--- Posting: {meta.get('title')} at {meta.get('company')} "
            f"({meta.get('location')}) ---\n{doc}"
        )
    return "\n\n".join(blocks)


def render_ask_market():
    groq_client = load_groq_client()
    if groq_client is None:
        st.error("GROQ_API_KEY not found. Add it to your .env file and restart.")
        return

    embed_model = load_embed_model()
    try:
        collection = load_collection()
    except Exception:
        st.error("Vector store not found. Run build_vector_store.py first.")
        return

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask about the Data Analyst job market...")
    if not question:
        return

    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching postings..."):
            query_embedding = embed_model.encode([question]).tolist()
            results = collection.query(query_embeddings=query_embedding, n_results=TOP_K)

            if not results["documents"] or not results["documents"][0]:
                answer = "I couldn't find any relevant postings in the data for that question."
                sources = []
            else:
                context = format_context(results)
                response = groq_client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Context (real job postings):\n\n{context}\n\nQuestion: {question}"},
                    ],
                    temperature=0.2,
                )
                answer = response.choices[0].message.content
                sources = [
                    f"{m.get('title')} @ {m.get('company')} ({m.get('location')})"
                    for m in results["metadatas"][0]
                ]

        st.markdown(answer)
        if sources:
            with st.expander("Sources used"):
                for s in sources:
                    st.markdown(f"- {s}")

    st.session_state.chat_history.append({"role": "assistant", "content": answer})


# ---------- Find Jobs view (live search, ANY industry) ----------

def render_find_jobs():
    if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
        st.error("ADZUNA_APP_ID / ADZUNA_APP_KEY not found. Add them to your .env "
                 "file (or Streamlit Cloud Secrets) and restart.")
        return

    st.caption("Search for any job, any industry — this searches live, current postings "
               "directly (not limited to the Data Analyst dataset used elsewhere in this app).")

    query = st.text_input("What job are you looking for?", placeholder="e.g. graphic designer, electrician, data entry")
    col1, col2 = st.columns([1, 1])
    with col1:
        city_filter = st.text_input("City (optional)", placeholder="e.g. Mumbai")
    with col2:
        num_results = st.slider("Number of results", 5, 30, 10)

    search_clicked = st.button("Search jobs", type="primary")

    if not search_clicked or not query.strip():
        return

    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_APP_KEY,
        "results_per_page": num_results,
        "what": query,
        "sort_by": "date",
        "content-type": "application/json",
    }
    if city_filter.strip():
        params["where"] = city_filter.strip()

    with st.spinner("Searching live postings..."):
        try:
            resp = requests.get(ADZUNA_SEARCH_URL, params=params, timeout=15)
            resp.raise_for_status()
            results = resp.json().get("results", [])
        except requests.exceptions.RequestException as e:
            st.error(f"Search failed: {e}")
            return

    if not results:
        st.warning("No matching postings found. Try a different phrasing or drop the city filter.")
        return

    st.success(f"Found {len(results)} matching postings.")
    for job in results:
        with st.container(border=True):
            title = job.get("title", "Untitled")
            company = (job.get("company") or {}).get("display_name", "")
            location = (job.get("location") or {}).get("display_name", "")
            url = job.get("redirect_url", "")
            salary_min = job.get("salary_min")
            salary_max = job.get("salary_max")

            header = f"**{title}** — {company}" if company else f"**{title}**"
            st.markdown(header)
            st.caption(location)
            if salary_min or salary_max:
                st.markdown(f"Salary: {salary_min:,.0f} – {salary_max:,.0f}" if salary_min and salary_max
                            else f"Salary: {salary_min or salary_max:,.0f}")
            if url:
                st.markdown(f"[View posting]({url})")


# ---------- Main layout ----------


st.title("📊 Data Analyst Job Market Intelligence — India")

tab1, tab2, tab3 = st.tabs(["Dashboard", "Ask the Market", "Find Jobs"])
with tab1:
    render_dashboard()
with tab2:
    render_ask_market()
with tab3:
    render_find_jobs()
