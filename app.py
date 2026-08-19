"""
app.py
Step 7: combines everything into one Streamlit app with three views:
  - Dashboard: key stats + the charts from eda.py
  - Ask the Market: chat interface over the RAG layer (ask_market.py logic)
  - Find Jobs: live any-industry search via the Adzuna API

Run with:
    streamlit run app.py

Assumes you've already run (in order): run_pipeline.py (or the
individual fetch/clean/extract/eda scripts) and build_vector_store.py,
so da_job_postings_clean.csv, the charts/ folder, and chroma_db/ all
exist.
"""

import os
import html
from pathlib import Path
from datetime import datetime, timezone

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

SUGGESTED_QUESTIONS = [
    "What skills are most in demand right now?",
    "Show me entry-level DA roles in Mumbai",
    "Which cities have the most openings?",
]

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

st.set_page_config(page_title="DA Job Market Intelligence", page_icon="💼", layout="wide")

# ---------- Visual theme (colorful, job-portal style) ----------

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700&family=Inter:wght@400;500;600&display=swap');

:root {
    --primary: #4F3FF0;
    --primary-2: #7C3FE4;
    --primary-light: #EEEBFF;
    --success: #16A34A;
    --success-light: #E8F9EE;
    --accent: #FF6B4A;
    --accent-dark: #E85A3A;
    --bg: #F6F7FB;
    --card-bg: #FFFFFF;
    --text: #1A1B2E;
    --text-muted: #6B6D85;
    --border: #E7E8F2;
}

.stApp { background-color: var(--bg); font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Poppins', sans-serif !important; }

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-2) 100%);
    border-radius: 20px;
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 1rem;
    box-shadow: 0 10px 30px rgba(79,63,240,0.25);
}
.hero-eyebrow {
    color: rgba(255,255,255,0.85);
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}
.hero-title {
    font-family: 'Poppins', sans-serif;
    color: #FFFFFF;
    font-size: 1.9rem;
    font-weight: 700;
    margin: 0 0 0.4rem 0;
}
.hero-tagline {
    color: rgba(255,255,255,0.9);
    font-size: 0.92rem;
    max-width: 620px;
    line-height: 1.5;
    margin: 0;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    color: #FFFFFF;
    padding: 0.45rem 0.95rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    white-space: nowrap;
}

/* Live pulse dot */
.pulse-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #22C55E;
}
@media (prefers-reduced-motion: no-preference) {
    .pulse-dot { animation: pulse 1.8s infinite; }
}
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(34,197,94,0.5); }
    70% { box-shadow: 0 0 0 6px rgba(34,197,94,0); }
    100% { box-shadow: 0 0 0 0 rgba(34,197,94,0); }
}

/* Tabs as colorful pills */
.stTabs [data-baseweb="tab-list"] { gap: 0.5rem; border-bottom: none; }
.stTabs [data-baseweb="tab"] {
    height: auto;
    padding: 0.6rem 1.4rem;
    background-color: var(--card-bg);
    border-radius: 999px;
    border: 1px solid var(--border);
    font-weight: 600;
    color: var(--text-muted);
    transition: all 0.15s ease;
}
.stTabs [data-baseweb="tab"]:hover { border-color: var(--primary); color: var(--primary); }
.stTabs [data-baseweb="tab-highlight"] { background-color: transparent; }
.stTabs [data-baseweb="tab-border"] { display: none; }
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--primary), var(--primary-2)) !important;
    color: #FFFFFF !important;
    border-color: transparent !important;
}
.stTabs [data-baseweb="tab"] p { color: inherit !important; font-weight: 600; }

/* Force widget labels + inputs to follow our light theme, regardless of
   the viewer's system/browser dark mode (Streamlit's native widgets
   otherwise inherit the ambient theme and can go invisible-on-white) */
[data-testid="stWidgetLabel"] p { color: var(--text) !important; font-weight: 600; }
.stTextInput input {
    background-color: var(--card-bg) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* Buttons */
.stButton>button {
    border-radius: 999px;
    font-weight: 600;
    padding: 0.5rem 1.25rem;
    transition: all 0.15s ease;
    border: 1px solid var(--border);
}
button[kind="primary"] {
    background: linear-gradient(135deg, var(--primary), var(--primary-2));
    border: none;
    color: #FFFFFF;
    box-shadow: 0 4px 14px rgba(79,63,240,0.3);
}
button[kind="primary"]:hover { transform: translateY(-1px); box-shadow: 0 6px 18px rgba(79,63,240,0.4); }
button[kind="secondary"] { background: var(--primary-light); color: var(--primary); border: 1px solid var(--primary-light); }
button[kind="secondary"]:hover { background: var(--primary); color: #FFFFFF; }

/* Section headers (reused across tabs) */
.header-row { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.4rem; }
.header-row .card-icon { font-size: 1.2rem; }
.section-title { font-family: 'Poppins', sans-serif; font-weight: 700; font-size: 1.15rem; color: var(--text); }
.mini-title { font-family: 'Poppins', sans-serif; font-weight: 600; font-size: 1rem; color: var(--text); }
.section-subtitle { color: var(--text-muted); font-size: 0.9rem; margin: -0.2rem 0 0.9rem 1.7rem; }

/* Stat cards */
.stat-card {
    background: var(--card-bg);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 2px 10px rgba(26,27,46,0.06);
    border-left: 4px solid var(--primary);
}
.stat-card-b { border-left-color: var(--success); }
.stat-card-c { border-left-color: var(--accent); }
.stat-card-d { border-left-color: #2563EB; }
.stat-icon { font-size: 1.3rem; }
.stat-value { font-family: 'Poppins', sans-serif; font-size: 1.7rem; font-weight: 700; color: var(--text); margin-top: 0.15rem; }
.stat-label { font-size: 0.82rem; color: var(--text-muted); margin-top: 0.1rem; }

/* Chart cards (wrap st.container(border=True)) */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 16px !important;
    box-shadow: 0 2px 10px rgba(26,27,46,0.06);
}

/* Job cards (Find Jobs tab) */
.job-card {
    background: var(--card-bg);
    border-radius: 16px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.9rem;
    border-left: 4px solid var(--primary);
    box-shadow: 0 2px 10px rgba(26,27,46,0.06);
    transition: box-shadow 0.15s ease, transform 0.15s ease;
}
.job-card:hover { box-shadow: 0 6px 20px rgba(26,27,46,0.12); transform: translateY(-1px); }
.job-card-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 0.75rem; flex-wrap: wrap; }
.job-title { font-family: 'Poppins', sans-serif; font-weight: 600; font-size: 1.08rem; color: var(--text); margin: 0; }
.job-company { color: var(--text-muted); font-size: 0.88rem; margin-top: 0.15rem; }
.job-meta { display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 0.55rem; font-size: 0.85rem; color: var(--text-muted); }
.salary-pill {
    background: var(--success-light); color: var(--success); font-weight: 600; font-size: 0.8rem;
    padding: 0.3rem 0.7rem; border-radius: 999px; white-space: nowrap;
}
.posted-pill { display: inline-flex; align-items: center; gap: 0.4rem; }
.view-posting-link {
    display: inline-block; margin-top: 0.85rem; padding: 0.4rem 1rem;
    background: var(--accent); color: #FFFFFF !important; font-weight: 600; font-size: 0.85rem;
    border-radius: 999px; text-decoration: none; transition: all 0.15s ease;
}
.view-posting-link:hover { background: var(--accent-dark); transform: translateY(-1px); }

/* Chat */
[data-testid="stChatMessage"] { border-radius: 14px; box-shadow: 0 1px 6px rgba(26,27,46,0.05); }
.suggested-label { font-size: 0.85rem; color: var(--text-muted); font-weight: 600; margin: 0.3rem 0 0.5rem; }

/* Footer */
.app-footer {
    text-align: center; color: var(--text-muted); font-size: 0.85rem;
    padding: 1.6rem 0 0.5rem; display: flex; justify-content: center; gap: 0.5rem; flex-wrap: wrap;
}
.app-footer a { color: var(--primary); font-weight: 600; text-decoration: none; }
.app-footer a:hover { text-decoration: underline; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


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


def section_header(icon, title, subtitle=None):
    st.markdown(
        f'<div class="header-row"><span class="card-icon">{icon}</span>'
        f'<span class="section-title">{title}</span></div>',
        unsafe_allow_html=True,
    )
    if subtitle:
        st.markdown(f'<p class="section-subtitle">{subtitle}</p>', unsafe_allow_html=True)


def format_posted(created_str):
    """Turn Adzuna's ISO 'created' timestamp into a short 'posted X ago' label."""
    if not created_str:
        return None
    try:
        dt = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
        days = (datetime.now(timezone.utc) - dt).days
        if days <= 0:
            return "Posted today"
        if days == 1:
            return "Posted 1 day ago"
        return f"Posted {days} days ago"
    except Exception:
        return None


# ---------- Dashboard view ----------

def render_dashboard():
    df = load_data()

    total = len(df)
    matched = (df["num_skills"] > 0).sum()
    has_salary = (df["salary_min"].notna() | df["salary_max"].notna()).sum()
    companies = df["company"].nunique()

    section_header("📊", "Market Overview")

    c1, c2, c3, c4 = st.columns(4)
    stat_cards = [
        (c1, "📄", f"{total:,}", "Total postings", ""),
        (c2, "🎯", f"{matched:,}", f"Skills identified ({matched/total*100:.0f}%)", "stat-card-b"),
        (c3, "💰", f"{has_salary:,}", f"Salary disclosed ({has_salary/total*100:.0f}%)", "stat-card-c"),
        (c4, "🏢", f"{companies:,}", "Companies hiring", "stat-card-d"),
    ]
    for col, icon, value, label, cls in stat_cards:
        with col:
            st.markdown(
                f'<div class="stat-card {cls}"><div class="stat-icon">{icon}</div>'
                f'<div class="stat-value">{value}</div><div class="stat-label">{label}</div></div>',
                unsafe_allow_html=True,
            )

    st.caption(
        "ℹ️ Data refreshes daily via an automated pipeline (Adzuna API → cleaning → "
        "skill extraction). Salary coverage is inherently low — most Indian job "
        "postings on Adzuna don't disclose salary."
    )

    st.write("")
    section_header("📈", "Trends & Breakdown")

    def chart_card(fname, icon, title):
        path = Path(CHARTS_DIR) / fname
        with st.container(border=True):
            st.markdown(
                f'<div class="header-row"><span class="card-icon">{icon}</span>'
                f'<span class="mini-title">{title}</span></div>',
                unsafe_allow_html=True,
            )
            if path.exists():
                with st.expander("Click to view chart"):
                    st.image(str(path), use_container_width=True)
            else:
                st.info(f"Chart not found — run eda.py to generate {fname}")

    # Full-width for the two charts that need horizontal room (15 skill
    # labels; 5 side-by-side city panels), a 2-up row for the simpler pair —
    # rather than one uniform grid that squeezes wide charts and wastes
    # space on the narrow ones.
    chart_card("top_skills.png", "🔥", "Most In-Demand Skills")
    st.write("")
    chart_card("skills_by_city.png", "🏙️", "Top Skills by City")
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        chart_card("experience_level.png", "📶", "Experience Level Breakdown")
    with c2:
        chart_card("postings_over_time.png", "📅", "Postings Over Time")


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

    section_header(
        "💬", "Ask the Market",
        "Ask in plain English — answers are grounded only in real, current postings, never generic advice.",
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if not st.session_state.chat_history:
        st.markdown('<p class="suggested-label">Try asking:</p>', unsafe_allow_html=True)
        chip_cols = st.columns(len(SUGGESTED_QUESTIONS))
        for col, q in zip(chip_cols, SUGGESTED_QUESTIONS):
            with col:
                if st.button(q, key=f"chip_{q}", use_container_width=True):
                    st.session_state.pending_question = q
                    st.rerun()

    question = st.chat_input("Ask about the Data Analyst job market...")
    if not question and st.session_state.get("pending_question"):
        question = st.session_state.pop("pending_question")
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
            with st.expander("🔗 Sources used"):
                for s in sources:
                    st.markdown(f"- {s}")

    st.session_state.chat_history.append({"role": "assistant", "content": answer})


# ---------- Find Jobs view (live search, ANY industry) ----------

def render_find_jobs():
    if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
        st.error("ADZUNA_APP_ID / ADZUNA_APP_KEY not found. Add them to your .env "
                 "file (or Streamlit Cloud Secrets) and restart.")
        return

    section_header(
        "🔍", "Find Jobs — Any Industry",
        "Live search across all industries — not limited to the Data Analyst dataset used elsewhere in this app.",
    )

    query = st.text_input("💼 What job are you looking for?", placeholder="e.g. graphic designer, electrician, data entry")
    col1, col2 = st.columns([1, 1])
    with col1:
        city_filter = st.text_input("📍 City (optional)", placeholder="e.g. Mumbai")
    with col2:
        num_results = st.slider("🔢 Number of results", 5, 30, 10)

    search_clicked = st.button("🔍 Search jobs", type="primary")

    if not search_clicked or not query.strip():
        return

    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_APP_KEY,
        "results_per_page": 50,  # Adzuna's max per page — fetch a large candidate
                                  # pool so client-side title filtering below still
                                  # has enough to work with (requesting only
                                  # num_results upfront left too few after filtering)
        "what": query,  # broad match — precision is handled client-side below
                        # by preferring postings whose TITLE contains the query
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

    # Adzuna's phrase/word matching searches the FULL description, not just
    # the title — so a query like "data entry" can match a "Senior Site
    # Manager" post that merely mentions "data entry" once among its duties.
    # Re-rank client-side: prefer postings whose TITLE actually contains the
    # search words (what someone searching a role name almost always means),
    # and only fall back to the broader description-matched set if that's empty.
    query_words = [w.lower() for w in query.split() if w]
    title_matches = [
        job for job in results
        if any(w in (job.get("title") or "").lower() for w in query_words)
    ]
    shown_results = (title_matches if title_matches else results)[:num_results]
    if title_matches and len(title_matches) < len(results):
        st.caption(f"Showing {len(title_matches)} postings with \"{query}\" in the job title "
                    f"(filtered from {len(results)} broader matches for relevance).")

    st.success(f"Found {len(shown_results)} matching postings.")
    for job in shown_results:
        title = html.escape(job.get("title", "Untitled"))
        company = html.escape((job.get("company") or {}).get("display_name", ""))
        location = html.escape((job.get("location") or {}).get("display_name", ""))
        url = job.get("redirect_url", "")
        salary_min = job.get("salary_min")
        salary_max = job.get("salary_max")
        posted = format_posted(job.get("created"))

        salary_html = ""
        if salary_min or salary_max:
            if salary_min and salary_max:
                salary_text = f"₹{salary_min:,.0f} – ₹{salary_max:,.0f}"
            else:
                salary_text = f"₹{salary_min or salary_max:,.0f}"
            salary_html = f'<span class="salary-pill">💰 {salary_text}</span>'

        posted_html = (
            f'<span class="posted-pill"><span class="pulse-dot"></span>{posted}</span>'
            if posted else ""
        )
        company_html = f'<div class="job-company">🏢 {company}</div>' if company else ""
        link_html = (
            f'<a class="view-posting-link" href="{url}" target="_blank">View posting →</a>'
            if url else ""
        )

        card_html = (
            f'<div class="job-card">'
            f'<div class="job-card-top">'
            f'<div><p class="job-title">{title}</p>{company_html}</div>'
            f'{salary_html}'
            f'</div>'
            f'<div class="job-meta"><span>📍 {location}</span>{posted_html}</div>'
            f'{link_html}'
            f'</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)


# ---------- Main layout ----------

st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-text">
            <div class="hero-eyebrow">💼 JOB MARKET INTELLIGENCE</div>
            <div class="hero-title">Data Analyst Job Market — India</div>
            <p class="hero-tagline">Real postings tracked daily across India — skills in demand,
            live openings, and a research assistant that only answers from real data.</p>
        </div>
        <div class="hero-badge"><span class="pulse-dot"></span> Live · auto-refreshed daily</div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["🏠 Dashboard", "💬 Ask the Market", "🔍 Find Jobs"])
with tab1:
    render_dashboard()
with tab2:
    render_ask_market()
with tab3:
    render_find_jobs()

st.markdown(
    """
    <div class="app-footer">
        <span>📊 Data refreshes daily via an automated pipeline</span>
        <span>·</span>
        <a href="https://github.com/SaadShaikh14/DA-job-market-tool" target="_blank">View source on GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)
