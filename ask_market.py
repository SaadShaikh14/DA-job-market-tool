"""
ask_market.py
Step 5, part 2: the "Ask the Market" Q&A layer. Takes a natural-language
question, retrieves the most relevant job postings from the Chroma
vector store, and asks Groq's LLM to answer using ONLY those postings
as grounding (so answers are based on real, current data, not the
model's general knowledge).

Usage:
    python ask_market.py "What skills do Mumbai companies want for DA freshers?"

Or run with no argument for an interactive prompt loop.
"""

import os
import sys
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "job_postings"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-120b"  # llama-3.3-70b-versatile was deprecated by Groq in June 2026
TOP_K = 8  # how many postings to retrieve as context

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


def answer_question(question, collection, embed_model, groq_client):
    query_embedding = embed_model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=TOP_K)

    if not results["documents"] or not results["documents"][0]:
        print("No relevant postings found in the data for that question.")
        return

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
    print("\n" + answer + "\n")

    print("Sources used:")
    for meta in results["metadatas"][0]:
        print(f"  - {meta.get('title')} @ {meta.get('company')} ({meta.get('location')})")


def main():
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("ERROR: GROQ_API_KEY not found. Check your .env file.")
        return

    print("Loading embedding model and vector store...")
    embed_model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        print(f"ERROR: Collection '{COLLECTION_NAME}' not found. "
              f"Run build_vector_store.py first.")
        return

    groq_client = Groq(api_key=groq_key)

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        answer_question(question, collection, embed_model, groq_client)
        return

    print("Ask a question about the Data Analyst job market (type 'exit' to quit):")
    while True:
        question = input("\n> ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        answer_question(question, collection, embed_model, groq_client)


if __name__ == "__main__":
    main()
