"""
build_vector_store.py
Step 5, part 1: turns the cleaned job postings into a searchable vector
store so the RAG layer can retrieve relevant postings for a question.

What it does:
- Loads da_job_postings_clean.csv
- Builds one text "chunk" per posting (title + company + location +
  experience level + matched skills + description). Adzuna descriptions
  are short excerpts, not full JDs, so one chunk per posting is enough —
  no need for the more complex multi-chunk-per-document splitting you'd
  use for long documents.
- Embeds each chunk locally with a small, free sentence-transformer
  model (no API cost, runs on CPU)
- Stores embeddings + metadata in a local Chroma database (a folder
  called chroma_db/) so retrieval is fast and persists between runs

Safe to re-run any time (e.g. after run_pipeline.py adds new postings):
it rebuilds the collection from the current CSV each time, so it always
reflects the latest data. For ~1,000 postings this takes under a minute
on a normal laptop.

Run this AFTER clean_and_extract.py / fetch_full_descriptions.py have
produced da_job_postings_clean.csv.
"""

import ast
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

CSV_PATH = "da_job_postings_clean.csv"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "job_postings"
EMBED_MODEL = "all-MiniLM-L6-v2"  # small, fast, free, runs locally


def build_chunk_text(row):
    """Combine the useful fields of one posting into a single text chunk."""
    parts = [
        f"Job title: {row['title']}",
        f"Company: {row['company']}",
        f"Location: {row['location_clean']}",
        f"Experience level: {row['experience_level_guess']}",
    ]
    if pd.notna(row.get('skills_matched_str')) and row['skills_matched_str']:
        parts.append(f"Skills mentioned: {row['skills_matched_str']}")
    if pd.notna(row.get('salary_min')) or pd.notna(row.get('salary_max')):
        parts.append(f"Salary range: {row.get('salary_min')} - {row.get('salary_max')}")
    if pd.notna(row.get('description')):
        parts.append(f"Description: {row['description']}")
    return "\n".join(parts)


def main():
    print(f"Loading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)
    df = df.reset_index(drop=True)
    print(f"Loaded {len(df)} postings.")

    print(f"Loading embedding model ({EMBED_MODEL}) — first run downloads it, ~80MB...")
    model = SentenceTransformer(EMBED_MODEL)

    print("Building text chunks...")
    documents = [build_chunk_text(row) for _, row in df.iterrows()]
    ids = [str(i) for i in df.index]

    metadatas = []
    for _, row in df.iterrows():
        metadatas.append({
            "title": str(row.get("title", "")),
            "company": str(row.get("company", "")),
            "location": str(row.get("location_clean", "")),
            "experience_level": str(row.get("experience_level_guess", "")),
            "skills": str(row.get("skills_matched_str", "")),
            "url": str(row.get("url", "")),
            "posted_date": str(row.get("posted_date", "")),
        })

    print("Computing embeddings (this is the slow part, runs once)...")
    embeddings = model.encode(documents, show_progress_bar=True, batch_size=32).tolist()

    print(f"Writing to Chroma at ./{CHROMA_DIR} ...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    # Rebuild fresh each run so the store always matches the latest CSV
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    # Chroma wants inserts in batches for large collections
    BATCH = 500
    for start in range(0, len(ids), BATCH):
        end = start + BATCH
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )

    print(f"\nDone. Vector store has {collection.count()} postings, ready for queries.")


if __name__ == "__main__":
    main()
