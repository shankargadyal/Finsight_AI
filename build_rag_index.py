"""
build_rag_index.py - Offline RAG indexing for FinSight AI.

Run this manually whenever you add/update documents in data/docs/:
    python build_rag_index.py

It reads every .txt/.md file in data/docs/ (SEC filings, annual reports,
earnings call transcripts, etc.), splits them into overlapping chunks,
embeds each chunk with Gemini's embedding model, and saves the result to
data/rag_index.json.

Why no FAISS/LangChain: for a personal-project-sized document set (tens to
low hundreds of chunks), a plain numpy array + cosine similarity is just as
fast and correct as FAISS, with none of FAISS's memory overhead or LangChain's
dependency weight -- important on a 512MB Render instance. Embeddings are
computed via the Gemini API, so no embedding model is ever loaded locally.

NEVER import this file from the Flask app -- it's an offline build step,
just like train_model.py.
"""
import os
import json
import glob
import logging

from google import genai

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("finsight.rag_index")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBED_MODEL = "gemini-embedding-001"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "data", "docs")
INDEX_PATH = os.path.join(BASE_DIR, "data", "rag_index.json")

CHUNK_SIZE = 1000       # characters per chunk
CHUNK_OVERLAP = 150     # characters of overlap between consecutive chunks


def chunk_text(text: str, source: str) -> list[dict]:
    """Fixed-size sliding-window chunking with overlap, tagged with source + position."""
    text = " ".join(text.split())  # normalise whitespace
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            chunks.append({
                "id": f"{source}::chunk{idx}",
                "source": source,
                "text": chunk,
            })
            idx += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def embed_chunks(client, chunks: list[dict]) -> list[dict]:
    """Embed each chunk via the Gemini embeddings API. Kept as separate calls
    (not batched) to stay compatible across google-genai SDK versions."""
    embedded = []
    for i, c in enumerate(chunks):
        try:
            resp = client.models.embed_content(model=EMBED_MODEL, contents=c["text"])
            vector = resp.embeddings[0].values
            embedded.append({**c, "embedding": vector})
            if (i + 1) % 10 == 0:
                logger.info("Embedded %d/%d chunks...", i + 1, len(chunks))
        except Exception as e:
            logger.error("Failed to embed chunk %s: %s", c["id"], e)
    return embedded


def build_index():
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is required to build the RAG index")

    doc_paths = glob.glob(os.path.join(DOCS_DIR, "*.txt")) + glob.glob(os.path.join(DOCS_DIR, "*.md"))
    if not doc_paths:
        raise RuntimeError(
            f"No documents found in {DOCS_DIR}. Add .txt/.md files "
            "(SEC filings, annual reports, earnings transcripts) and re-run."
        )

    logger.info("Found %d source documents", len(doc_paths))

    all_chunks = []
    for path in doc_paths:
        source = os.path.basename(path)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        doc_chunks = chunk_text(text, source)
        logger.info("[%s] -> %d chunks", source, len(doc_chunks))
        all_chunks.extend(doc_chunks)

    client = genai.Client(api_key=GEMINI_API_KEY)
    embedded_chunks = embed_chunks(client, all_chunks)

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    with open(INDEX_PATH, "w") as f:
        json.dump({
            "model": EMBED_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "chunks": embedded_chunks,
        }, f)

    logger.info("Saved %d embedded chunks -> %s", len(embedded_chunks), INDEX_PATH)


if __name__ == "__main__":
    build_index()
