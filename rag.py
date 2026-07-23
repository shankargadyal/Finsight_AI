"""
rag.py - Lightweight RAG (Retrieval-Augmented Generation) for FinSight AI.

Design choice: no LangChain, no FAISS. For a document set of this scale
(tens-to-low-hundreds of chunks from data/docs/), a numpy array + cosine
similarity is exactly as correct as FAISS and far lighter on RAM -- FAISS
holds its own index structure in memory on top of the vectors themselves,
which isn't worth it here and is exactly the kind of extra weight that
caused OOM issues on Render Free before.

The index itself (embeddings + text) is built OFFLINE by build_rag_index.py
and just loaded here. This module NEVER computes embeddings for documents
at request time -- only for the user's query, which is one small API call.

Flow per request:
    query -> embed query (Gemini) -> cosine similarity vs cached chunk
    vectors -> top-k chunks -> build grounded prompt -> Gemini generate
    -> {answer, sources, confidence, retrieved_chunks}
"""
import os
import json
import logging

import numpy as np
from google import genai
from google.genai import types
from net_timeout import call_with_timeout, NetworkTimeoutError

logger = logging.getLogger("finsight.rag")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL = "gemini-flash-latest"
TOP_K = 4

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "data", "rag_index.json")

SYSTEM_PROMPT = """You are FinSight AI's document research assistant.

Answer the user's question using ONLY the excerpts provided below under
"Retrieved context". If the excerpts don't contain enough information to
answer confidently, say so plainly rather than guessing or using outside
knowledge. Cite which source each fact comes from using the source names
given (e.g. "According to AAPL_10K.txt...").

Keep answers concise and factual. This is educational analysis, not
financial advice."""


class RAGUnavailableError(Exception):
    """Raised when the RAG index hasn't been built yet, or retrieval/generation fails."""
    pass


# ─────────────────────────────────────────────────────────────────────────
# Index loading (cached in-process; small enough to hold entirely in RAM)
# ─────────────────────────────────────────────────────────────────────────
_index_cache = None


def _load_index():
    global _index_cache
    if _index_cache is not None:
        return _index_cache

    if not os.path.exists(INDEX_PATH):
        raise RAGUnavailableError(
            "RAG index not found. Run `python build_rag_index.py` after adding "
            "documents to data/docs/."
        )

    with open(INDEX_PATH) as f:
        data = json.load(f)

    chunks = data.get("chunks", [])
    if not chunks:
        raise RAGUnavailableError("RAG index is empty — no chunks were embedded.")

    vectors = np.array([c["embedding"] for c in chunks], dtype=float)
    # Pre-normalise so similarity at query time is a single dot product.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    unit_vectors = vectors / norms

    _index_cache = {"chunks": chunks, "unit_vectors": unit_vectors}
    logger.info("Loaded RAG index: %d chunks", len(chunks))
    return _index_cache


def reload_index():
    """Call after rebuilding the index (e.g. via an admin action) to drop the cache."""
    global _index_cache
    _index_cache = None


# ─────────────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────────────
def _embed_query(client, query: str) -> np.ndarray:
    resp = call_with_timeout(
        client.models.embed_content, timeout=8, model=EMBED_MODEL, contents=query
    )
    return np.array(resp.embeddings[0].values, dtype=float)


def _retrieve(client, query: str, top_k: int = TOP_K):
    index = _load_index()
    q_vec = _embed_query(client, query)
    q_norm = np.linalg.norm(q_vec)
    if q_norm == 0:
        raise RAGUnavailableError("Query embedding failed")
    q_unit = q_vec / q_norm

    sims = index["unit_vectors"] @ q_unit  # cosine similarity, one dot product per chunk
    top_idx = np.argsort(sims)[::-1][:top_k]

    results = []
    for i in top_idx:
        chunk = index["chunks"][i]
        results.append({
            "source": chunk["source"],
            "text": chunk["text"],
            "similarity": round(float(sims[i]), 4),
        })
    return results


# ─────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────
def answer(query: str, top_k: int = TOP_K) -> dict:
    """
    Returns:
        {
          "answer": str,
          "sources": ["AAPL_10K.txt", ...],
          "confidence": float,        # derived from top retrieval similarity
          "retrieved_chunks": [ {source, text, similarity}, ... ],
        }
    Raises RAGUnavailableError if the index is missing/empty or the API call fails.
    """
    if not GEMINI_API_KEY:
        raise RAGUnavailableError("GEMINI_API_KEY is not configured")
    if not query or not query.strip():
        raise RAGUnavailableError("Empty query")

    client = genai.Client(api_key=GEMINI_API_KEY)

    try:
        retrieved = _retrieve(client, query, top_k=top_k)
    except RAGUnavailableError:
        raise
    except Exception as e:
        logger.error("Retrieval failed: %s", e)
        raise RAGUnavailableError(str(e))

    if not retrieved:
        raise RAGUnavailableError("No relevant documents found")

    context_block = "\n\n".join(
        f"[Source: {r['source']}]\n{r['text']}" for r in retrieved
    )
    user_prompt = f"Retrieved context:\n\n{context_block}\n\nQuestion: {query}"

    try:
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=500,
            temperature=0.2,
        )
        response = call_with_timeout(
            client.models.generate_content,
            timeout=10,
            model=GEN_MODEL,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])],
            config=config,
        )
        reply = response.text or ""
    except Exception as e:
        logger.error("Generation failed: %s", e)
        raise RAGUnavailableError(str(e))

    top_similarity = retrieved[0]["similarity"] if retrieved else 0
    # Simple, honest confidence proxy: how close the best-matching chunk was.
    # Cosine similarity of ~0.75+ on Gemini embeddings tends to indicate a strong topical match.
    confidence = round(min(max(top_similarity, 0), 1) * 100, 1)

    sources = sorted({r["source"] for r in retrieved})

    return {
        "answer": reply,
        "sources": sources,
        "confidence": confidence,
        "retrieved_chunks": retrieved,
    }
