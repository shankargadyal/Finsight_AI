"""
rag/retriever.py — Lightweight RAG retrieval for FinSight AI.

Design constraints (Render Free tier: ~512MB RAM, fractional shared CPU):
  - No FAISS, no neural embedding model. A real sentence-embedding model is
    typically 80-400MB+ once loaded, which is a meaningful chunk of this
    instance's entire memory budget on its own — not worth it for a handful
    of short documents.
  - Uses scikit-learn's TfidfVectorizer + cosine similarity instead. This
    adds zero new dependencies (scikit-learn is already required for the
    forecasting models) and the whole index for this document set is a few
    KB in memory.
  - The index is built once at process start (documents are static, bundled
    files) and cached in module-level state — not rebuilt per-request.

This retrieves relevant chunks from rag/knowledge_base/*.txt; the caller
(assistant.py) is responsible for sending those chunks to the LLM as
grounding context.
"""
import os
import re
import glob

_KB_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")

_vectorizer = None
_doc_matrix = None
_chunks = []       # list of {"text": str, "source": str}


def _chunk_text(text: str, source: str):
    """Split a document into paragraph-level chunks (blank-line separated)."""
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return [{"text": p, "source": source} for p in parts]


def _build_index():
    global _vectorizer, _doc_matrix, _chunks
    from sklearn.feature_extraction.text import TfidfVectorizer

    _chunks = []
    for path in sorted(glob.glob(os.path.join(_KB_DIR, "*.txt"))):
        source = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r", encoding="utf-8") as f:
            _chunks.extend(_chunk_text(f.read(), source))

    if not _chunks:
        _vectorizer, _doc_matrix = None, None
        return

    _vectorizer = TfidfVectorizer(stop_words="english", max_df=0.9)
    _doc_matrix = _vectorizer.fit_transform([c["text"] for c in _chunks])


def retrieve(query: str, top_k: int = 3, min_score: float = 0.05):
    """
    Return up to top_k chunks most relevant to `query`, each as
    {"text", "source", "score"}, sorted by relevance. Returns an empty list
    if nothing clears min_score (better to say "no grounded context found"
    than force in an irrelevant chunk).
    """
    global _vectorizer, _doc_matrix
    if _vectorizer is None:
        _build_index()
    if _vectorizer is None or _doc_matrix is None:
        return []

    from sklearn.metrics.pairwise import cosine_similarity

    q_vec = _vectorizer.transform([query])
    scores = cosine_similarity(q_vec, _doc_matrix)[0]

    ranked = sorted(zip(_chunks, scores), key=lambda t: t[1], reverse=True)
    results = [
        {"text": c["text"], "source": c["source"], "score": round(float(s), 4)}
        for c, s in ranked[:top_k]
        if s >= min_score
    ]
    return results
