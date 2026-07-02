"""
Retrieval-Augmented Generation engine for document Q&A.

Pipeline:
    1. Chunk the (masked) document text.
    2. Embed chunks with a local sentence-transformers model.
    3. Store vectors in a per-document FAISS index (persisted to disk so
       the app survives a restart without re-embedding).
    4. On a question: embed the question, retrieve top-K similar chunks,
       and either
         a) send {question, chunks} to the Groq LLM for a grounded answer, or
         b) if no GROQ_API_KEY is configured, fall back to a simple
            extractive answer built from the retrieved chunks.

IMPORTANT: the RAG index is built over the MASKED document text, never
the raw text — so even the LLM / retrieval layer never sees raw PII.
"""

import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from backend.config import EMBEDDING_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVAL, INDEX_DIR, GROQ_API_KEY, GROQ_MODEL

_embedder = None  # lazy singleton — loading the model is the slow part


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
    return chunks


def _index_paths(document_id: str) -> Tuple[Path, Path]:
    return (
        INDEX_DIR / f"{document_id}.faiss",
        INDEX_DIR / f"{document_id}.chunks.pkl",
    )


def build_index(document_id: str, masked_text: str) -> int:
    """Chunk + embed + persist a FAISS index for one document. Returns #chunks."""
    chunks = chunk_text(masked_text)
    if not chunks:
        chunks = [masked_text or ""]

    embedder = get_embedder()
    vectors = embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine similarity
    index.add(vectors.astype(np.float32))

    faiss_path, chunks_path = _index_paths(document_id)
    faiss.write_index(index, str(faiss_path))
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    return len(chunks)


def _load_index(document_id: str):
    faiss_path, chunks_path = _index_paths(document_id)
    if not faiss_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(f"No index found for document_id={document_id}")
    index = faiss.read_index(str(faiss_path))
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def retrieve(document_id: str, question: str, top_k: int = TOP_K_RETRIEVAL) -> List[str]:
    index, chunks = _load_index(document_id)
    embedder = get_embedder()
    q_vec = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    k = min(top_k, len(chunks))
    _, indices = index.search(q_vec, k)
    return [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]


def _call_groq(system_prompt: str, user_prompt: str) -> str:
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=700,
    )
    return completion.choices[0].message.content.strip()


def answer_question(document_id: str, question: str) -> Tuple[str, List[str]]:
    """Returns (answer_text, source_chunks_used)."""
    sources = retrieve(document_id, question)
    context = "\n\n---\n\n".join(sources)

    if GROQ_API_KEY:
        system_prompt = (
            "You are a compliance and data-security assistant. Answer the user's "
            "question using ONLY the provided document excerpts (which have already "
            "had sensitive values masked). If the answer is not in the excerpts, say "
            "so plainly. Be concise and factual. Never attempt to reconstruct masked "
            "values."
        )
        user_prompt = f"Document excerpts:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        try:
            answer = _call_groq(system_prompt, user_prompt)
            return answer, sources
        except Exception as e:  # noqa: BLE001 - degrade gracefully to extractive mode
            fallback = _extractive_answer(question, sources)
            return f"[LLM unavailable, showing extractive answer] {fallback}", sources

    return _extractive_answer(question, sources), sources


def _extractive_answer(question: str, sources: List[str]) -> str:
    """No-LLM fallback: just surface the most relevant retrieved passages."""
    if not sources:
        return "No relevant content was found in the document for this question."
    joined = "\n\n".join(f"- {s[:400]}" for s in sources[:3])
    return (
        "Here are the most relevant excerpts from the document "
        f"(configure GROQ_API_KEY for a synthesized answer):\n\n{joined}"
    )