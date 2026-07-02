"""
Retrieval-Augmented Generation engine for document Q&A — built with LangChain.

Pipeline (all LangChain components):
    1. RecursiveCharacterTextSplitter chunks the (masked) document text.
    2. HuggingFaceEmbeddings (local sentence-transformers model, no API
       key needed) embeds each chunk.
    3. A LangChain FAISS vectorstore indexes the chunks and is persisted
       to disk per document_id, so the app survives a restart without
       re-embedding.
    4. On a question: similarity_search retrieves the top-K chunks, which
       are fed into an LCEL chain (ChatPromptTemplate | ChatGroq) to
       produce a grounded answer.
    5. If no GROQ_API_KEY is configured, the app falls back to a simple
       extractive answer built directly from the retrieved chunks — no
       LangChain LLM call is made in that case.

IMPORTANT: the RAG index is built over the MASKED document text, never
the raw text — so even the retrieval/LLM layer never sees raw PII.
"""
from pathlib import Path
from typing import List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from backend.config import EMBEDDING_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVAL, INDEX_DIR, GROQ_API_KEY, GROQ_MODEL
_embeddings = None  # lazy singleton — loading the model is the slow part
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embeddings


def _index_dir(document_id: str) -> Path:
    return INDEX_DIR / document_id


def build_index(document_id: str, masked_text: str) -> int:
    """Chunk + embed + persist a LangChain FAISS index for one document. Returns #chunks."""
    chunks = _splitter.split_text(masked_text) or [masked_text or ""]
    docs = [
        Document(page_content=chunk, metadata={"document_id": document_id, "chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]

    vectorstore = FAISS.from_documents(docs, get_embeddings())

    path = _index_dir(document_id)
    path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(path))

    return len(chunks)


def _load_vectorstore(document_id: str) -> FAISS:
    path = _index_dir(document_id)
    if not path.exists():
        raise FileNotFoundError(f"No index found for document_id={document_id}")
    # allow_dangerous_deserialization=True is safe here: we only ever load
    # indexes this same app wrote to /uploads/_indexes, never external files.
    return FAISS.load_local(str(path), get_embeddings(), allow_dangerous_deserialization=True)


def retrieve(document_id: str, question: str, top_k: int = TOP_K_RETRIEVAL) -> List[str]:
    vectorstore = _load_vectorstore(document_id)
    results = vectorstore.similarity_search(question, k=top_k)
    return [doc.page_content for doc in results]


def answer_question(document_id: str, question: str) -> Tuple[str, List[str]]:
    """Returns (answer_text, source_chunks_used)."""
    sources = retrieve(document_id, question)
    context = "\n\n---\n\n".join(sources)

    if GROQ_API_KEY:
        try:
            answer = _run_qa_chain(context, question)
            return answer, sources
        except Exception:
            # Degrade gracefully to extractive mode rather than erroring out.
            fallback = _extractive_answer(question, sources)
            return f"[LLM unavailable, showing extractive answer] {fallback}", sources

    return _extractive_answer(question, sources), sources


def _run_qa_chain(context: str, question: str) -> str:
    """LCEL chain: ChatPromptTemplate | ChatGroq -> plain string answer."""
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0.2, max_tokens=700)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a compliance and data-security assistant. Answer the user's "
         "question using ONLY the provided document excerpts (which have already "
         "had sensitive values masked). If the answer is not in the excerpts, say "
         "so plainly. Be concise and factual. Never attempt to reconstruct masked "
         "values."),
        ("human", "Document excerpts:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
    ])

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question}).strip()


def _extractive_answer(question: str, sources: List[str]) -> str:
    """No-LLM fallback: just surface the most relevant retrieved passages."""
    if not sources:
        return "No relevant content was found in the document for this question."
    joined = "\n\n".join(f"- {s[:400]}" for s in sources[:3])
    return (
        "Here are the most relevant excerpts from the document "
        f"(configure GROQ_API_KEY for a synthesized answer):\n\n{joined}"
    )
