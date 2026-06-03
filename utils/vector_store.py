"""
Embedding generation and FAISS vector store management.
"""

from typing import List, Dict, Optional, Tuple

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Singleton embeddings instance
_embeddings_instance = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load embeddings model once and reuse it.
    """

    global _embeddings_instance

    if _embeddings_instance is None:

        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={
                "device": "cpu"
            },
            encode_kwargs={
                "normalize_embeddings": True
            },
        )

    return _embeddings_instance


def build_vector_store(chunks: List[Document]) -> FAISS:
    """
    Create FAISS vector database from document chunks.
    """

    embeddings = get_embeddings()

    vector_db = FAISS.from_documents(
        chunks,
        embeddings,
    )

    return vector_db


def merge_vector_stores(
    existing: Optional[FAISS],
    new_store: FAISS,
) -> FAISS:
    """
    Merge multiple uploaded document indexes.
    """

    if existing is None:
        return new_store

    existing.merge_from(new_store)

    return existing


def similarity_search(
    vector_db: FAISS,
    query: str,
    k: int = 5,
    score_threshold: float = 0.3,
) -> List[Tuple[Document, float]]:
    """
    Perform semantic similarity search.
    """

    results = vector_db.similarity_search_with_relevance_scores(
        query,
        k=k,
    )

    # Filter low relevance
    filtered = [
        (doc, score)
        for doc, score in results
        if score >= score_threshold
    ]

    # Fallback if nothing passes threshold
    return filtered if filtered else results[:3]


def multi_query_search(
    vector_db: FAISS,
    queries: List[str],
    k: int = 5,
    score_threshold: float = 0.3,
) -> List[Tuple[Document, float]]:
    """
    Perform semantic search for multiple queries and deduplicate results.
    """
    all_results = []
    seen_content = set()
    
    for q in queries:
        results = similarity_search(vector_db, q, k=k, score_threshold=score_threshold)
        for doc, score in results:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                all_results.append((doc, score))
                
    # Sort by score descending (higher is usually better or worse depending on FAISS metric? FAISS default is L2, lower is better. 
    # But `similarity_search_with_relevance_scores` returns higher for better match).
    all_results.sort(key=lambda x: x[1], reverse=True)
    return all_results[:k * 2]


def format_context(
    results: List[Tuple[Document, float]]
) -> Tuple[str, List[Dict]]:
    """
    Build LLM context string and citation metadata.
    """

    context_parts = []

    citations = []

    for i, (doc, score) in enumerate(results):

        meta = doc.metadata

        filename = meta.get("filename", "Unknown")

        page = meta.get("page", "?")

        file_type = meta.get("file_type", "").upper()

        source_label = f"[{i+1}] {filename} • Page {page}"

        # Context for LLM
        context_parts.append(
            f"""
--- Source {i+1}: {filename} (Page {page}) ---

{doc.page_content}
"""
        )

        # Citation info for UI
        citations.append({
            "index": i + 1,
            "filename": filename,
            "page": page,
            "file_type": file_type,
            "score": round(score, 3),
            "snippet": doc.page_content[:200].strip() + "...",
            "label": source_label,
        })

    context = "\n\n".join(context_parts)

    return context, citations