import math
import re
from collections import Counter, defaultdict
import heapq
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

# Common English stopwords to downweight purely grammatical words
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", 
    "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", 
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", 
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", 
    "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", 
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", 
    "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", 
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", 
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", 
    "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"
}

_WORD_PATTERN = re.compile(r'\b[a-zA-Z0-9_\u00C0-\u017F]+\b')

def _tokenize(text: str) -> List[str]:
    """Fast regex-based tokenization with lowercasing."""
    if not text:
        return []
    return _WORD_PATTERN.findall(text.lower())


class DocumentVectorStore:
    """
    Production-grade Hybrid Document Store combining:
    1. Dense Semantic Vector Retrieval (Google Gemini text-embedding-004)
    2. Sparse Lexical BM25 Retrieval via Inverted Index
    3. Reciprocal Rank Fusion (RRF) for optimal ranking accuracy
    
    Operates with zero heavy C++ dependencies. Gracefully falls back to BM25
    if embeddings are not configured or when operating offline.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75, embedding_model: str = "text-embedding-004"):
        self.k1 = k1
        self.b = b
        self.embedding_model = embedding_model
        
        self.chunks: List[Dict[str, Any]] = []
        
        # Sparse BM25 index data
        self.doc_len: List[int] = []
        self.doc_term_freqs: List[Counter] = []
        self.doc_term_sets: List[set] = []
        self.inverted_index: Dict[str, List[int]] = defaultdict(list)
        
        self.total_docs: int = 0
        self.avg_doc_len: float = 0.0
        self.total_words: int = 0
        self._idf_cache: Dict[str, float] = {}

        # Dense vector index data
        self.embeddings: Optional[np.ndarray] = None  # shape: (N, dim), normalized L2

    def add_chunks(self, chunks: List[Dict[str, Any]], client: Optional[Any] = None):
        """
        Add chunks to the store, index lexical tokens into the inverted index,
        and optionally compute dense semantic embeddings if a Gemini client is provided.
        """
        if not chunks:
            return
            
        start_idx = len(self.chunks)
        self.chunks.extend(chunks)
        
        # 1. Update Sparse BM25 Inverted Index
        for i, chunk in enumerate(chunks):
            doc_idx = start_idx + i
            text = chunk.get("page_content", "")
            tokens = _tokenize(text)
            
            length = len(tokens)
            self.doc_len.append(length)
            self.total_words += length
            
            tf = Counter(tokens)
            self.doc_term_freqs.append(tf)
            self.doc_term_sets.append(set(tf.keys()))
            
            for term in tf:
                self.inverted_index[term].append(doc_idx)
                
        self.total_docs = len(self.chunks)
        self.avg_doc_len = (self.total_words / self.total_docs) if self.total_docs > 0 else 0.0
        self._idf_cache.clear()

        # 2. Compute Dense Semantic Embeddings if client is provided
        if client:
            self._compute_dense_embeddings(chunks, client)

    def _compute_dense_embeddings(self, new_chunks: List[Dict[str, Any]], client: Any):
        """
        Batch compute normalized dense embeddings using Gemini text-embedding-004.
        """
        try:
            texts_to_embed = [c.get("page_content", "")[:2000] for c in new_chunks]
            batch_size = 50
            new_vectors = []

            for i in range(0, len(texts_to_embed), batch_size):
                batch = texts_to_embed[i:i + batch_size]
                response = client.models.embed_content(
                    model=self.embedding_model,
                    contents=batch
                )
                if hasattr(response, "embeddings") and response.embeddings:
                    for emb in response.embeddings:
                        vals = np.array(emb.values, dtype=np.float32)
                        norm = np.linalg.norm(vals)
                        if norm > 0:
                            vals /= norm
                        new_vectors.append(vals)

            if new_vectors:
                new_arr = np.vstack(new_vectors)
                if self.embeddings is None:
                    self.embeddings = new_arr
                else:
                    self.embeddings = np.vstack([self.embeddings, new_arr])
        except Exception:
            # If embedding fails (e.g. rate limit, offline), BM25 fallback is preserved
            pass

    def _get_idf(self, term: str) -> float:
        """Calculate BM25 inverse document frequency with caching."""
        if term in self._idf_cache:
            return self._idf_cache[term]
            
        doc_freq = len(self.inverted_index.get(term, []))
        if doc_freq == 0:
            idf = 0.0
        else:
            idf = math.log(1.0 + (self.total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            if term in STOPWORDS:
                idf = max(0.1, idf * 0.3)
            else:
                idf = max(0.2, idf)
                
        self._idf_cache[term] = idf
        return idf

    def _bm25_search(self, query: str, top_n: int = 20) -> List[Tuple[int, float]]:
        """Compute BM25 scores for candidate documents."""
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
            
        query_terms = [t for t in query_tokens if t not in STOPWORDS]
        if not query_terms:
            query_terms = query_tokens

        candidate_docs = set()
        for term in query_terms:
            if term in self.inverted_index:
                candidate_docs.update(self.inverted_index[term])
                
        if not candidate_docs:
            for term in query_tokens:
                if term in self.inverted_index:
                    candidate_docs.update(self.inverted_index[term])
                    
        if not candidate_docs:
            return [(i, 0.05) for i in range(min(top_n, self.total_docs))]

        k1, b = self.k1, self.b
        avg_len = self.avg_doc_len if self.avg_doc_len > 0 else 1.0

        scores: List[Tuple[int, float]] = []
        for doc_id in candidate_docs:
            tf_map = self.doc_term_freqs[doc_id]
            dl = self.doc_len[doc_id]
            doc_score = 0.0
            
            len_norm = 1.0 - b + b * (dl / avg_len)
            
            for term in query_terms:
                freq = tf_map.get(term, 0)
                if freq > 0:
                    idf = self._get_idf(term)
                    term_score = idf * ((freq * (k1 + 1.0)) / (freq + k1 * len_norm))
                    doc_score += term_score

            doc_score += (1.0 / (1.0 + doc_id)) * 0.001
            scores.append((doc_id, doc_score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]

    def _dense_search(self, query: str, client: Any, top_n: int = 20) -> List[Tuple[int, float]]:
        """Compute Cosine Similarity against dense vector embeddings."""
        if self.embeddings is None or len(self.embeddings) == 0:
            return []
            
        try:
            resp = client.models.embed_content(
                model=self.embedding_model,
                contents=query[:2000]
            )
            if not hasattr(resp, "embeddings") or not resp.embeddings:
                return []
                
            q_vec = np.array(resp.embeddings[0].values, dtype=np.float32)
            norm = np.linalg.norm(q_vec)
            if norm > 0:
                q_vec /= norm

            # Cosine similarity via dot product (both vectors are normalized)
            # Ensure dimension alignment
            if self.embeddings.shape[1] == len(q_vec):
                sims = np.dot(self.embeddings[:self.total_docs], q_vec)
                top_indices = np.argsort(sims)[::-1][:top_n]
                return [(int(idx), float(sims[idx])) for idx in top_indices]
        except Exception:
            pass
            
        return []

    def similarity_search(
        self, 
        query: str, 
        k: int = 4, 
        client: Optional[Any] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find top-k most relevant chunks using Reciprocal Rank Fusion (RRF)
        combining Dense Semantic Vector search and Sparse BM25 search.
        """
        if not self.chunks or not query or not query.strip():
            return []

        # 1. Retrieve dense vector candidates if client and embeddings exist
        dense_results = []
        if client and self.embeddings is not None and len(self.embeddings) >= len(self.chunks):
            dense_results = self._dense_search(query, client, top_n=max(k * 4, 20))

        # 2. Retrieve sparse BM25 candidates
        bm25_results = self._bm25_search(query, top_n=max(k * 4, 20))

        # 3. If only BM25 is available, normalize and return
        if not dense_results:
            max_score = bm25_results[0][1] if bm25_results and bm25_results[0][1] > 0 else 1.0
            return [
                (self.chunks[doc_id], min(1.0, max(0.1, score / max_score)))
                for doc_id, score in bm25_results[:k]
            ]

        # 4. Hybrid Reciprocal Rank Fusion (RRF)
        # RRF Score = w_dense / (60 + rank_dense) + w_sparse / (60 + rank_sparse)
        rrf_scores: Dict[int, float] = defaultdict(float)
        w_dense = 0.65
        w_sparse = 0.35

        for rank, (doc_id, _) in enumerate(dense_results):
            rrf_scores[doc_id] += w_dense / (60.0 + rank + 1)

        for rank, (doc_id, _) in enumerate(bm25_results):
            rrf_scores[doc_id] += w_sparse / (60.0 + rank + 1)

        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        max_rrf = sorted_rrf[0][1] if sorted_rrf and sorted_rrf[0][1] > 0 else 1.0
        
        results = []
        for doc_id, score in sorted_rrf:
            normalized_score = min(1.0, max(0.1, score / max_rrf))
            results.append((self.chunks[doc_id], round(normalized_score, 3)))

        return results


def format_context(results: list) -> tuple:
    """
    Format search results into structured, numbered sources for the AI prompt
    and extract citation information for UI citation cards.
    """
    context_parts = []
    citations = []
    
    for i, (chunk, score) in enumerate(results):
        meta = chunk.get("metadata", {}) or {}
        filename = meta.get("filename", "Unknown")
        page = meta.get("page", "?")
        content = chunk.get("page_content", "")
        
        # 1. Clear numbered citation format for LLM grounding: [Source 1]
        context_parts.append(
            f"[Source {i+1}] (File: {filename}, Page: {page})\n{content.strip()}"
        )
        
        # 2. Structured data for the UI
        clean_content = content.strip()
        snippet = (clean_content[:200] + "...") if len(clean_content) > 200 else clean_content
        citations.append({
            "index": i + 1,
            "filename": filename,
            "page": page,
            "score": round(score, 3),
            "snippet": snippet,
            "label": f"[{i+1}] {filename} • Page {page}"
        })
        
    context = "\n\n".join(context_parts)
    return context, citations