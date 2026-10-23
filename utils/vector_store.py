import math
import re
from collections import Counter, defaultdict
import heapq
from typing import List, Tuple, Dict, Any, Optional

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
    High-performance in-memory Document Store using an inverted index
    and BM25 ranking algorithm.
    
    Provides sub-millisecond retrieval across thousands of chunks without
    heavy C++ dependencies or slow linear-scanning tokenization.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[Dict[str, Any]] = []
        
        # Pre-computed chunk index data
        self.doc_len: List[int] = []
        self.doc_term_freqs: List[Counter] = []
        self.doc_term_sets: List[set] = []
        self.inverted_index: Dict[str, List[int]] = defaultdict(list)
        
        self.total_docs: int = 0
        self.avg_doc_len: float = 0.0
        self.total_words: int = 0
        self._idf_cache: Dict[str, float] = {}

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Add chunks to the store and incrementally index their tokens.
        """
        if not chunks:
            return
            
        start_idx = len(self.chunks)
        self.chunks.extend(chunks)
        
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
            
            # Populate inverted index
            for term in tf:
                self.inverted_index[term].append(doc_idx)
                
        self.total_docs = len(self.chunks)
        self.avg_doc_len = (self.total_words / self.total_docs) if self.total_docs > 0 else 0.0
        # Invalidate IDF cache when new docs are added
        self._idf_cache.clear()

    def _get_idf(self, term: str) -> float:
        """Calculate Lucene/BM25 inverse document frequency with caching."""
        if term in self._idf_cache:
            return self._idf_cache[term]
            
        doc_freq = len(self.inverted_index.get(term, []))
        if doc_freq == 0:
            idf = 0.0
        else:
            # Standard BM25 IDF formulation with floor
            idf = math.log(1.0 + (self.total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            if term in STOPWORDS:
                idf = max(0.1, idf * 0.3)  # Soften impact of stop words
            else:
                idf = max(0.2, idf)
                
        self._idf_cache[term] = idf
        return idf

    def similarity_search(self, query: str, k: int = 4) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find top-k most relevant chunks using BM25 ranking via inverted index lookup.
        Runs in O(matching docs) instead of O(total docs).
        """
        if not self.chunks or not query or not query.strip():
            return []
            
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
            
        # Filter into content terms vs stopwords
        query_terms = [t for t in query_tokens if t not in STOPWORDS]
        if not query_terms:
            query_terms = query_tokens

        # Collect candidate document IDs from inverted index
        candidate_docs = set()
        for term in query_terms:
            if term in self.inverted_index:
                candidate_docs.update(self.inverted_index[term])
                
        if not candidate_docs:
            # Fallback to checking full query tokens (including stopwords)
            for term in query_tokens:
                if term in self.inverted_index:
                    candidate_docs.update(self.inverted_index[term])
                    
        if not candidate_docs:
            # Return first k chunks if completely unmatched
            return [(self.chunks[i], 0.05) for i in range(min(k, len(self.chunks)))]

        # Score candidates using BM25
        k1 = self.k1
        b = self.b
        avg_len = self.avg_doc_len if self.avg_doc_len > 0 else 1.0

        scores: List[Tuple[float, int]] = []
        for doc_id in candidate_docs:
            tf_map = self.doc_term_freqs[doc_id]
            dl = self.doc_len[doc_id]
            doc_score = 0.0
            
            len_norm = 1.0 - b + b * (dl / avg_len)
            
            for term in query_terms:
                freq = tf_map.get(term, 0)
                if freq > 0:
                    idf = self._get_idf(term)
                    # BM25 term score
                    term_score = idf * ((freq * (k1 + 1.0)) / (freq + k1 * len_norm))
                    doc_score += term_score

            # Small tie-breaker for chronological / stable order
            doc_score += (1.0 / (1.0 + doc_id)) * 0.001
            scores.append((doc_score, doc_id))

        # Use heapq.nlargest for O(N log k) top-k selection
        top_k_scores = heapq.nlargest(k, scores, key=lambda x: x[0])
        
        max_score = top_k_scores[0][0] if top_k_scores and top_k_scores[0][0] > 0 else 1.0
        
        normalized_results = []
        for score, doc_id in top_k_scores:
            norm_score = min(1.0, max(0.1, score / max_score)) if max_score > 0 else 0.1
            normalized_results.append((self.chunks[doc_id], norm_score))
            
        return normalized_results


def format_context(results: list) -> tuple:
    """
    Format the search results into a readable string for the AI 
    and extract citation information for the UI.
    """
    context_parts = []
    citations = []
    
    for i, (chunk, score) in enumerate(results):
        meta = chunk.get("metadata", {}) or {}
        filename = meta.get("filename", "Unknown")
        page = meta.get("page", "?")
        content = chunk.get("page_content", "")
        
        # 1. Text format for the AI to read
        context_parts.append(f"--- Source {i+1}: {filename} (Page {page}) ---\n{content}")
        
        # 2. Structured data for the UI to display citations
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