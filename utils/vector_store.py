class DocumentVectorStore:
    """
    A simplified Document Store that replaces the FAISS vector database.
    This class manages text chunks and retrieves the most similar ones
    using a simple keyword matching algorithm, removing the need for heavy
    machine learning dependencies.
    """
    def __init__(self):
        # Store metadata and original text for each chunk
        self.chunks = []

    def add_chunks(self, chunks: list):
        """
        Add chunks to the store.
        """
        if not chunks:
            return
        
        self.chunks.extend(chunks)

    def similarity_search(self, query: str, k: int = 4) -> list:
        """
        Find the most relevant document chunks for a given user question 
        using a simple keyword overlap scoring mechanism.
        """
        if not self.chunks:
            return []
            
        # Convert query to a set of lowercase words for matching
        # In a real app we might remove stop words, but this is a simple implementation
        query_words = set("".join(c for c in query.lower() if c.isalnum() or c.isspace()).split())
        
        results = []
        for i, chunk in enumerate(self.chunks):
            text = chunk["page_content"].lower()
            text_words = set("".join(c for c in text if c.isalnum() or c.isspace()).split())
            
            # Simple score: how many query words are in the chunk text?
            overlap = len(query_words.intersection(text_words))
            
            # We also add a tiny tie-breaker based on index so order is consistent
            tie_breaker = 1.0 / (1.0 + i)
            
            score = float(overlap) + (tie_breaker * 0.01)
            results.append((chunk, score))
            
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Normalize scores to max 1.0 (or just return as is if max is 0)
        max_score = results[0][1] if results else 0
        if max_score > 0:
            normalized_results = [(chunk, score / max_score) for chunk, score in results[:k]]
        else:
            normalized_results = [(chunk, 0.1) for chunk, score in results[:k]]
            
        return normalized_results

def format_context(results: list) -> tuple:
    """
    Format the search results into a readable string for the AI 
    and extract citation information for the UI.
    """
    context_parts = []
    citations = []
    
    for i, (chunk, score) in enumerate(results):
        meta = chunk["metadata"]
        filename = meta.get("filename", "Unknown")
        page = meta.get("page", "?")
        
        # 1. Text format for the AI to read
        context_parts.append(f"--- Source {i+1}: {filename} (Page {page}) ---\n{chunk['page_content']}")
        
        # 2. Structured data for the UI to display citations
        citations.append({
            "index": i + 1,
            "filename": filename,
            "page": page,
            "score": round(score, 3),
            "snippet": chunk["page_content"][:200].strip() + "...",
            "label": f"[{i+1}] {filename} • Page {page}"
        })
        
    context = "\n\n".join(context_parts)
    return context, citations