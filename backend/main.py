import os
import shutil
import tempfile
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils.document_processor import load_document, chunk_documents
from utils.vector_store import build_vector_store, merge_vector_stores, format_context, multi_query_search
from utils.llm_handler import decompose_query, generate_answer, generate_summary, extract_keywords

app = FastAPI(title="DocMind API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session management for vector stores (Multi-user concurrency)
# session_id -> FAISS vector store
sessions: Dict[str, Any] = {}
# session_id -> chat history
chat_histories: Dict[str, List[Dict[str, str]]] = {}

class QueryRequest(BaseModel):
    session_id: str
    query: str
    language: str = "English"

@app.post("/api/upload")
async def upload_documents(
    session_id: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    Upload and process multiple documents for a specific session.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    all_new_chunks = []
    docs_metadata = []

    for uf in files:
        suffix = os.path.splitext(uf.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(uf.file, tmp)
            tmp_path = tmp.name

        try:
            docs = load_document(tmp_path)
            chunks = chunk_documents(docs)
            for c in chunks:
                c.metadata["filename"] = uf.filename

            all_new_chunks.extend(chunks)

            sample_text = " ".join([c.page_content for c in chunks[:8]])
            try:
                summary = generate_summary(sample_text, uf.filename)
                keywords = extract_keywords(sample_text)
            except Exception:
                summary = "Summary unavailable."
                keywords = []

            docs_metadata.append({
                "filename": uf.filename,
                "file_type": suffix.lstrip(".").upper(),
                "pages": max((c.metadata.get("page", 1) for c in docs), default=1),
                "chunks": len(chunks),
                "summary": summary,
                "keywords": keywords,
            })
        except Exception as e:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise HTTPException(status_code=500, detail=f"Error processing {uf.filename}: {str(e)}")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    if all_new_chunks:
        new_store = build_vector_store(all_new_chunks)
        if session_id in sessions:
            sessions[session_id] = merge_vector_stores(sessions[session_id], new_store)
        else:
            sessions[session_id] = new_store
            chat_histories[session_id] = []

    return {"status": "success", "indexed_documents": len(docs_metadata), "metadata": docs_metadata}

@app.post("/api/query")
async def query_document(request: QueryRequest):
    """
    Perform decomposed semantic search and generate answer.
    """
    session_id = request.session_id
    query = request.query

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Please upload documents first.")

    vector_db = sessions[session_id]
    history = chat_histories.get(session_id, [])

    # 1. Query Decomposition
    sub_queries = decompose_query(query)

    # 2. Multi-query Search
    results = multi_query_search(vector_db, sub_queries, k=4)
    context, citations = format_context(results)

    # 3. Generate Answer
    answer = generate_answer(query, context, chat_history=history, language=request.language)

    # 4. Update History
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})
    chat_histories[session_id] = history

    return {
        "answer": answer,
        "citations": citations,
        "sub_queries": sub_queries
    }

@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    """
    Clear session data.
    """
    if session_id in sessions:
        del sessions[session_id]
    if session_id in chat_histories:
        del chat_histories[session_id]
    return {"status": "success"}
