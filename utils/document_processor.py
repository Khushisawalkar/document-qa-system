import os
import io
from typing import List, Dict, Any, Union
from pypdf import PdfReader
import docx

def load_document(
    file_source: Union[str, bytes, io.BytesIO], 
    filename: str = None, 
    file_type: str = None
) -> List[Dict[str, Any]]:
    """
    Load a PDF, DOCX, TXT, or Markdown document directly from a filepath,
    in-memory bytes, or a file-like stream without requiring disk I/O.
    
    Each piece is a dictionary:
    {"page_content": text, "metadata": {"filename": name, "page": num, "file_type": type}}
    """
    # Determine filename and extension
    if isinstance(file_source, str):
        actual_filename = filename or os.path.basename(file_source)
        ext = (file_type or os.path.splitext(file_source)[1]).lower()
        if not ext.startswith("."):
            ext = f".{ext}"
        is_path = True
    else:
        actual_filename = filename or "document"
        ext = (file_type or os.path.splitext(actual_filename)[1]).lower()
        if not ext.startswith("."):
            ext = f".{ext}"
        is_path = False

    docs = []

    if ext == ".pdf":
        try:
            if is_path:
                reader = PdfReader(file_source)
            elif isinstance(file_source, bytes):
                reader = PdfReader(io.BytesIO(file_source))
            else:
                reader = PdfReader(file_source)

            for i, page in enumerate(reader.pages):
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""
                if text.strip():
                    docs.append({
                        "page_content": text,
                        "metadata": {"filename": actual_filename, "page": i + 1, "file_type": "PDF"}
                    })
        except Exception as e:
            raise ValueError(f"Error reading PDF '{actual_filename}': {str(e)}")

    elif ext in [".txt", ".md"]:
        text = ""
        try:
            if is_path:
                try:
                    with open(file_source, "r", encoding="utf-8") as f:
                        text = f.read()
                except UnicodeDecodeError:
                    with open(file_source, "r", encoding="latin-1", errors="replace") as f:
                        text = f.read()
            else:
                raw_bytes = file_source if isinstance(file_source, bytes) else file_source.read()
                try:
                    text = raw_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    text = raw_bytes.decode("latin-1", errors="replace")
        except Exception as e:
            raise ValueError(f"Error reading text file '{actual_filename}': {str(e)}")

        if text.strip():
            docs.append({
                "page_content": text,
                "metadata": {"filename": actual_filename, "page": 1, "file_type": ext.lstrip(".").upper()}
            })

    elif ext == ".docx":
        try:
            if is_path:
                doc = docx.Document(file_source)
            elif isinstance(file_source, bytes):
                doc = docx.Document(io.BytesIO(file_source))
            else:
                doc = docx.Document(file_source)

            paras = [para.text for para in doc.paragraphs if para.text.strip()]
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                    if row_text:
                        paras.append(row_text)
            text = "\n".join(paras)
            if text.strip():
                docs.append({
                    "page_content": text,
                    "metadata": {"filename": actual_filename, "page": 1, "file_type": "DOCX"}
                })
        except Exception as e:
            raise ValueError(f"Error reading DOCX '{actual_filename}': {str(e)}")

    elif ext == ".doc":
        raise ValueError(
            f"Legacy Word format (.doc) for '{actual_filename}' is not supported. "
            "Please convert the file to .docx or .pdf."
        )

    else:
        raise ValueError(f"Unsupported file type: {ext}")
        
    return docs


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split long text into overlapping chunks, attempting to break cleanly
    at natural paragraph or sentence boundaries when possible.
    """
    if not text:
        return []
        
    text_length = len(text)
    if text_length <= chunk_size:
        clean = text.strip()
        return [clean] if clean else []
        
    chunks = []
    start = 0
    step = max(50, chunk_size - chunk_overlap)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        
        # If we are not at the end of text, try to split at a clean boundary
        if end < text_length:
            boundary = -1
            # Search backwards within the overlap window for newline or sentence end
            search_window = text[max(start, end - 120):end]
            nl_pos = search_window.rfind("\n")
            if nl_pos != -1:
                boundary = max(start, end - 120) + nl_pos + 1
            else:
                dot_pos = search_window.rfind(". ")
                if dot_pos != -1:
                    boundary = max(start, end - 120) + dot_pos + 2
                    
            if boundary > start + 200:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
            
        if end >= text_length:
            break
            
        start = max(start + 1, end - chunk_overlap)
        
    return chunks


def chunk_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Split large documents into smaller chunks for better search accuracy.
    Preserves the metadata (like filename and page number) for each chunk.
    """
    all_chunks = []
    for doc in documents:
        content = doc.get("page_content", "")
        meta = doc.get("metadata", {})
        text_chunks = chunk_text(content, chunk_size=1000, chunk_overlap=200)
        for chunk in text_chunks:
            all_chunks.append({
                "page_content": chunk,
                "metadata": meta.copy()
            })
    return all_chunks