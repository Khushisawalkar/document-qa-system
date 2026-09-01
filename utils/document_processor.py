import os
from pypdf import PdfReader
import docx

def load_document(filepath: str) -> list:
    """
    Load a PDF, DOCX, or TXT document and return a list of document pieces with metadata.
    Each piece is a dictionary: {"page_content": text, "metadata": {"filename": name, "page": num}}
    """
    ext = os.path.splitext(filepath)[1].lower()
    filename = os.path.basename(filepath)
    docs = []

    if ext == ".pdf":
        reader = PdfReader(filepath)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append({
                    "page_content": text,
                    "metadata": {"filename": filename, "page": i + 1, "file_type": "PDF"}
                })
    elif ext in [".txt", ".md"]:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        if text.strip():
            docs.append({
                "page_content": text,
                "metadata": {"filename": filename, "page": 1, "file_type": ext.lstrip(".").upper()}
            })
    elif ext in [".docx", ".doc"]:
        doc = docx.Document(filepath)
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        if text.strip():
            docs.append({
                "page_content": text,
                "metadata": {"filename": filename, "page": 1, "file_type": ext.lstrip(".").upper()}
            })
    else:
        raise ValueError(f"Unsupported file type: {ext}")
        
    return docs

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """
    Split a long text into smaller overlapping chunks.
    This simple character-based chunking is easy to explain in interviews.
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        if end >= text_length:
            break
        # Move forward by (chunk_size - chunk_overlap)
        start += chunk_size - chunk_overlap
        
    return chunks

def chunk_documents(documents: list) -> list:
    """
    Split large documents into smaller chunks for better search accuracy.
    Preserves the metadata (like filename and page number) for each chunk.
    """
    all_chunks = []
    for doc in documents:
        text_chunks = chunk_text(doc["page_content"], chunk_size=1000, chunk_overlap=200)
        for chunk in text_chunks:
            all_chunks.append({
                "page_content": chunk,
                "metadata": doc["metadata"].copy()
            })
    return all_chunks