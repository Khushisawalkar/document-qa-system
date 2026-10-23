import os
import io
import re
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
                # Clean up null characters or unusual control bytes
                clean_text = text.replace("\x00", "").strip()
                if clean_text:
                    docs.append({
                        "page_content": clean_text,
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

        clean_text = text.replace("\x00", "").strip()
        if clean_text:
            docs.append({
                "page_content": clean_text,
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

            paras = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                    if row_text:
                        paras.append(row_text)
            text = "\n\n".join(paras).replace("\x00", "").strip()
            if text:
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


def _split_text_recursively(
    text: str, 
    chunk_size: int, 
    chunk_overlap: int, 
    separators: List[str]
) -> List[str]:
    """
    Core recursive splitter that splits by the highest priority separator
    before falling back to finer-grained splits.
    """
    final_chunks = []
    
    # Base separator selection
    separator = separators[-1]
    new_separators = []
    for i, sep in enumerate(separators):
        if sep == "":
            separator = ""
            break
        if sep in text:
            separator = sep
            new_separators = separators[i + 1:]
            break

    splits = text.split(separator) if separator else list(text)

    # Recombine small splits up to chunk_size with overlap
    good_splits = []
    for s in splits:
        if s:
            good_splits.append(s)

    current_chunk = []
    current_len = 0

    for s in good_splits:
        s_len = len(s)
        sep_len = len(separator) if current_chunk else 0
        
        if current_len + s_len + sep_len <= chunk_size:
            current_chunk.append(s)
            current_len += s_len + sep_len
        else:
            if current_chunk:
                merged = separator.join(current_chunk).strip()
                if merged:
                    final_chunks.append(merged)
                
                # Keep overlap from the end of the previous chunk
                overlap_accum = []
                overlap_len = 0
                for item in reversed(current_chunk):
                    if overlap_len + len(item) + len(separator) <= chunk_overlap:
                        overlap_accum.insert(0, item)
                        overlap_len += len(item) + len(separator)
                    else:
                        break
                current_chunk = overlap_accum
                current_len = overlap_len

            # If an individual split is still too large, recurse with finer separator
            if s_len > chunk_size:
                if new_separators:
                    sub_chunks = _split_text_recursively(s, chunk_size, chunk_overlap, new_separators)
                    final_chunks.extend(sub_chunks)
                else:
                    # Hard character split as final fallback
                    for st in range(0, s_len, max(1, chunk_size - chunk_overlap)):
                        final_chunks.append(s[st:st + chunk_size].strip())
            else:
                current_chunk.append(s)
                current_len += s_len + (len(separator) if len(current_chunk) > 1 else 0)

    if current_chunk:
        merged = separator.join(current_chunk).strip()
        if merged:
            final_chunks.append(merged)

    return final_chunks


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split text into semantically cohesive chunks using recursive hierarchical splitting:
    Paragraphs -> Newlines -> Sentences -> Words.
    """
    if not text:
        return []
        
    text_length = len(text)
    if text_length <= chunk_size:
        clean = text.strip()
        return [clean] if clean else []

    separators = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]
    raw_chunks = _split_text_recursively(text, chunk_size, chunk_overlap, separators)
    
    # Deduplicate empty or whitespace-only chunks
    cleaned = [c.strip() for c in raw_chunks if c and len(c.strip()) > 10]
    return cleaned if cleaned else [text[:chunk_size].strip()]


def chunk_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Split large documents into smaller semantic chunks for high-precision retrieval.
    Preserves and enriches metadata for each chunk.
    """
    all_chunks = []
    for doc in documents:
        content = doc.get("page_content", "")
        meta = doc.get("metadata", {})
        text_chunks = chunk_text(content, chunk_size=1000, chunk_overlap=200)
        
        for idx, chunk in enumerate(text_chunks):
            chunk_meta = meta.copy()
            chunk_meta["chunk_index"] = idx + 1
            chunk_meta["total_chunks_in_doc"] = len(text_chunks)
            all_chunks.append({
                "page_content": chunk,
                "metadata": chunk_meta
            })
    return all_chunks