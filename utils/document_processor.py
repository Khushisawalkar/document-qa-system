"""
Document processing pipeline — PDF, DOCX, TXT support.
Optimized for RAG retrieval quality.
"""

import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


# ─────────────────────────────────────────────────────────────
# Text Cleaning
# ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Clean extracted text for better embeddings and retrieval.
    """

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove weird unicode artifacts
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # Remove repeated dots
    text = re.sub(r"\.{2,}", ".", text)

    return text.strip()


# ─────────────────────────────────────────────────────────────
# PDF Loader
# ─────────────────────────────────────────────────────────────
def load_pdf(filepath: str) -> List[Document]:
    """
    Load PDF documents with page tracking.
    """

    loader = PyPDFLoader(filepath)

    docs = loader.load()

    for doc in docs:

        # Clean text
        doc.page_content = clean_text(
            doc.page_content
        )

        # Metadata
        doc.metadata.setdefault("page", 1)

        doc.metadata["file_type"] = "pdf"

    return docs


# ─────────────────────────────────────────────────────────────
# DOCX Loader
# ─────────────────────────────────────────────────────────────
def load_docx(filepath: str) -> List[Document]:
    """
    Load DOCX documents with simulated page grouping.
    """

    try:
        import docx

    except ImportError:

        raise ImportError(
            "python-docx required. Install using:\n"
            "pip install python-docx"
        )

    doc = docx.Document(filepath)

    pages = []

    text_blocks = []

    page_num = 1

    for para in doc.paragraphs:

        text = para.text.strip()

        if text:

            text_blocks.append(
                clean_text(text)
            )

        # Simulate page every ~50 paragraphs
        if len(text_blocks) >= 50:

            pages.append(
                Document(
                    page_content="\n".join(text_blocks),
                    metadata={
                        "source": filepath,
                        "page": page_num,
                        "file_type": "docx",
                    },
                )
            )

            text_blocks = []

            page_num += 1

    # Remaining content
    if text_blocks:

        pages.append(
            Document(
                page_content="\n".join(text_blocks),
                metadata={
                    "source": filepath,
                    "page": page_num,
                    "file_type": "docx",
                },
            )
        )

    return pages


# ─────────────────────────────────────────────────────────────
# TXT Loader
# ─────────────────────────────────────────────────────────────
def load_txt(filepath: str) -> List[Document]:
    """
    Load TXT / Markdown documents.
    """

    with open(
        filepath,
        "r",
        encoding="utf-8",
        errors="replace"
    ) as f:

        content = clean_text(
            f.read()
        )

    artificial_page_size = 3000

    pages = []

    for i, start in enumerate(
        range(0, len(content), artificial_page_size)
    ):

        chunk = content[start:start + artificial_page_size]

        pages.append(
            Document(
                page_content=chunk,
                metadata={
                    "source": filepath,
                    "page": i + 1,
                    "file_type": "txt",
                },
            )
        )

    # Empty fallback
    if not pages:

        pages.append(
            Document(
                page_content=content,
                metadata={
                    "source": filepath,
                    "page": 1,
                    "file_type": "txt",
                },
            )
        )

    return pages


# ─────────────────────────────────────────────────────────────
# Universal Loader
# ─────────────────────────────────────────────────────────────
def load_document(filepath: str) -> List[Document]:
    """
    Automatically detect and load supported document type.
    """

    ext = Path(filepath).suffix.lower()

    loaders = {
        ".pdf": load_pdf,
        ".docx": load_docx,
        ".doc": load_docx,
        ".txt": load_txt,
        ".md": load_txt,
    }

    loader_fn = loaders.get(ext)

    if not loader_fn:

        raise ValueError(
            f"Unsupported file type: {ext}"
        )

    docs = loader_fn(filepath)

    filename = Path(filepath).name

    # Attach metadata
    for doc in docs:

        doc.metadata.setdefault(
            "file_type",
            ext.lstrip(".")
        )

        doc.metadata["filename"] = filename

    return docs


# ─────────────────────────────────────────────────────────────
# Smart Chunking
# ─────────────────────────────────────────────────────────────
def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1200,
    chunk_overlap: int = 250,
) -> List[Document]:
    """
    Split documents into semantically meaningful chunks.

    Larger chunks + overlap improve RAG answer quality.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,

        separators=[
            "\n\n",
            "\n",
            ". ",
            "! ",
            "? ",
            "; ",
            ", ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(
        documents
    )

    # Add chunk metadata
    for i, chunk in enumerate(chunks):

        chunk.metadata["chunk_id"] = i

        # Store chunk size
        chunk.metadata["chunk_length"] = len(
            chunk.page_content
        )

    return chunks