"""
DocMind CLI — Terminal-based Document QA.
Usage: python app.py notes.pdf
"""

import sys
import os
from pathlib import Path
from utils.document_processor import load_document, chunk_documents
from utils.vector_store import build_vector_store, similarity_search, format_context
from utils.llm_handler import generate_answer


def main():
    filepath = sys.argv[1] if len(sys.argv) > 1 else "notes.pdf"

    if not Path(filepath).exists():
        print(f"❌ File not found: {filepath}")
        sys.exit(1)

    print(f"🧠 DocMind CLI — {filepath}")
    print("Loading and indexing document...")

    docs = load_document(filepath)
    chunks = chunk_documents(docs)

    print(f"✅ {len(docs)} pages → {len(chunks)} chunks indexed")
    print("Type 'exit' to quit.\n")

    vector_db = build_vector_store(chunks)
    history = []

    while True:
        try:
            query = input("Ask: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query or query.lower() in ("exit", "quit"):
            print("Bye!")
            break

        results = similarity_search(vector_db, query, k=4)
        context, citations = format_context(results)
        answer = generate_answer(query, context, chat_history=history)

        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})

        print(f"\n{'─'*60}")
        print(answer)
        print(f"\nSources: {', '.join(c['label'] for c in citations)}")
        print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
