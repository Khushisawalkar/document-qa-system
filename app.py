from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def main():
    # Load PDF
    loader = PyPDFLoader("notes.pdf")
    documents = loader.load()

    print(f"\nTotal pages: {len(documents)}")

    # Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    print(f"\nTotal chunks created: {len(chunks)}\n")

    # Show sample chunks
    for i, chunk in enumerate(chunks[:3]):
        print(f"--- Chunk {i+1} ---")
        print(chunk.page_content)
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()