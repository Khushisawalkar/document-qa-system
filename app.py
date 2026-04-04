from langchain_community.document_loaders import PyPDFLoader

def load_pdf():
    loader = PyPDFLoader("notes.pdf")
    documents = loader.load()

    print(f"\nTotal pages loaded: {len(documents)}\n")

    for i, doc in enumerate(documents):
        print(f"--- Page {i+1} ---")
        print(doc.page_content[:300])
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    load_pdf()