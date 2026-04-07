from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import pipeline


def load_and_prepare_data():
    # Load PDF
    loader = PyPDFLoader("notes.pdf")
    documents = loader.load()

    print(f"\nTotal pages loaded: {len(documents)}")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    return chunks


def create_vector_store(chunks):
    # Embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create FAISS DB
    vector_db = FAISS.from_documents(chunks, embeddings)

    return vector_db


def load_llm():
    # Local LLM (free)
    generator = pipeline(
        "text-generation",
        model="distilgpt2"
    )
    return generator


def ask_questions(vector_db, generator):
    while True:
        query = input("\nAsk something (or type 'exit'): ")

        if query.lower() == "exit":
            print("Exiting...")
            break

        # Retrieve relevant chunks
        results = vector_db.similarity_search(query, k=3)

        context = "\n".join([doc.page_content for doc in results])

        # Prompt
        prompt = f"""
You are an AI assistant. Answer the question using ONLY the context below.

Context:
{context}

Question: {query}

Answer:
"""

        # Generate response
        response = generator(
            prompt,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7
        )

        print("\nAI Answer:\n")
        print(response[0]['generated_text'])


def main():
    print("🚀 Starting Document QA System...")

    chunks = load_and_prepare_data()
    vector_db = create_vector_store(chunks)
    generator = load_llm()

    ask_questions(vector_db, generator)


if __name__ == "__main__":
    main()