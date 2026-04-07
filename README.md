# 📄 Document QA System (RAG-based)

A smart AI-powered application that allows users to upload a PDF and ask questions about its content. The system uses Retrieval-Augmented Generation (RAG) to provide context-aware answers.

---

## 🌟 Features

- 📂 Upload any PDF document
- 💬 Ask questions in natural language
- 🧠 Context-aware answers using AI
- 🔍 Semantic search using embeddings
- ⚡ Fast retrieval with FAISS
- 🎨 Clean dark-themed UI using Streamlit

---

## 🧠 How It Works

1. **Document Loading**
   - PDF is loaded using PyPDFLoader

2. **Text Chunking**
   - Document is split into smaller chunks for better processing

3. **Embeddings**
   - Each chunk is converted into vector form using sentence-transformers

4. **Vector Database**
   - FAISS is used to store and retrieve similar chunks

5. **Query Processing**
   - User query is matched with relevant chunks

6. **Answer Generation**
   - A language model generates answers based on retrieved context

---

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- HuggingFace Transformers
- Sentence Transformers

---

## 📂 Project Structure
