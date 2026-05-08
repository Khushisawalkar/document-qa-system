# 🧠 DocMind — AI Document Q&A System

DocMind is an AI-powered multi-document question answering system built using **Streamlit**, **FAISS**, **LangChain**, and **Google Gemini AI**.

Upload PDFs, DOCX, TXT, or Markdown files and chat with your documents using intelligent semantic search and AI-generated answers.

---

# ✨ Features

- 📄 PDF, DOCX, TXT, MD support
- 🧠 AI-powered question answering using Gemini
- 🔍 Semantic search with FAISS vector database
- 📚 Multi-document querying
- 📎 Source citations with page references
- 📝 AI-generated summaries
- 🏷️ Automatic keyword extraction
- 🌍 Multi-language support
- 💬 Chat memory for contextual conversations
- 🎨 Premium dark editorial UI

---

# 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- HuggingFace Embeddings
- Google Gemini API
- PyPDF
- python-docx

---

# 📂 Project Structure

```bash
docmind/
│
├── ui.py
├── app.py
├── requirements.txt
├── README.md
├── .env
│
└── utils/
    ├── __init__.py
    ├── document_processor.py
    ├── vector_store.py
    └── llm_handler.py
