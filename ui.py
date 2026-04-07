import streamlit as st
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

st.set_page_config(page_title="Document QA", layout="wide")

# 🌑 DARK PREMIUM UI
st.markdown("""
<style>

/* Base */
html, body, [class*="css"] {
    background-color: #0f1117;
    color: #e6e6e6;
    font-family: 'Inter', sans-serif;
}

/* Title */
h1 {
    text-align: center;
    color: #9bbf9e;
}

/* Subtitle */
p {
    text-align: center;
    color: #a8b3a8;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161a21;
    border-right: 1px solid #2a2f3a;
}

/* Chat bubbles */
[data-testid="stChatMessage"] {
    background-color: #1c212b;
    border-radius: 14px;
    padding: 12px;
    margin-bottom: 10px;
    border: 1px solid #2a2f3a;
    animation: fadeIn 0.3s ease;
}

/* Input */
textarea {
    background-color: #1c212b !important;
    color: #e6e6e6 !important;
    border-radius: 10px !important;
    border: 1px solid #2a2f3a !important;
}

/* Button */
button {
    background-color: #7a9e7e !important;
    color: white !important;
    border-radius: 10px !important;
    border: none !important;
}

/* Upload box */
[data-testid="stFileUploader"] {
    background-color: #1c212b;
    border: 2px dashed #2a2f3a;
    border-radius: 10px;
    padding: 10px;
}

/* Smooth animation */
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(6px);}
    to {opacity: 1; transform: translateY(0);}
}

</style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
<h1>📄 Document QA System</h1>
<p>Ask your document anything</p>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Upload", type="pdf")

# Session state
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# LLM
@st.cache_resource
def load_llm():
    return pipeline("text-generation", model="distilgpt2")

llm = load_llm()

# Process PDF
if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Processing document..."):
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        st.session_state.vector_db = FAISS.from_documents(chunks, embeddings)

    st.sidebar.success("Ready")

# Chat input
query = st.chat_input("Ask something...")

if query and st.session_state.vector_db:
    st.session_state.chat_history.append(("user", query))

    results = st.session_state.vector_db.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in results])

    prompt = f"""
Answer using ONLY the context.

If not found say: Not found in document.

Context:
{context}

Question: {query}

Answer:
"""

    response = llm(
        prompt,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.7
    )

    full = response[0]["generated_text"]
    answer = full.split("Answer:")[-1].strip()

    st.session_state.chat_history.append(("assistant", answer))

# Chat display
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        if role == "assistant":
            placeholder = st.empty()
            text = ""
            for ch in msg:
                text += ch
                placeholder.markdown(text)
                time.sleep(0.004)
        else:
            st.markdown(msg)