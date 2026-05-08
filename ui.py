"""
DocMind — AI Document Q&A System
Premium dark editorial UI with multi-doc support, citations, summaries.
"""

import os
import time
import tempfile
from pathlib import Path
from typing import Optional
import streamlit as st

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind · AI Document QA",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    background: #0a0b0f !important;
    color: #c8cdd8;
    font-family: 'DM Sans', sans-serif;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0d0e13 !important;
    border-right: 1px solid #1e2030;
}
section[data-testid="stSidebar"] * {
    color: #9aa3b8 !important;
}

/* ── Header ── */
.docmind-header {
    padding: 2rem 0 1.5rem;
    text-align: center;
    border-bottom: 1px solid #1e2030;
    margin-bottom: 2rem;
}
.docmind-logo {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #e8eaf0;
    line-height: 1;
}
.docmind-logo span {
    color: #5b8af5;
}
.docmind-tagline {
    font-size: 0.85rem;
    color: #4a5168;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.4rem;
    font-weight: 300;
}

/* ── Doc Pills ── */
.doc-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #141620;
    border: 1px solid #252840;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.75rem;
    color: #7b85a0;
    margin: 3px 2px;
    font-family: 'DM Sans', sans-serif;
}
.doc-pill .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #5b8af5;
    flex-shrink: 0;
}
.doc-pill.pdf .dot { background: #f55b5b; }
.doc-pill.docx .dot { background: #5bf5a0; }
.doc-pill.txt .dot { background: #f5d45b; }

/* ── Chat Messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

.msg-user {
    background: #141620;
    border: 1px solid #1e2030;
    border-radius: 12px 12px 4px 12px;
    padding: 14px 18px;
    margin: 8px 0 8px 10%;
    color: #d0d5e8;
    font-size: 0.95rem;
    line-height: 1.6;
    animation: slideIn 0.2s ease;
}

.msg-assistant {
    background: #111318;
    border: 1px solid #1a1d2a;
    border-left: 3px solid #5b8af5;
    border-radius: 4px 12px 12px 12px;
    padding: 16px 20px;
    margin: 8px 10% 8px 0;
    color: #bcc3d8;
    font-size: 0.93rem;
    line-height: 1.7;
    animation: slideIn 0.2s ease;
}
.msg-assistant code {
    background: #1a1d2a;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 0.85em;
    color: #88b4f8;
}
.msg-assistant strong { color: #d8dde8; }
.msg-assistant ul { padding-left: 1.2rem; }

/* ── Citation Cards ── */
.citation-section {
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid #1a1d2a;
}
.citation-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #3d4460;
    margin-bottom: 6px;
    font-weight: 500;
}
.citation-card {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    background: #0d0f16;
    border: 1px solid #1e2030;
    border-radius: 6px;
    padding: 8px 12px;
    margin: 4px 0;
    font-size: 0.78rem;
}
.citation-num {
    color: #5b8af5;
    font-weight: 700;
    font-family: 'Syne', sans-serif;
    min-width: 20px;
}
.citation-meta {
    color: #4a5168;
    flex: 1;
}
.citation-meta .fname {
    color: #7b85a0;
    font-weight: 500;
}
.citation-meta .snippet {
    color: #3d4460;
    font-size: 0.72rem;
    margin-top: 2px;
    font-style: italic;
}
.score-badge {
    background: #141620;
    border: 1px solid #252840;
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 0.68rem;
    color: #3d4460;
    white-space: nowrap;
}

/* ── Summary Box ── */
.summary-box {
    background: #0d0f16;
    border: 1px solid #1e2030;
    border-top: 3px solid #5b8af5;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 12px 0;
    font-size: 0.88rem;
    line-height: 1.7;
    color: #9aa3b8;
}
.summary-box strong { color: #c8cdd8; }

/* ── Stat Cards ── */
.stat-row {
    display: flex;
    gap: 10px;
    margin: 12px 0;
    flex-wrap: wrap;
}
.stat-card {
    background: #0d0f16;
    border: 1px solid #1e2030;
    border-radius: 8px;
    padding: 10px 14px;
    text-align: center;
    flex: 1;
    min-width: 80px;
}
.stat-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #5b8af5;
}
.stat-lbl {
    font-size: 0.68rem;
    color: #3d4460;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 2px;
}

/* ── Input ── */
[data-testid="stChatInput"] textarea {
    background: #0d0f16 !important;
    border: 1px solid #252840 !important;
    border-radius: 10px !important;
    color: #c8cdd8 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #5b8af5 !important;
    box-shadow: 0 0 0 2px rgba(91,138,245,0.1) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #141620 !important;
    color: #7b85a0 !important;
    border: 1px solid #252840 !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.83rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #1a1d2a !important;
    color: #c8cdd8 !important;
    border-color: #3d4a70 !important;
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    background: #0d0f16 !important;
    border: 1px dashed #252840 !important;
    border-radius: 10px !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div {
    background: #0d0f16 !important;
    border: 1px solid #1e2030 !important;
    border-radius: 8px !important;
}

/* ── Empty State ── */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: #2a2f3a;
}
.empty-state .icon { font-size: 3rem; margin-bottom: 1rem; }
.empty-state .title {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    color: #3d4460;
    margin-bottom: 0.5rem;
}
.empty-state p { font-size: 0.85rem; color: #252840; }

/* ── Keyword Tags ── */
.kw-tag {
    display: inline-block;
    background: #0d0f16;
    border: 1px solid #1e2030;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.72rem;
    color: #4a5168;
    margin: 2px;
}

/* ── Dividers ── */
hr { border-color: #1e2030 !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #5b8af5 !important; }

/* ── Animations ── */
@keyframes slideIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0a0b0f; }
::-webkit-scrollbar-thumb { background: #1e2030; border-radius: 3px; }

/* ── Toast / Info ── */
[data-testid="stAlert"] {
    background: #0d0f16 !important;
    border: 1px solid #1e2030 !important;
    border-radius: 8px !important;
    color: #9aa3b8 !important;
}

/* ── Tabs ── */
.stTabs [data-testid="stTab"] {
    color: #4a5168 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.83rem !important;
}
.stTabs [aria-selected="true"] {
    color: #5b8af5 !important;
    border-bottom: 2px solid #5b8af5 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Init Session State ────────────────────────────────────────────────────────
defaults = {
    "vector_db": None,
    "chat_history": [],
    "documents_meta": [],   # [{filename, pages, chunks, file_type, keywords, summary}]
    "total_chunks": 0,
    "language": "English",
    "show_citations": True,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Lazy Imports (avoid blocking) ────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _get_embeddings():
    from utils.vector_store import get_embeddings
    return get_embeddings()


# ─── Helpers ──────────────────────────────────────────────────────────────────
def get_doc_type_class(file_type: str) -> str:
    ft = file_type.lower().lstrip(".")
    return ft if ft in ("pdf", "docx", "txt") else "txt"


def render_doc_pill(filename: str, file_type: str):
    cls = get_doc_type_class(file_type)
    icon = {"pdf": "📕", "docx": "📘", "txt": "📄"}.get(cls, "📄")
    return f'<span class="doc-pill {cls}"><span class="dot"></span>{icon} {filename}</span>'


def stream_text(placeholder, text: str, delay: float = 0.006):
    buf = ""
    for ch in text:
        buf += ch
        placeholder.markdown(f'<div class="msg-assistant">{buf}▌</div>', unsafe_allow_html=True)
        time.sleep(delay)
    placeholder.markdown(f'<div class="msg-assistant">{buf}</div>', unsafe_allow_html=True)


def render_citations(citations):
    if not citations:
        return ""
    cards = ""
    for c in citations:
        score_pct = int(c["score"] * 100)
        cards += f"""
        <div class="citation-card">
            <div class="citation-num">[{c['index']}]</div>
            <div class="citation-meta">
                <div class="fname">{c['filename']} · Page {c['page']}</div>
                <div class="snippet">{c['snippet']}</div>
            </div>
            <div class="score-badge">{score_pct}% match</div>
        </div>"""
    return f"""<div class="citation-section">
        <div class="citation-label">📎 Sources</div>
        {cards}
    </div>"""


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="docmind-header">
    <div class="docmind-logo">Doc<span>Mind</span></div>
    <div class="docmind-tagline">AI-Powered Document Intelligence</div>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT",
        type=["pdf", "docx", "doc", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("⚡ Process Documents", use_container_width=True):
            from utils.document_processor import load_document, chunk_documents
            from utils.vector_store import build_vector_store, merge_vector_stores
            from utils.llm_handler import generate_summary, extract_keywords

            progress = st.progress(0, text="Loading embeddings...")
            _get_embeddings()  # Warm up

            new_metas = []
            all_new_chunks = []

            for idx, uf in enumerate(uploaded_files):
                progress.progress((idx / len(uploaded_files)) * 0.6, text=f"Processing {uf.name}...")

                # Already processed?
                already = any(m["filename"] == uf.name for m in st.session_state.documents_meta)
                if already:
                    continue

                suffix = Path(uf.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uf.read())
                    tmp_path = tmp.name

                try:
                    docs = load_document(tmp_path)
                    chunks = chunk_documents(docs)

                    # Tag filename properly
                    for c in chunks:
                        c.metadata["filename"] = uf.name

                    all_new_chunks.extend(chunks)

                    # Summary + keywords from first chunks
                    sample_text = " ".join([c.page_content for c in chunks[:8]])

                    new_metas.append({
                        "filename": uf.name,
                        "file_type": suffix.lstrip(".").upper(),
                        "pages": max((c.metadata.get("page", 1) for c in docs), default=1),
                        "chunks": len(chunks),
                        "summary": None,
                        "keywords": [],
                        "sample_text": sample_text,
                    })

                except Exception as e:
                    st.error(f"❌ {uf.name}: {e}")
                finally:
                    os.unlink(tmp_path)

            if all_new_chunks:
                progress.progress(0.7, text="Building vector index...")
                new_store = build_vector_store(all_new_chunks)
                st.session_state.vector_db = merge_vector_stores(
                    st.session_state.vector_db, new_store
                )
                st.session_state.total_chunks += len(all_new_chunks)

                # Generate summaries + keywords
                for i, meta in enumerate(new_metas):
                    progress.progress(0.75 + i * 0.05, text=f"Summarizing {meta['filename']}...")
                    try:
                        meta["summary"] = generate_summary(meta["sample_text"], meta["filename"])
                        meta["keywords"] = extract_keywords(meta["sample_text"])
                    except Exception:
                        meta["summary"] = "Summary unavailable."
                        meta["keywords"] = []
                    del meta["sample_text"]
                    st.session_state.documents_meta.append(meta)

                progress.progress(1.0, text="Done!")
                time.sleep(0.5)
                progress.empty()
                st.success(f"✅ {len(new_metas)} document(s) indexed")

    # Document list
    if st.session_state.documents_meta:
        st.markdown("---")
        st.markdown("**Loaded Documents**")
        for m in st.session_state.documents_meta:
            cls = get_doc_type_class(m["file_type"])
            icon = {"PDF": "📕", "DOCX": "📘", "DOC": "📘", "TXT": "📄", "MD": "📄"}.get(m["file_type"], "📄")
            st.markdown(
                f'<div class="doc-pill {cls}"><span class="dot"></span>{icon} {m["filename"]}<br>'
                f'<small style="color:#2a2f3a">{m["pages"]} pages · {m["chunks"]} chunks</small></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Settings
    st.markdown("**⚙️ Settings**")
    st.session_state.language = st.selectbox(
        "Response Language",
        ["English", "Hindi", "French", "Spanish", "German", "Japanese", "Chinese"],
        index=0,
    )
    st.session_state.show_citations = st.toggle("Show Source Citations", value=True)

    # Stats
    if st.session_state.documents_meta:
        st.markdown("---")
        total_docs = len(st.session_state.documents_meta)
        total_pages = sum(m["pages"] for m in st.session_state.documents_meta)
        total_chunks = st.session_state.total_chunks
        st.markdown(
            f'<div class="stat-row">'
            f'<div class="stat-card"><div class="stat-val">{total_docs}</div><div class="stat-lbl">Docs</div></div>'
            f'<div class="stat-card"><div class="stat-val">{total_pages}</div><div class="stat-lbl">Pages</div></div>'
            f'<div class="stat-card"><div class="stat-val">{total_chunks}</div><div class="stat-lbl">Chunks</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Clear
    if st.session_state.documents_meta:
        if st.button("🗑️ Clear All", use_container_width=True):
            for k in ["vector_db", "chat_history", "documents_meta", "total_chunks"]:
                st.session_state[k] = None if k == "vector_db" else ([] if k != "total_chunks" else 0)
            st.rerun()

# ─── Main Area ─────────────────────────────────────────────────────────────────
if not st.session_state.documents_meta:
    # Empty state
    st.markdown("""
    <div class="empty-state">
        <div class="icon">🧠</div>
        <div class="title">No documents loaded</div>
        <p>Upload PDF, DOCX, or TXT files from the sidebar<br>to start asking questions about your content.</p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Tabs: Chat | Summaries | Keywords
    tab_chat, tab_summaries, tab_keywords = st.tabs(["💬 Chat", "📋 Summaries", "🏷️ Keywords"])

    # ── Chat Tab ──
    with tab_chat:
        # Suggested questions
        if not st.session_state.chat_history:
            st.markdown('<div style="margin-bottom:1rem;">', unsafe_allow_html=True)
            cols = st.columns(3)
            suggestions = [
                "What is this document about?",
                "Summarize the key points",
                "What are the main conclusions?",
            ]
            for col, sug in zip(cols, suggestions):
                with col:
                    if st.button(sug, use_container_width=True):
                        st.session_state._pending_query = sug
                        st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        # Render history
        for turn in st.session_state.chat_history:
            if turn["role"] == "user":
                st.markdown(f'<div class="msg-user">{turn["content"]}</div>', unsafe_allow_html=True)
            else:
                content = turn["content"]
                citations_html = ""
                if st.session_state.show_citations and turn.get("citations"):
                    citations_html = render_citations(turn["citations"])
                st.markdown(
                    f'<div class="msg-assistant">{content}{citations_html}</div>',
                    unsafe_allow_html=True,
                )

        # Chat input
        query = st.chat_input("Ask anything about your documents...")

        # Handle suggested question click
        if hasattr(st.session_state, "_pending_query") and st.session_state._pending_query:
            query = st.session_state._pending_query
            st.session_state._pending_query = None

        if query and st.session_state.vector_db:
            from utils.vector_store import similarity_search, format_context
            from utils.llm_handler import generate_answer

            # Show user msg
            st.markdown(f'<div class="msg-user">{query}</div>', unsafe_allow_html=True)
            st.session_state.chat_history.append({"role": "user", "content": query})

            # Retrieve
            with st.spinner("Searching documents..."):
                results = similarity_search(st.session_state.vector_db, query, k=5)
                context, citations = format_context(results)

            # Stream answer
            placeholder = st.empty()
            with st.spinner("Generating answer..."):
                try:
                    answer = generate_answer(
                        query, context,
                        chat_history=st.session_state.chat_history,
                        language=st.session_state.language,
                    )
                except Exception as e:
                    answer = f"⚠️ Error: {e}"
                    citations = []

            stream_text(placeholder, answer)

            # Citations
            if st.session_state.show_citations and citations:
                citations_html = render_citations(citations)
                placeholder.markdown(
                    f'<div class="msg-assistant">{answer}{citations_html}</div>',
                    unsafe_allow_html=True,
                )

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "citations": citations,
            })

        elif query and not st.session_state.vector_db:
            st.warning("Please process documents first.")

    # ── Summaries Tab ──
    with tab_summaries:
        if not any(m.get("summary") for m in st.session_state.documents_meta):
            st.info("Summaries are generated during document processing.")
        else:
            for m in st.session_state.documents_meta:
                ft = m["file_type"]
                icon = {"PDF": "📕", "DOCX": "📘", "TXT": "📄"}.get(ft, "📄")
                st.markdown(f"#### {icon} {m['filename']}")
                if m.get("summary"):
                    st.markdown(f'<div class="summary-box">{m["summary"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown("*Summary not available.*")
                st.markdown("---")

    # ── Keywords Tab ──
    with tab_keywords:
        for m in st.session_state.documents_meta:
            ft = m["file_type"]
            icon = {"PDF": "📕", "DOCX": "📘", "TXT": "📄"}.get(ft, "📄")
            st.markdown(f"#### {icon} {m['filename']}")
            if m.get("keywords"):
                tags_html = " ".join(f'<span class="kw-tag">{kw}</span>' for kw in m["keywords"])
                st.markdown(tags_html, unsafe_allow_html=True)
            else:
                st.markdown("*Keywords not available.*")
            st.markdown("")