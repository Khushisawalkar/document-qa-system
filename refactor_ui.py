import sys

def main():
    with open('ui.py', 'r', encoding='utf-8') as f:
        content = f.read()
        
    # 1. Init Session State
    old_init = '''# ─── Init Session State ────────────────────────────────────────────────────────
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
    return get_embeddings()'''
    
    new_init = '''# ─── Init Session State ────────────────────────────────────────────────────────
import uuid
import requests

API_URL = "http://127.0.0.1:8000/api"

defaults = {
    "session_id": str(uuid.uuid4()),
    "chat_history": [],
    "documents_meta": [],   # [{filename, pages, chunks, file_type, keywords, summary}]
    "total_chunks": 0,
    "language": "English",
    "show_citations": True,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v'''

    content = content.replace(old_init, new_init)

    # 2. Upload block
    import re
    old_upload_match = re.search(r'        if st\.button\("⚡ Process Documents", use_container_width=True\):.*?st\.success\(f"✅ \{len\(new_metas\)\} document\(s\) indexed"\)', content, re.DOTALL)
    
    if old_upload_match:
        old_upload = old_upload_match.group(0)
        new_upload = '''        if st.button("⚡ Process Documents", use_container_width=True):
            progress = st.progress(0, text="Uploading documents...")
            
            files_to_upload = []
            for uf in uploaded_files:
                # Already processed?
                already = any(m["filename"] == uf.name for m in st.session_state.documents_meta)
                if already:
                    continue
                files_to_upload.append(("files", (uf.name, uf.getvalue(), uf.type)))

            if files_to_upload:
                try:
                    res = requests.post(
                        f"{API_URL}/upload",
                        data={"session_id": st.session_state.session_id},
                        files=files_to_upload
                    )
                    if res.status_code == 200:
                        data = res.json()
                        for meta in data["metadata"]:
                            st.session_state.documents_meta.append(meta)
                            st.session_state.total_chunks += meta["chunks"]
                        st.success(f"✅ {data['indexed_documents']} document(s) indexed")
                    else:
                        st.error(f"❌ Upload failed: {res.text}")
                except Exception as e:
                    st.error(f"❌ API Error: {str(e)}")
            
            progress.progress(1.0, text="Done!")
            import time
            time.sleep(0.5)
            progress.empty()'''
        content = content.replace(old_upload, new_upload)

    # 3. Clear Data block
    old_clear = '''        if st.button("🗑️ Clear All Data", use_container_width=True):
            for k in ["vector_db", "chat_history", "documents_meta", "total_chunks"]:
                st.session_state[k] = None if k == "vector_db" else ([] if k != "total_chunks" else 0)
            st.rerun()'''
            
    new_clear = '''        if st.button("🗑️ Clear All Data", use_container_width=True):
            try:
                requests.delete(f"{API_URL}/session/{st.session_state.session_id}")
            except:
                pass
            for k in ["chat_history", "documents_meta", "total_chunks"]:
                st.session_state[k] = [] if k != "total_chunks" else 0
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()'''
            
    content = content.replace(old_clear, new_clear)

    # 4. Query block
    old_query_match = re.search(r'        if query and st\.session_state\.vector_db:.*?st\.warning\("Please process documents first\."\)', content, re.DOTALL)
    
    if old_query_match:
        old_query = old_query_match.group(0)
        new_query = '''        if query and st.session_state.documents_meta:
            # Show user msg
            st.markdown(f'<div class="msg-user">{query}</div>', unsafe_allow_html=True)
            st.session_state.chat_history.append({"role": "user", "content": query})

            # Retrieve
            placeholder = st.empty()
            with st.spinner("Searching and generating answer..."):
                try:
                    res = requests.post(
                        f"{API_URL}/query",
                        json={
                            "session_id": st.session_state.session_id,
                            "query": query,
                            "language": st.session_state.language
                        }
                    )
                    if res.status_code == 200:
                        data = res.json()
                        answer = data["answer"]
                        citations = data["citations"]
                        sub_queries = data.get("sub_queries", [])
                    else:
                        answer = f"⚠️ Error: {res.text}"
                        citations = []
                except Exception as e:
                    answer = f"⚠️ API Error: {str(e)}"
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

        elif query and not st.session_state.documents_meta:
            st.warning("Please process documents first.")'''
            
        content = content.replace(old_query, new_query)

    with open('ui.py', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    main()
