# run.py
import os
import tempfile
import time
import streamlit as st
from src.preprocessing import load_corpus
from src.retrieval import SemanticRetriever, TfIdfRetriever, HybridRetriever
from src.summarizer import meta_summarize

st.set_page_config(page_title="LLM DocSearch & Summarizer", layout="wide")
st.title("📚 LLM Document Search & Summarizer")

# ---- 1️⃣ Mode selection ----
mode = st.radio(
    "Select mode:",
    ("User-uploaded files", "Development backend files")
)

# ---- 2️⃣ Sidebar settings ----
with st.sidebar:
    st.header("Settings")
    retrieval_mode = st.radio(
        "Retrieval Mode",
        ["Semantic (SentenceTransformer)", "Keyword (TF-IDF)", "Hybrid (Semantic+TF-IDF)"],
        index=0
    )

    if retrieval_mode.startswith("Hybrid"):
        sem_weight = st.slider("Semantic weight", 0.0, 1.0, 0.7, 0.05)
        tfidf_weight = 1.0 - sem_weight
        st.caption(f"Semantic {sem_weight:.2f} | TF-IDF {tfidf_weight:.2f}")
    else:
        sem_weight, tfidf_weight = 0.7, 0.3

    top_k = st.slider("Top K chunks", min_value=1, max_value=8, value=5, step=1)
    length_preset = st.selectbox("Summary length", ["Short", "Medium", "Long"], index=1)
    if length_preset == "Short":
        min_len, max_len = 60, 140
    elif length_preset == "Medium":
        min_len, max_len = 100, 220
    else:
        min_len, max_len = 150, 300
    st.caption("Soft targets; actual length may vary slightly.")

# ---- 3️⃣ Load documents ----
docs = []

if mode == "User-uploaded files":
    uploaded_files = st.file_uploader(
        "Upload one or more PDF/TXT files:",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded!")
        with st.spinner("📦 Processing uploaded files..."):
            tmp_dir = tempfile.mkdtemp()
            for f in uploaded_files:
                path = os.path.join(tmp_dir, f.name)
                with open(path, "wb") as out:
                    out.write(f.read())
            docs = load_corpus(tmp_dir)
    else:
        st.info("Waiting for uploaded files...")
        st.stop()
else:
    st.info("Using development backend files")
    # backend_files = [
    #     "data/doc1.txt",
    #     "data/doc2.txt",
    #     "data/doc3.txt",
    #     "data/doc4.txt",
    #     "data/doc5.txt",
    #     "data/doc6.txt",
    #     "data/doc7.txt",
    #     "data/doc8.txt",
    #     "data/doc9.txt",
    #     "data/doc10.txt"
    # ]
    with st.spinner("📦 Loading backend files..."):
        time.sleep(1)  # simulate loading
        docs = load_corpus("data/corpus")

if not docs:
    st.error("No documents found. Upload files or ensure backend files exist.")
    st.stop()

# ---- 4️⃣ Initialize retriever ----
doc_fingerprint = (len(docs), tuple(sorted([d["id"] for d in docs])[:20]), retrieval_mode, sem_weight, tfidf_weight)
if "retriever_fp" not in st.session_state or st.session_state.retriever_fp != doc_fingerprint:
    with st.spinner("⚙️ Setting up retrieval engine..."):
        time.sleep(1)  # simulate processing time
        if retrieval_mode.startswith("Semantic"):
            st.session_state.retriever = SemanticRetriever(docs)
        elif retrieval_mode.startswith("Keyword"):
            st.session_state.retriever = TfIdfRetriever(docs)
        else:
            st.session_state.retriever = HybridRetriever(docs, weight_semantic=sem_weight, weight_tfidf=tfidf_weight)
        st.session_state.retriever_fp = doc_fingerprint
st.success("✅ Retrieval engine ready!")

# ---- 5️⃣ Chat interface ----
st.subheader("💬 Ask a question about your documents")
query = st.text_input("Type your query here:")

if query:
    with st.spinner("🔄 Processing your request..."):
        time.sleep(0.5)  # simulate backend query processing
        results = st.session_state.retriever.retrieve(query, top_k=top_k)
    
    if not results:
        st.warning("No relevant chunks found for this query.")
    else:
        st.subheader("📘 Meta-Summary")
        with st.spinner("📝 Summarizing results..."):
            summary = meta_summarize(results, min_len=min_len, max_len=max_len)
            time.sleep(0.5)  # simulate processing
        st.write(summary)

        with st.expander("Show retrieved chunks"):
            for r in results:
                st.markdown(f"**{r['id']}** (score: {r['score']:.4f})")
                st.write(r["text"][:500] + ("..." if len(r["text"]) > 500 else ""))
                st.markdown("---")
