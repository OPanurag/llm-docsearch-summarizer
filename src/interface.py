import streamlit as st
from src.preprocessing import load_corpus
from src.retrieval import HybridRetriever
from src.summarizer import summarize
from src.config import EMBEDDING_MODEL, TOP_K

st.title("📚 LLM Document Search & Summarizer")

query = st.text_input("Enter your query:")
if "retriever" not in st.session_state:
    docs = load_corpus()
    st.session_state.retriever = HybridRetriever(docs, EMBEDDING_MODEL)

if query:
    results = st.session_state.retriever.retrieve(query, top_k=TOP_K)
    st.subheader("Top Results")
    for r in results:
        st.write(f"**{r['id']}**: {r['text'][:500]}...")
        with st.expander("Summary"):
            st.write(summarize(r["text"]))
