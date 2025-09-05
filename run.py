import argparse
from src.preprocessing import load_corpus
from src.retrieval import TfIdfRetriever
from src.summarizer import summarize

def cli_mode():
    corpus = load_corpus()
    retriever = TfIdfRetriever(corpus)

    query = input("Enter your query: ")
    results = retriever.retrieve(query, top_k=2)

    for r in results:
        print(f"\nDocument: {r['id']}")
        print(f"Content (first 200 chars): {r['text'][:200]}...")

        print("\nSummary:")
        print(summarize(r["text"]))
        print("=" * 60)

def ui_mode():
    import streamlit as st

    st.title("📚 LLM Document Search & Summarizer")

    query = st.text_input("Enter your query:")

    if "retriever" not in st.session_state:
        docs = load_corpus()
        st.session_state.retriever = TfIdfRetriever(docs)

    if query:
        results = st.session_state.retriever.retrieve(query, top_k=3)
        st.subheader("Top Results")

        for r in results:
            st.markdown(f"**{r['id']}**")
            st.write(r["text"][:300] + "...")
            with st.expander("Summary"):
                st.write(summarize(r["text"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run document search & summarizer")
    parser.add_argument("--mode", choices=["cli", "ui"], default="cli",
                        help="Choose 'cli' for terminal mode or 'ui' for Streamlit app")
    args = parser.parse_args()

    if args.mode == "cli":
        cli_mode()
    else:
        ui_mode()
