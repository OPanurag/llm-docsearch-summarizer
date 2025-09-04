from src.preprocessing import load_corpus
from src.retrieval import HybridRetriever
from src.summarizer import summarize
from src.config import EMBEDDING_MODEL, TOP_K

if __name__ == "__main__":
    corpus = load_corpus()
    retriever = HybridRetriever(corpus, EMBEDDING_MODEL)

    query = input("Enter your query: ")
    results = retriever.retrieve(query, top_k=TOP_K)
    for r in results:
        print(f"\nDoc: {r['id']}\n")
        print(f"Summary: {summarize(r['text'])}\n")
