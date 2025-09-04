from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .embeddings import EmbeddingIndex

class HybridRetriever:
    def __init__(self, docs, embedding_model):
        self.docs = docs
        self.tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.tfidf.fit_transform([d["text"] for d in docs])
        self.embedding_index = EmbeddingIndex(embedding_model)
        self.embedding_index.build_index(docs)

    def retrieve(self, query, top_k=5):
        # TF-IDF
        tfidf_vec = self.tfidf.transform([query])
        tfidf_scores = cosine_similarity(tfidf_vec, self.tfidf_matrix).flatten()
        tfidf_top = sorted(range(len(tfidf_scores)), key=lambda i: tfidf_scores[i], reverse=True)[:top_k]

        # Embeddings
        embedding_results = self.embedding_index.search(query, top_k)

        # Merge unique results
        seen = set()
        results = []
        for i in tfidf_top:
            if self.docs[i]["id"] not in seen:
                results.append(self.docs[i])
                seen.add(self.docs[i]["id"])
        for r in embedding_results:
            if r["id"] not in seen:
                results.append(r)
                seen.add(r["id"])
        return results[:top_k]
