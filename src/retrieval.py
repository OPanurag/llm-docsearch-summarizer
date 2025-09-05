# src/retrieval.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticRetriever:
    def __init__(self, docs, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.docs = docs
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode([d["text"] for d in docs], convert_to_numpy=True)
        self.embeddings = self.embeddings.astype("float32")
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    def retrieve(self, query, top_k=3):
        query_vec = self.model.encode([query], convert_to_numpy=True).astype("float32")
        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                "id": self.docs[idx]["id"],
                "text": self.docs[idx]["text"],
                "score": float(scores[0][i])
            })
        return results

class TfIdfRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([d["text"] for d in docs])

    def retrieve(self, query, top_k=3):
        vec = self.vectorizer.transform([query])
        scores = cosine_similarity(vec, self.matrix).flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                "id": self.docs[idx]["id"],
                "text": self.docs[idx]["text"],
                "score": float(scores[idx])
            })
        return results

class HybridRetriever:
    def __init__(self, docs, weight_semantic=0.7, weight_tfidf=0.3,
                 model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.docs = docs
        self.semantic = SemanticRetriever(docs, model_name=model_name)
        self.tfidf = TfIdfRetriever(docs)
        self.w_semantic = weight_semantic
        self.w_tfidf = weight_tfidf

    def retrieve(self, query, top_k=3):
        # Semantic
        sem_emb = self.semantic.model.encode([query], convert_to_numpy=True).astype("float32")
        sem_scores, sem_indices = self.semantic.index.search(sem_emb, len(self.docs))
        sem_scores = -sem_scores[0]  # FAISS L2 -> similarity
        sem_ranking = {idx: sem_scores[i] for i, idx in enumerate(sem_indices[0])}

        # TF-IDF
        tfidf_vec = self.tfidf.vectorizer.transform([query])
        tfidf_scores = cosine_similarity(tfidf_vec, self.tfidf.matrix).flatten()
        tfidf_ranking = {i: tfidf_scores[i] for i in range(len(self.docs))}

        # Combine scores
        combined_scores = {}
        for i in range(len(self.docs)):
            s_score = sem_ranking.get(i, 0.0)
            t_score = tfidf_ranking.get(i, 0.0)
            combined_scores[i] = self.w_semantic * s_score + self.w_tfidf * t_score

        ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for idx, score in ranked:
            results.append({
                "id": self.docs[idx]["id"],
                "text": self.docs[idx]["text"],
                "score": float(score)
            })
        return results
