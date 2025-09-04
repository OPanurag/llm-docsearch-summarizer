from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

class EmbeddingIndex:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.docs = []

    def build_index(self, docs):
        self.docs = docs
        embeddings = self.model.encode([d["text"] for d in docs], convert_to_numpy=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def search(self, query, top_k=5):
        q_emb = self.model.encode([query], convert_to_numpy=True)
        scores, idx = self.index.search(q_emb, top_k)
        results = [self.docs[i] for i in idx[0]]
        return results
