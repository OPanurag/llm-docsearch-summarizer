from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TfIdfRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([d["text"] for d in docs])

    def retrieve(self, query, top_k=3):
        vec = self.vectorizer.transform([query])
        scores = cosine_similarity(vec, self.matrix).flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        return [self.docs[i] for i in top_indices]
