# src/preprocessing.py
import os
import re
from PyPDF2 import PdfReader

# Lazy stopwords
_STOPWORDS = None

def _load_stopwords():
    global _STOPWORDS
    if _STOPWORDS is None:
        try:
            import nltk
            from nltk.corpus import stopwords
            try:
                _ = stopwords.words("english")
            except LookupError:
                nltk.download("stopwords", quiet=True)
            _STOPWORDS = set(stopwords.words("english"))
        except Exception:
            _STOPWORDS = set()
    return _STOPWORDS

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9.,!?;:()'\"-]", " ", text)
    stops = _load_stopwords()
    if stops:
        text = " ".join([w for w in text.split() if w not in stops])
    return text

def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        t = page.extract_text() or ""
        text += t + " "
    return text.strip()

def chunk_text(text, max_len=500):
    """Chunk text into max_len word segments"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_len):
        chunks.append(" ".join(words[i:i + max_len]))
    return chunks

def load_corpus(data_dir="data/corpus"):
    corpus = []
    if not os.path.isdir(data_dir):
        return corpus
    for fname in os.listdir(data_dir):
        path = os.path.join(data_dir, fname)
        raw_text = ""
        if fname.endswith(".txt"):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()
        elif fname.endswith(".pdf"):
            raw_text = read_pdf(path)
        else:
            continue
        raw_text = raw_text.strip()
        if not raw_text:
            continue
        # Chunk text
        for i, chunk in enumerate(chunk_text(clean_text(raw_text), max_len=500)):
            corpus.append({"id": f"{fname}_chunk{i}", "text": chunk})
    return corpus
