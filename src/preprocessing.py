import os
import re
import nltk
from nltk.corpus import stopwords
from PyPDF2 import PdfReader

# Ensure stopwords are available
try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9.,!?;:()'\"-]", " ", text)
    stops = set(stopwords.words("english"))
    text = " ".join([w for w in text.split() if w not in stops])
    return text

def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def load_corpus(data_dir="data/corpus"):
    corpus = []
    for fname in os.listdir(data_dir):
        path = os.path.join(data_dir, fname)
        if fname.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                raw_text = f.read()
        elif fname.endswith(".pdf"):
            raw_text = read_pdf(path)
        else:
            continue  # skip unsupported files

        if raw_text.strip():
            corpus.append({"id": fname, "text": clean_text(raw_text)})
    return corpus
