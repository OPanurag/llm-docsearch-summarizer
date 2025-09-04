import os
import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9.,!?;:()'\"-]", " ", text)
    stops = set(stopwords.words("english"))
    text = " ".join([w for w in text.split() if w not in stops])
    return text

def load_corpus(data_dir="data/corpus"):
    corpus = []
    for fname in os.listdir(data_dir):
        if fname.endswith(".txt"):
            with open(os.path.join(data_dir, fname), "r", encoding="utf-8") as f:
                corpus.append({"id": fname, "text": clean_text(f.read())})
    return corpus
