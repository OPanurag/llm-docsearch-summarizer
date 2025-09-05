# llm-docsearch-summarizer

An end-to-end system for intelligent document search and summarization using LLMs. 
It combines:
- Semantic retrieval with SentenceTransformers + FAISS
- Keyword retrieval with TF-IDF + cosine similarity
- Hybrid scoring to blend both
- Abstractive summarization using BART
- A Streamlit interface and evaluation utilities

---

## 1) Features at a Glance

- **Document ingestion**: Load TXT/PDF, clean, and chunk.
- **Retrieval**:
  - **Semantic**: SentenceTransformer embeddings + FAISS L2 index.
  - **Keyword**: TF-IDF + cosine similarity.
  - **Hybrid**: Weighted combination.
- **Summarization**:
  - Single chunk summarization.
  - Meta-summarization over multiple retrieved chunks.
- **Interface**: Streamlit app supporting uploads or local corpus.
- **Evaluation**: ROUGE-based scoring utilities.
- **Configurable**: Models, top-k, lengths via `src/config.py` and UI controls.

---

## 2) Getting the Project Locally

### 2.1. Prerequisites

- Python 3.11 recommended
- macOS/Linux/Windows supported; Apple Silicon (MPS) or CUDA can accelerate summarization
- Git for cloning the repository

### 2.2. Clone the Repository

```bash
# Using HTTPS
git clone https://github.com/<your-org-or-user>/llm-docsearch-summarizer.git
cd llm-docsearch-summarizer

# Or using SSH
# git clone git@github.com:<your-org-or-user>/llm-docsearch-summarizer.git
# cd llm-docsearch-summarizer
```

> Replace `<your-org-or-user>` with the appropriate GitHub owner.

### 2.3. Create and Activate a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# .\\venv\\Scripts\\Activate.ps1   # Windows PowerShell
```

### 2.4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If FAISS wheels fail on macOS, use CPU-only FAISS:
```bash
pip install faiss-cpu
```

If NLTK stopwords error occurs at runtime:
```bash
python -c "import nltk; nltk.download('stopwords')"
```

---

## 3) Running the App Locally

### 3.1. Start the Streamlit App (Full UI)

```bash
streamlit run app.py
```

- Open the local URL shown by Streamlit.
- Choose between:
  - “User-uploaded files” to upload PDFs/TXTs.
  - “Development backend files” to use `data/corpus/`.
- Configure in the sidebar:
  - Retrieval mode: Semantic, Keyword (TF-IDF), or Hybrid (with adjustable weights).
  - Top K chunks.
  - Summary length preset (Short/Medium/Long).
- Enter a query to see the meta-summary and retrieved chunks.

### 3.2. Minimal Demo UI

```bash
streamlit run src/interface.py
```

This simpler UI loads `data/corpus/` on startup, uses `HybridRetriever`, and summarizes each result.

---

## 4) Programmatic Usage (Python)

### 4.1. Load and Inspect Corpus

```python
# demo_load_corpus.py
# Purpose: Load corpus and inspect basic stats.

from src.preprocessing import load_corpus

# Load documents from the default directory (data/corpus)
docs = load_corpus("data/corpus")

# Print how many chunks were created
print(f"Loaded {len(docs)} chunks")

# Show first chunk preview
if docs:
    print(docs[0]["id"], "->", docs[0]["text"][:200])
```

### 4.2. Retrieval: Semantic, TF-IDF, and Hybrid

```python
# demo_retrieval.py
# Purpose: Run different retrieval modes on a query.

from src.preprocessing import load_corpus
from src.retrieval import SemanticRetriever, TfIdfRetriever, HybridRetriever

# Load corpus
docs = load_corpus("data/corpus")

# Define a query
query = "What are the key points about renewable energy?"

# Semantic retrieval
semantic = SemanticRetriever(docs)
sem_results = semantic.retrieve(query, top_k=5)

# TF-IDF retrieval
tfidf = TfIdfRetriever(docs)
tfidf_results = tfidf.retrieve(query, top_k=5)

# Hybrid retrieval with weights (semantic 0.7, tf-idf 0.3)
hybrid = HybridRetriever(docs, weight_semantic=0.7, weight_tfidf=0.3)
hybrid_results = hybrid.retrieve(query, top_k=5)

# Print top result from each method
def show_top(label, results):
    print(f"\n[{label}] Top result:")
    if results:
        print(results[0]["id"], "score:", results[0]["score"])
        print(results[0]["text"][:200])
    else:
        print("No results")

show_top("Semantic", sem_results)
show_top("TF-IDF", tfidf_results)
show_top("Hybrid", hybrid_results)
```

### 4.3. Summarization and Meta-Summarization

```python
# demo_summarization.py
# Purpose: Summarize a single chunk and then meta-summarize top retrieved chunks.

from src.preprocessing import load_corpus
from src.retrieval import HybridRetriever
from src.summarizer import summarize, meta_summarize

# Load corpus
docs = load_corpus("data/corpus")

# Set query
query = "Explain the main risks discussed in the documents."

# Retrieve top chunks
retriever = HybridRetriever(docs, weight_semantic=0.7, weight_tfidf=0.3)
results = retriever.retrieve(query, top_k=5)

# Summarize first chunk
if results:
    single_summary = summarize(results[0]["text"], min_len=40, max_len=120)
    print("Single chunk summary:\n", single_summary)

# Meta-summarize across all retrieved chunks
meta = meta_summarize(results, min_len=80, max_len=200)
print("\nMeta summary:\n", meta)
```

### 4.4. Evaluation (ROUGE)

```python
# demo_evaluation.py
# Purpose: Compute ROUGE for a generated summary vs. reference.

from src.evaluation import evaluate_summary

# Reference and generated texts for evaluation
reference = "Solar energy can be used to power homes and reduce emissions."
generated = "Solar power helps run households and cuts greenhouse gases."

# Compute ROUGE scores (1, 2, L)
scores = evaluate_summary(reference, generated)
print(scores)  # Dictionary with precision/recall/f-measure per ROUGE metric
```

---

## 5) Configuration

Configure defaults in `src/config.py`:
- **DATA_DIR**: Default corpus folder, e.g., `data/corpus`.
- **EMBEDDING_MODEL**: SentenceTransformers model, e.g., `sentence-transformers/all-MiniLM-L6-v2`.
- **LLM_MODEL**: Summarizer model, e.g., `facebook/bart-large-cnn`.
- **TOP_K**: Default number of chunks returned.
- **SUMMARY_MAX_LENGTH / SUMMARY_MIN_LENGTH**: Default summarization lengths.

Note: `run.py` exposes additional UI-based overrides (weights, top_k, length presets).

---

## 6) Data Preparation

- Place `.txt` and `.pdf` files under `data/corpus/`. Example repository contains `doc1.txt` … `doc10.txt`.
- The loader:
  - Reads raw text (TXT directly, PDFs via PyPDF2).
  - Cleans and tokenizes text.
  - Chunks text into ~500-word segments.
- Each chunk gets a unique ID like `filename_chunkX`.

---

## 7) Hardware Acceleration

- The summarizer auto-selects device:
  - **MPS** (Apple Silicon) if available.
  - **CUDA** if available.
  - **CPU** otherwise.
- On CPU, a smaller model is used: `sshleifer/distilbart-cnn-12-6` for speed.
- Tip: Ensure PyTorch is installed with MPS/CUDA support to maximize performance.

---

## 8) Running Tests

The project currently doess not have extensive testing but includes unit tests for core functionality placeholders.

Tests live under `tests/`:
- `tests/test_preprocessing.py`
- `tests/test_retrieval.py`
- `tests/test_summarizer.py`

Run with:

```bash
pytest -q
```

---

## 9) Troubleshooting

- **FAISS install issues on macOS**:
  - Use `pip install faiss-cpu`.
- **NLTK `stopwords` missing**:
  - Run `python -c "import nltk; nltk.download('stopwords')"` or allow lazy download at runtime.
- **Slow summarization**:
  - Reduce `max_len` and `min_len`.
  - Use CPU-friendly model (handled automatically).
  - Lower `top_k`.
- **No results returned**:
  - Verify documents exist and contain text.
  - Check cleaning didn’t strip meaningful content excessively.
- **Model download errors**:
  - Ensure internet access and correct Hugging Face model IDs.

---

## 10) Security and Privacy

- Uploaded files in the UI are written to a temp directory in the current session.
- Models run locally via Hugging Face; no external API calls by default.
- For sensitive docs, secure your environment and clean temp dirs as needed.

---

## 11) Extending the System

- Swap embedding model in `config.py` or pass `model_name` to retrievers.
- Adjust chunk size in `chunk_text(max_len=...)`.
- Add BM25 via `rank_bm25` and extend `HybridRetriever`.
- Replace summarizer model in `src/summarizer.py` (align tokenizer/generation args).

---

## 12) Quick Start Checklist

1. Clone the repo and `cd` into it.
2. Create a virtual environment and activate it.
3. `pip install -r requirements.txt` (use `faiss-cpu` if needed).
4. Add TXT/PDF files to `data/corpus/` or prepare files to upload in the app.
5. `streamlit run app.py`.
6. Select mode, configure retrieval and summary settings.
7. Ask a question and view the meta-summary.
