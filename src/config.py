import os

# Paths
DATA_DIR = os.path.join("data", "corpus")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "facebook/bart-large-cnn"

# Retrieval
TOP_K = 5

# Summarization
SUMMARY_MAX_LENGTH = 150
SUMMARY_MIN_LENGTH = 50
