from transformers import pipeline
from .config import LLM_MODEL, SUMMARY_MAX_LENGTH, SUMMARY_MIN_LENGTH

summarizer = pipeline("summarization", model=LLM_MODEL)

def summarize(text, min_len=SUMMARY_MIN_LENGTH, max_len=SUMMARY_MAX_LENGTH):
    return summarizer(text, min_length=min_len, max_length=max_len, do_sample=False)[0]["summary_text"]
