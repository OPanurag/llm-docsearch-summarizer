from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize(text, min_len=30, max_len=120):
    return summarizer(text, min_length=min_len, max_length=max_len, do_sample=False)[0]["summary_text"]
