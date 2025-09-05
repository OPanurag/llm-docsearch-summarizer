# src/summarizer.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def _pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = _pick_device()
MODEL_NAME = "facebook/bart-large-cnn" if DEVICE.type in ("cuda", "mps") else "sshleifer/distilbart-cnn-12-6"

print(f"🔧 Summarizer loading: {MODEL_NAME} on {DEVICE}")
_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
_MODEL = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
_MODEL.eval()

def summarize(text: str, min_len: int = 30, max_len: int = 120) -> str:
    if not text or not text.strip():
        return ""
    # truncate long text
    text = text[:1024]
    inputs = _TOKENIZER([text], truncation=True, max_length=1024, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        output_ids = _MODEL.generate(
            **inputs,
            num_beams=4,
            length_penalty=1.0,
            min_length=min_len,
            max_length=max_len,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
    return _TOKENIZER.decode(output_ids[0], skip_special_tokens=True)

def meta_summarize(chunks, min_len: int = 80, max_len: int = 200) -> str:
    if not chunks:
        return ""
    partials = []
    for c in chunks:
        partials.append(summarize(c["text"], min_len=30, max_len=100))
    combined = " ".join(partials)
    return summarize(combined, min_len=min_len, max_len=max_len)
