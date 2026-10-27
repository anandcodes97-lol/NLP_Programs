# Q29: Text summarization using BART
# Install: pip install transformers torch
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
text = input("Paste a paragraph to summarize: ").strip()
if not text:
    text = ("Natural language processing helps computers process human language. "
            "It is used in translation, search engines, chatbots, sentiment analysis, "
            "and summarization. Modern systems often use transformer models.")
result = summarizer(text, max_length=80, min_length=20, do_sample=False, truncation=True)
print("Summary:", result[0]["summary_text"])
