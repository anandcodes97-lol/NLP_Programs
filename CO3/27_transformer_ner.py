# Q27: Named Entity Recognition using a Transformer
# Install: pip install transformers torch
from transformers import pipeline

ner = pipeline("token-classification", model="dslim/bert-base-NER",
               aggregation_strategy="simple")
text = input("Enter text: ").strip() or "Sundar Pichai works at Google in California."
for e in ner(text):
    print(f"Text: {e['word']:<20} Label: {e['entity_group']:<8} Confidence: {e['score']:.3f}")
