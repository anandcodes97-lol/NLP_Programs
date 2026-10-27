# Q28: English-to-French translation using MarianMT
# Install: pip install transformers torch sentencepiece
from transformers import MarianMTModel, MarianTokenizer

name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = MarianTokenizer.from_pretrained(name)
model = MarianMTModel.from_pretrained(name)
text = input("Enter English text: ").strip() or "How are you today?"
batch = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
translated = model.generate(**batch, max_new_tokens=128)
print("French translation:", tokenizer.decode(translated[0], skip_special_tokens=True))
