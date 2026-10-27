# Q25: Sentence embeddings from BERT and GPT-2 hidden states
# Install: pip install transformers torch
import torch
from transformers import AutoTokenizer, AutoModel

sentences = [
    "Natural language processing helps computers understand text.",
    "Deep learning is used in many language applications."
]

def mean_pool(hidden, mask):
    mask = mask.unsqueeze(-1).expand(hidden.size()).float()
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

for name in ["bert-base-uncased", "gpt2"]:
    print("\\nModel:", name)
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(name)
    batch = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        output = model(**batch)
    embeddings = mean_pool(output.last_hidden_state, batch["attention_mask"])
    print("Embedding shape:", tuple(embeddings.shape))
    print("First embedding (first 8 values):", embeddings[0, :8].tolist())
