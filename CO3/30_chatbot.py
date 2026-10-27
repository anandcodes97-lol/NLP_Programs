# Q30: Conversational chatbot using DialoGPT
# Install: pip install transformers torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)
history = None
print("Chatbot ready. Type 'quit' to stop.")
while True:
    text = input("You: ").strip()
    if text.lower() in {"quit", "exit"}:
        break
    if not text:
        continue
    ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt")
    bot_input = torch.cat([history, ids], dim=-1) if history is not None else ids
    history = model.generate(bot_input, max_new_tokens=80,
                             pad_token_id=tokenizer.eos_token_id,
                             do_sample=True, top_k=50, top_p=0.95)
    reply = tokenizer.decode(history[:, bot_input.shape[-1]:][0], skip_special_tokens=True)
    print("Bot:", reply or "I'm not sure how to respond.")
