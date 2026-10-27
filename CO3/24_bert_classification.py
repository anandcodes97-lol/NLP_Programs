# Q24: Fine-tune a Transformer for text classification
# Install: pip install transformers datasets evaluate torch accelerate
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate

model_name = "distilbert-base-uncased"
texts = [
    "I love this product", "This is fantastic", "Very good quality",
    "I hate this product", "This is awful", "Very poor quality",
    "Really wonderful", "Completely disappointing"
]
labels = [1, 1, 1, 0, 0, 0, 1, 0]
ds = Dataset.from_dict({"text": texts, "label": labels}).train_test_split(test_size=0.25, seed=42)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)

ds = ds.map(tokenize, batched=True).remove_columns(["text"])
ds.set_format("torch")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
metric = evaluate.load("accuracy")

def compute_metrics(pred):
    logits, labels = pred
    return metric.compute(predictions=np.argmax(logits, axis=-1), references=labels)

args = TrainingArguments(
    output_dir="./bert_results", eval_strategy="epoch", save_strategy="no",
    num_train_epochs=1, per_device_train_batch_size=2,
    per_device_eval_batch_size=2, report_to="none"
)
trainer = Trainer(model=model, args=args, train_dataset=ds["train"],
                  eval_dataset=ds["test"], compute_metrics=compute_metrics)
trainer.train()
print(trainer.evaluate())
trainer.save_model("./bert_text_classifier")
tokenizer.save_pretrained("./bert_text_classifier")
