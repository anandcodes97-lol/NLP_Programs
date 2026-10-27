# Q32: Hybrid probabilistic + FSM-based document boundary detection
# Install: pip install scikit-learn
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

examples = [
    "Document Title: Annual Report", "Chapter 1: Introduction",
    "Invoice Number: 12345", "----- New Document -----",
    "This is a sentence in the middle of a document.",
    "The company reported growth during the year.",
    "More details are provided in the next paragraph.",
    "The results are shown in the table below."
]
labels = [1, 1, 1, 1, 0, 0, 0, 0]  # 1 = likely document start
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
classifier = LogisticRegression(random_state=0).fit(vectorizer.fit_transform(examples), labels)

def is_boundary(line):
    line = line.strip()
    if not line:
        return False
    # FSM/rule-based boundary cues
    if re.match(r"^(document title:|chapter\s+\d+|invoice number:|[-=]{3,})", line, re.I):
        return True
    probability = classifier.predict_proba(vectorizer.transform([line]))[0][1]
    return probability >= 0.55

def split_documents(text):
    docs, current, state = [], [], "OUTSIDE"
    for line in text.splitlines():
        if is_boundary(line):
            if current:
                docs.append("\\n".join(current).strip())
            current, state = [line], "INSIDE"
        elif line.strip():
            if state == "OUTSIDE":
                state = "INSIDE"
            current.append(line)
    if current:
        docs.append("\\n".join(current).strip())
    return docs

sample = """Document Title: Annual Report
The company reported growth during the year.
More details are provided in the next paragraph.

Chapter 1: Introduction
This document explains the project.

Invoice Number: 12345
Payment is due within thirty days."""
for i, doc in enumerate(split_documents(sample), 1):
    print(f"\\n--- Document {i} ---\\n{doc}")
