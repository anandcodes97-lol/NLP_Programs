# Q31: Classify documents and separate them into category folders using TF-IDF + SVM
# Install: pip install scikit-learn
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

documents = [
    ("doc1.txt", "The team won the football match and scored a goal."),
    ("doc2.txt", "The player trained hard before the cricket tournament."),
    ("doc3.txt", "Python programming uses functions classes and variables."),
    ("doc4.txt", "The software developer fixed a bug in the application."),
    ("doc5.txt", "The cricket team won the final match."),
    ("doc6.txt", "Machine learning models are trained using data.")
]
labels = ["sports", "sports", "technology", "technology", "sports", "technology"]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform([text for _, text in documents])
model = LinearSVC().fit(X, labels)
root = Path("separated_documents")

for filename, content in documents:
    category = model.predict(vectorizer.transform([content]))[0]
    folder = root / category
    folder.mkdir(parents=True, exist_ok=True)
    (folder / filename).write_text(content, encoding="utf-8")
    print(f"{filename} -> {folder}")
print("Files created under:", root.resolve())
