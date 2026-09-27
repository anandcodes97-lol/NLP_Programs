from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "I love machine learning",
    "Machine learning is interesting",
    "I love artificial intelligence"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

print("Vocabulary:")
print(vectorizer.get_feature_names_out())

print("\nTF-IDF Matrix:")
print(X.toarray())
