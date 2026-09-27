from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "I love machine learning",
    "Machine learning is interesting",
    "I love artificial intelligence"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)

print("Vocabulary:")
print(vectorizer.get_feature_names_out())

print("\nBoW Matrix:")
print(X.toarray())
