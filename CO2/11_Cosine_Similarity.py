from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "Machine learning is amazing",
    "Machine learning is interesting",
    "I like artificial intelligence"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

similarity = cosine_similarity(X)

print("Cosine Similarity Matrix:")
print(similarity)
