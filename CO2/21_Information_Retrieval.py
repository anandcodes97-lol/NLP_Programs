from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "Machine learning is a branch of artificial intelligence",
    "Deep learning uses neural networks",
    "Natural language processing deals with human language",
    "Machine learning algorithms learn from data",
    "Artificial intelligence is used in many applications"
]

query = "machine learning artificial intelligence"

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(documents)
query_vector = vectorizer.transform([query])

scores = cosine_similarity(query_vector, X)[0]

ranking = scores.argsort()[::-1]

print("Search Results:")

for index in ranking:
    print("\nDocument:", index + 1)
    print("Score:", scores[index])
    print(documents[index])
