from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

documents = [
    "machine learning algorithms use data",
    "deep learning uses neural networks",
    "artificial intelligence uses machine learning",
    "neural networks are used in deep learning",
    "data science uses machine learning"
]

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(documents)

lsa = TruncatedSVD(n_components=2, random_state=42)
X_lsa = lsa.fit_transform(X)

words = vectorizer.get_feature_names_out()

for topic_index, topic in enumerate(lsa.components_):
    print("Topic", topic_index + 1)
    top_words = topic.argsort()[-5:][::-1]
    print([words[i] for i in top_words])

print("\nDocument-Topic Matrix:")
print(X_lsa)
