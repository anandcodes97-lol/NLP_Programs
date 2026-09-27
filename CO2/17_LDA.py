from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

documents = [
    "machine learning algorithms use data",
    "deep learning uses neural networks",
    "artificial intelligence uses machine learning",
    "neural networks are used in deep learning",
    "data science uses machine learning"
]

vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(documents)

lda = LatentDirichletAllocation(
    n_components=2,
    random_state=42
)

lda.fit(X)

words = vectorizer.get_feature_names_out()

for topic_index, topic in enumerate(lda.components_):
    print("Topic", topic_index + 1)
    top_words = topic.argsort()[-5:][::-1]
    print([words[i] for i in top_words])
