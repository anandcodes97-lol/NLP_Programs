from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

texts = [
    "I love this product",
    "This product is amazing",
    "I hate this product",
    "This product is terrible",
    "Very good product",
    "Very bad product"
]

labels = [1, 1, 0, 0, 1, 0]

X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.3,
    random_state=42
)

vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))

new_text = ["I love this amazing product"]
new_vector = vectorizer.transform(new_text)

print("Prediction:", model.predict(new_vector))
