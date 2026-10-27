# Q22: Sequence classification using an LSTM
# Install: pip install tensorflow numpy
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

texts = [
    "this movie is wonderful", "i really enjoyed this film", "excellent and lovely",
    "this was a great experience", "the story was amazing", "i hated this movie",
    "this film is terrible", "very boring and disappointing", "awful experience",
    "the story was bad"
]
labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
vectorizer = layers.TextVectorization(max_tokens=1000, output_sequence_length=12)
vectorizer.adapt(texts)
x = vectorizer(np.array(texts))

model = models.Sequential([
    layers.Embedding(1000, 16, mask_zero=True),
    layers.LSTM(16),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(x, labels, epochs=12, verbose=1)

sample = tf.constant(["the movie was excellent"])
p = float(model.predict(vectorizer(sample), verbose=0)[0][0])
print("Prediction:", "positive" if p >= 0.5 else "negative")
print("Positive probability:", round(p, 3))
