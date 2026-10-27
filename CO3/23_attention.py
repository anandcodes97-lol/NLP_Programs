# Q23: Basic scaled dot-product attention
# Install: pip install numpy
import numpy as np

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

def attention(query, key, value):
    scores = query @ key.T / np.sqrt(query.shape[-1])
    weights = softmax(scores)
    return weights @ value, weights

np.random.seed(7)
Q, K, V = np.random.rand(3, 4), np.random.rand(3, 4), np.random.rand(3, 4)
output, weights = attention(Q, K, V)
print("Attention weights:\n", np.round(weights, 3))
print("Attention output:\n", np.round(output, 3))
