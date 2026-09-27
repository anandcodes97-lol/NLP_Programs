from nltk.util import ngrams
from nltk.tokenize import word_tokenize

text = "I love machine learning and artificial intelligence"

words = word_tokenize(text)

unigrams = list(ngrams(words, 1))
bigrams = list(ngrams(words, 2))
trigrams = list(ngrams(words, 3))

print("Unigrams:")
print(unigrams)

print("\nBigrams:")
print(bigrams)

print("\nTrigrams:")
print(trigrams)
