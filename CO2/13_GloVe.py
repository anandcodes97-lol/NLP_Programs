import gensim.downloader as api

model = api.load("glove-wiki-gigaword-50")

word = "computer"

print("GloVe Vector:")
print(model[word])

print("\nSimilar Words:")
print(model.most_similar(word))
