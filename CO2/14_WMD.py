import gensim.downloader as api

model = api.load("word2vec-google-news-300")

text1 = "machine learning is powerful"
text2 = "artificial intelligence is useful"

distance = model.wmdistance(
    text1.lower().split(),
    text2.lower().split()
)

print("Word Mover's Distance:")
print(distance)
