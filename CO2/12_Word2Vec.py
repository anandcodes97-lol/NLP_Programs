from gensim.models import Word2Vec

sentences = [
    ["machine", "learning", "is", "interesting"],
    ["artificial", "intelligence", "is", "powerful"],
    ["machine", "learning", "uses", "data"],
    ["deep", "learning", "is", "a", "part", "of", "AI"]
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4
)

word = "learning"

print("Vector for", word)
print(model.wv[word])

print("\nSimilar words:")
print(model.wv.most_similar(word))
