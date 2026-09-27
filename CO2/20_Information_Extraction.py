import spacy

nlp = spacy.load("en_core_web_sm")

text = "Elon Musk founded SpaceX in 2002. SpaceX is located in California."

doc = nlp(text)

print("Named Entities:")

for entity in doc.ents:
    print(entity.text, "->", entity.label_)

print("\nNoun Phrases:")

for chunk in doc.noun_chunks:
    print(chunk.text)
