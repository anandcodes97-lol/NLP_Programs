# NLP Unit 1 Programs

## Description

This repository contains six basic Natural Language Processing (NLP) programs using Python, NLTK, spaCy, and Regular Expressions. Each program demonstrates a common NLP operation in a simple way.

## Programs Included

1. Tokenization using NLTK and spaCy
2. Stemming and Lemmatization
3. Stop-word Removal
4. Part-of-Speech (POS) Tagging
5. Parsing and Chunking using RegEx and spaCy
6. Named Entity Recognition (NER) using spaCy

## Technologies Used

- Python 3
- NLTK
- spaCy
- Regular Expressions (RegEx)

## Installation

Install the required libraries:

```bash
pip install nltk spacy
```

Download the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

---

# 1. Tokenization using NLTK and spaCy

### Description

This program breaks a given text into smaller parts called tokens. It shows how to divide text into individual words and sentences using both NLTK and spaCy.

### Code

```python
import nltk
import spacy

nltk.download('punkt')

text = "I am learning Natural Language Processing."

print("NLTK Words:", nltk.word_tokenize(text))
print("NLTK Sentences:", nltk.sent_tokenize(text))

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

print("spaCy Words:", [word.text for word in doc])
print("spaCy Sentences:", [sent.text for sent in doc.sents])
```

### How it Works

The program takes a sentence as input and uses NLTK and spaCy to separate it into words and sentences. This is one of the first steps in most NLP applications because text needs to be divided into smaller units before further processing.

---

# 2. Stemming and Lemmatization

### Description

This program demonstrates two methods of reducing words to their basic forms: stemming and lemmatization.

### Code

```python
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('wordnet')

words = ["playing", "played", "studies", "studying"]

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

for word in words:
    print(word, "-> Stem:", stemmer.stem(word),
          "Lemma:", lemmatizer.lemmatize(word))
```

### How it Works

The program takes a list of words and processes each word in two ways. The stemmer finds a shortened root form, while the lemmatizer tries to find the meaningful dictionary form of the word. The results are displayed together so the difference can be observed.

---

# 3. Stop-word Removal

### Description

This program removes common English words, known as stop words, from a sentence.

### Code

```python
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

text = "This is a simple example of natural language processing."

stop_words = set(stopwords.words('english'))

words = text.split()
result = [word for word in words if word.lower() not in stop_words]

print("Original:", text)
print("After Stop-word Removal:", " ".join(result))
```

### How it Works

The sentence is first divided into words. The program then checks each word against NLTK's English stop-word list and keeps only the words that are not stop words. This helps reduce unnecessary words during text analysis.

---

# 4. Part-of-Speech (POS) Tagging

### Description

This program identifies the grammatical role of each word in a sentence, such as noun, verb, adjective, or determiner.

### Code

```python
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

text = "The boy is playing football."

words = nltk.word_tokenize(text)
pos = nltk.pos_tag(words)

print(pos)
```

### How it Works

The sentence is first divided into individual words. NLTK then assigns a grammatical tag to each word. For example, a noun can be identified as `NN`, a verb as `VB`, and an adjective as `JJ`.

---

# 5. Parsing and Chunking using RegEx and spaCy

### Description

This program demonstrates simple text extraction using Regular Expressions and identifies noun phrases using spaCy.

### Code

```python
import re
import spacy

text = "The quick brown fox jumps over the dog."

print("Words:", re.findall(r'\w+', text))

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

for chunk in doc.noun_chunks:
    print("Chunk:", chunk.text)
```

### How it Works

Regular Expressions are used to extract words from the sentence. The text is then processed using spaCy, which identifies groups of words that form noun phrases. This gives a simple introduction to parsing and chunking.

---

# 6. Named Entity Recognition (NER) using spaCy

### Description

This program identifies named entities in a sentence, such as people, places, organizations, and other important entities.

### Code

```python
import spacy

nlp = spacy.load("en_core_web_sm")

text = "Narendra Modi visited New Delhi."

doc = nlp(text)

for entity in doc.ents:
    print(entity.text, "->", entity.label_)
```

### How it Works

The sentence is processed using spaCy's English language model. spaCy detects important names or entities and assigns a label to each one. For example, a person's name can be labelled `PERSON` and a location can be labelled `GPE`.

---

# Repository Structure

```text
NLP-Unit-1/
│
├── README.md
├── tokenization.py
├── stemming.py
├── stopwords.py
├── pos_tagging.py
├── parsing_chunking.py
└── ner.py
```

# How to Run

Open the terminal in the project folder and run the required program:

```bash
python tokenization.py
```

Similarly, replace the filename with:

```bash
python stemming.py
python stopwords.py
python pos_tagging.py
python parsing_chunking.py
python ner.py
```

# Objective

The objective of these programs is to understand the basic concepts of Natural Language Processing and learn how text can be processed using Python libraries.

# Conclusion

These programs provide a simple introduction to important NLP techniques such as tokenization, stemming, lemmatization, stop-word removal, POS tagging, chunking, and named entity recognition.

# Author

**Anand Jaiswal**
