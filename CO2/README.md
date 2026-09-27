# NLP Lab Programs – CO2

This repository contains the CO2 Natural Language Processing laboratory programs listed in the course instructions.

## CO2 Programs

| No. | Program | File |
|---|---|---|
| 8 | Bag-of-Words (BoW) Vectorization and Representation | `08_BOW.py` |
| 9 | TF-IDF Implementation and Comparison with BoW | `09_TFIDF.py` |
| 10 | N-Gram Model (Uni-, Bi-, Tri-gram) Generation from Corpus | `10_NGram.py` |
| 11 | Cosine Similarity Computation between Text Documents | `11_Cosine_Similarity.py` |
| 12 | Word2Vec Word Embeddings using Gensim on a Custom Corpus | `12_Word2Vec.py` |
| 13 | GloVe Embeddings Loading and Vector Representation | `13_GloVe.py` |
| 14 | Text Similarity using Word Mover's Distance (WMD) | `14_WMD.py` |
| 15 | Text Classification using Naive Bayes with TF-IDF | `15_Text_Classification.py` |
| 16 | Sentiment Analysis using TextBlob and VADER | `16_Sentiment_Analysis.py` |
| 17 | Topic Modeling using Latent Dirichlet Allocation (LDA) | `17_LDA.py` |
| 18 | Topic Modeling using Latent Semantic Analysis (LSA) | `18_LSA.py` |
| 19 | Opinion Mining on Product/Service Reviews Dataset | `19_Opinion_Mining.py` |
| 20 | Information Extraction (IE) from Structured/Unstructured Documents | `20_Information_Extraction.py` |
| 21 | Information Retrieval System with Ranking using TF-IDF | `21_Information_Retrieval.py` |

## Requirements

Python 3.x is recommended.

Install the required libraries:

```bash
pip install scikit-learn nltk gensim textblob vaderSentiment spacy
```

For Information Extraction, install the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

For the N-Gram program, download the required NLTK tokenizer data:

```python
import nltk
nltk.download('punkt')
```

## How to Run

Open Command Prompt or Terminal in the CO2 folder.

Run any program using:

```bash
python 08_BOW.py
```

For example:

```bash
python 09_TFIDF.py
python 10_NGram.py
python 11_Cosine_Similarity.py
```

You can replace the filename with any of the other program files.

## Folder Structure

```text
NLP-Lab-Programs/
│
├── CO1/
│   └── CO1 Programs
│
├── CO2/
│   ├── 08_BOW.py
│   ├── 09_TFIDF.py
│   ├── 10_NGram.py
│   ├── 11_Cosine_Similarity.py
│   ├── 12_Word2Vec.py
│   ├── 13_GloVe.py
│   ├── 14_WMD.py
│   ├── 15_Text_Classification.py
│   ├── 16_Sentiment_Analysis.py
│   ├── 17_LDA.py
│   ├── 18_LSA.py
│   ├── 19_Opinion_Mining.py
│   ├── 20_Information_Extraction.py
│   └── 21_Information_Retrieval.py
│
└── README.md
```

## Notes

Some Gensim programs download pretrained models when they are run for the first time. The download may take some time and requires an internet connection.

The CO2 programs should be uploaded to the same GitHub repository used for CO1 and organized in a separate `CO2` folder.
