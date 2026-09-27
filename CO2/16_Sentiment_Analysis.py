from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

text = "I really love this product. It is amazing and wonderful."

blob = TextBlob(text)

print("TextBlob Polarity:", blob.sentiment.polarity)
print("TextBlob Subjectivity:", blob.sentiment.subjectivity)

analyzer = SentimentIntensityAnalyzer()
score = analyzer.polarity_scores(text)

print("\nVADER Scores:")
print(score)

if score["compound"] >= 0.05:
    print("Sentiment: Positive")
elif score["compound"] <= -0.05:
    print("Sentiment: Negative")
else:
    print("Sentiment: Neutral")
