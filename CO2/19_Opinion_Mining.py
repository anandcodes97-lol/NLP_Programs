from textblob import TextBlob

reviews = [
    "The product is excellent and very useful",
    "The quality is poor and disappointing",
    "I really like this product",
    "The product is bad and expensive",
    "Amazing quality and great service"
]

for review in reviews:
    polarity = TextBlob(review).sentiment.polarity

    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    print("Review:", review)
    print("Polarity:", polarity)
    print("Sentiment:", sentiment)
    print()
