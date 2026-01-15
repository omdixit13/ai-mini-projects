# Fast AI Mini Project: Sentiment Analysis

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

texts = [
    "I love this product",
    "This is an amazing experience",
    "I hate this service",
    "Very bad quality"
]

labels = [1, 1, 0, 0]  # 1 = Positive, 0 = Negative

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

test_text = ["this product is amazing"]
test_vector = vectorizer.transform(test_text)

prediction = model.predict(test_vector)
print("Prediction (1=Positive, 0=Negative):", prediction[0])
