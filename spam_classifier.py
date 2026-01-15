# Fast AI Mini Project: Spam Classifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample SMS data
messages = [
    "Win a free iPhone now",
    "Limited offer click now",
    "Hey are we meeting today",
    "Let's go to the gym tomorrow"
]

# Labels: 1 = Spam, 0 = Not Spam
labels = [1, 1, 0, 0]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)

model = MultinomialNB()
model.fit(X, labels)

test_message = ["free offer just for you"]
test_vector = vectorizer.transform(test_message)

prediction = model.predict(test_vector)
print("Prediction (1=Spam, 0=Not Spam):", prediction[0])
