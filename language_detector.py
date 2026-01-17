# Fast AI Mini Project: Language Detector

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample training data
texts = [
    "Hello how are you",
    "This is a good day",
    "Hola como estas",
    "Este es un buen dia",
    "Bonjour comment ca va",
    "Ceci est une bonne journee"
]

languages = [
    "English", "English",
    "Spanish", "Spanish",
    "French", "French"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, languages)

# Test sentence
test_text = ["how are you today"]
test_vector = vectorizer.transform(test_text)

prediction = model.predict(test_vector)
print("Detected Language:", prediction[0])
