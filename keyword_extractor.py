# Fast AI Mini Project: Keyword Extractor

from sklearn.feature_extraction.text import TfidfVectorizer

text = [
    "Artificial intelligence and machine learning are transforming technology"
]

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(text)

feature_names = vectorizer.get_feature_names_out()
scores = tfidf_matrix.toarray()[0]

keywords = sorted(
    zip(feature_names, scores),
    key=lambda x: x[1],
    reverse=True
)

print("Top Keywords:")
for word, score in keywords[:5]:
    print(word, round(score, 3))
