# Fast AI Mini Project: Text Similarity

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sentences = [
    "AI is transforming the world",
    "Artificial intelligence is changing everything",
    "I love playing football"
]

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(sentences)

similarity_matrix = cosine_similarity(vectors)

print("Cosine Similarity Matrix:\n", similarity_matrix)
