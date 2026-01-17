# Fast AI Mini Project: Resume Matcher (JD vs Resume)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

resume_text = """
Python developer with experience in machine learning,
data analysis, and building AI applications.
"""

job_description = """
Looking for a Python engineer with knowledge of machine learning,
data analytics, and AI-based solutions.
"""

vectorizer = TfidfVectorizer(stop_words="english")

vectors = vectorizer.fit_transform([resume_text, job_description])

similarity_score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

print("Resume–JD Match Score:", round(similarity_score * 100, 2), "%")
