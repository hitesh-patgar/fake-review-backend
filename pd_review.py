# predict_review.py

import joblib
import string
import nltk
from nltk.corpus import stopwords

# -------- Download stopwords (only first time) --------
# nltk.download('stopwords')

# -------- Load the Saved Model and Vectorizer --------
model = joblib.load("fake_review_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# -------- Text Cleaning Function --------
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

print("Type your review below to test. Type 'exit' to quit.\n")

while True:
    user_input = input("📝 Enter a review: ")
    if user_input.lower() == "exit":
        print("👋 Exiting the review detector. Goodbye!")
        break

    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    print(f"🔍 Prediction: {prediction}\n")
