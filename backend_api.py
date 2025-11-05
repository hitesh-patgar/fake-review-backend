from flask import Flask, request, jsonify
import joblib
import string
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)

# ✅ Ensure NLTK stopwords are available
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

# Load model and vectorizer
model = joblib.load("fake_review_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    review = data.get("review", "")
    cleaned = clean_text(review)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return jsonify({"label": prediction})

@app.route("/detect-fake-review", methods=["POST"])
def detect_fake_review():
    data = request.get_json()
    review = data.get("review", "")
    if not review:
        return jsonify({"error": "No review text provided"}), 400

    cleaned = clean_text(review)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    # ✅ Normalize output to match frontend expectations
    is_fake = True if prediction.lower() == "fake" else False

    return jsonify({
        "label": prediction,
        "is_fake": is_fake
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
