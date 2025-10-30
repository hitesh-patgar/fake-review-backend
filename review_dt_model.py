# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import string
import nltk
from nltk.corpus import stopwords
import joblib

# -------- Download stopwords (only first time) --------
nltk.download('stopwords')

# -------- Load Dataset --------
df = pd.read_csv("reviews_400.csv")  # Replace with your dataset

# -------- Data Preprocessing --------
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

df['clean_review'] = df['review'].apply(clean_text)

# -------- Feature Extraction --------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_review'])
y = df['label']

# -------- Train the Model --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = MultinomialNB()
model.fit(X_train, y_train)

# -------- Save Model and Vectorizer --------
joblib.dump(model, "fake_review_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("✅ Model trained and saved successfully!")
print("Files saved:")
print(" - fake_review_model.pkl")
print(" - tfidf_vectorizer.pkl")
