import pandas as pd
import numpy as np
import os

# Check current directory
print("Current Directory:", os.getcwd())

# File paths
fake_path = r"dataset/Fake.csv"
true_path = r"dataset/True.csv"

# Load datasets
fake = pd.read_csv(fake_path)
true = pd.read_csv(true_path)

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine datasets
data = pd.concat([fake, true], axis=0)

# Shuffle data
data = data.sample(frac=1).reset_index(drop=True)

# Keep required columns
data = data[["text", "label"]]

# Text preprocessing
import nltk
from nltk.corpus import stopwords
import re

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

data["text"] = data["text"].apply(clean_text)

# Train-test split
from sklearn.model_selection import train_test_split

X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model training
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Prediction & accuracy
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Prediction function
def predict_news(text):
    text = clean_text(text)
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)

    if prediction[0] == 1:
        return "Real News 🟢"
    else:
        return "Fake News 🔴"

# ✅ Your test lines
print(predict_news("India won the cricket world cup"))
print(predict_news("Aliens landed in Bangalore yesterday"))
