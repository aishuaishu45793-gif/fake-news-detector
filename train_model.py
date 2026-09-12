import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Sample dataset (you can replace with real CSV later)
data = {
    "text": [
        "Government announces new policy",
        "Aliens landed in India",
        "Stock market is growing",
        "Earth will stop rotating tomorrow",
        "Scientists discover new vaccine",
        "Drinking petrol cures disease"
    ],
    "label": [0, 1, 0, 1, 0, 1]  # 0 = Real, 1 = Fake
}

df = pd.DataFrame(data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2
)

# Vectorizer
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# Model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Save files
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model and vectorizer saved!")
