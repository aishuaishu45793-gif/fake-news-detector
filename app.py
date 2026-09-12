import pickle
from utils import clean_text

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def predict_news(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    result = model.predict(vec)
    
    if result[0] == 1:
        return "Real News ✅"
    else:
        return "Fake News ❌"

# Test
if __name__ == "__main__":
    news = input("Enter news: ")
    print(predict_news(news))
