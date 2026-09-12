import cv2
import pickle

# Load model and vectorizer
try:
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
except Exception as e:
    print("Error loading model/vectorizer:", e)
    exit()

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Start camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

result_text = ""

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw face box and name
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, "Aishwarya", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # Show result text
    cv2.putText(frame, result_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    cv2.imshow("Fake News Detection System", frame)

    key = cv2.waitKey(1) & 0xFF

    # Press N to check news
    if key == ord('n'):
        print("\n👉 Enter News:")
        news = input()

        try:
            news_vector = vectorizer.transform([news])
            prediction = model.predict(news_vector)

            if prediction[0] == 0:
                result_text = "REAL NEWS"
            else:
                result_text = "FAKE NEWS"

        except Exception as e:
            print("ERROR:", e)   # 🔥 THIS IS IMPORTANT
            result_text = "MODEL ERROR"

    # Press Q to quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
