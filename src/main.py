import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load face cascade & model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
model = load_model("model_optimal.h5") 

# Emotion categories (MUST match training!)
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Colors for each bar
colors = [
    (255, 0, 0),     # Angry - Blue
    (0, 255, 0),     # Disgust - Green
    (0, 0, 255),     # Fear - Red
    (0, 255, 255),   # Happy - Yellow
    (255, 0, 255),   # Sad - Purple
    (255, 255, 0),   # Surprise - Cyan
    (200, 200, 200)  # Neutral - Gray
]

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Mirror effect

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Face ROI (48x48 grayscale)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=-1)
        roi = np.expand_dims(roi, axis=0)

        preds = model.predict(roi, verbose=0)[0]
        emotion_index = np.argmax(preds)
        emotion_label = EMOTIONS[emotion_index]
        confidence = preds[emotion_index] * 100

        # Draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 140, 255), 2)

        # Text above face
        text = f"{emotion_label} ({confidence:.1f}%)"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 
                    0.7, (0, 140, 255), 2)

        # Draw emotion probability bars
        bar_x = x + w + 10
        bar_y = y

        for i, (emotion, prob) in enumerate(zip(EMOTIONS, preds)):
            bar_width = int(prob * 150)
            cv2.rectangle(frame, 
                          (bar_x, bar_y + i*25),
                          (bar_x + bar_width, bar_y + i*25 + 20),
                          colors[i],
                          -1)
            cv2.putText(frame, f"{emotion} {prob*100:.1f}%", 
                        (bar_x + 160, bar_y + i*25 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

    cv2.imshow("VibeSense - Live Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
