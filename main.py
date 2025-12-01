import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load trained model
model = load_model("model_optimal.h5") 

# Emotion Labels - MUST match your training order
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale (model trained on grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect face
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Extract ROI
        roi_gray = gray[y:y+h, x:x+w]

        # Preprocess
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype("float") / 255.0
        roi = np.expand_dims(roi, axis=-1) # (48,48) → (48,48,1)
        roi = np.expand_dims(roi, axis=0)  # (1,48,48,1)

        # Predict emotion
        preds = model.predict(roi, verbose=0)[0]
        label = EMOTIONS[np.argmax(preds)]

        # Draw rectangle & label
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255,0,0), 2)

    # Show frame
    cv2.imshow("Real-Time Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release everything
cap.release()
cv2.destroyAllWindows()
