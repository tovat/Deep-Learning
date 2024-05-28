from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np
from collections import Counter

# Load face and eye classifiers
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

# Load emotion classifier model
classifier = load_model('model_1.h5', compile=False)
classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def get_eye_color(h, s, v):
    if v < 50:
        return "Black"  # Very dark brown eyes
    elif s < 50 and v > 200:
        return "Gray"
    elif 33 <= h < 78 and s >= 50:
        return "Green"
    elif 22 <= h < 33 and s >= 50:
        return "Hazel"
    elif h < 22 or h >= 130:
        return "Brown"
    elif 78 <= h < 100 and s >= 50:
        return "Blue"
    elif 100 <= h < 130 and s >= 50:
        return "Amber"
    else:
        return "Unknown"

# Enable camera
cap = cv2.VideoCapture(1)  # 1 to get the external webcam
cap.set(3, 640)
cap.set(4, 420)

while True:
    success, frame = cap.read()
    if not success:
        break

    imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_classifier.detectMultiScale(imgGray, 1.3, 5)

    # Draw bounding box around faces and perform emotion detection
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        roi_gray = imgGray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        roi_gray_resized = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray_resized]) != 0:
            roi = roi_gray_resized.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y - 10)  # Adjust the label position
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Detect eyes within the face ROI
        eyes = eye_classifier.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            eye_roi_color = roi_color[ey:ey + eh, ex:ex + ew]
            hsv_eye = cv2.cvtColor(eye_roi_color, cv2.COLOR_BGR2HSV)
            
            # Find the dominant color in the eye region
            h, s, v = cv2.split(hsv_eye)
            h_mean = int(np.mean(h))
            s_mean = int(np.mean(s))
            v_mean = int(np.mean(v))

            eye_color = get_eye_color(h_mean, s_mean, v_mean)
            
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
            cv2.putText(roi_color, eye_color, (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow('Eye Color and Emotion Detector', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
