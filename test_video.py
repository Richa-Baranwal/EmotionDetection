import cv2
import numpy as np
from keras.models import load_model

# Load trained model
model = load_model('model_file_30epochs.h5')

video = cv2.VideoCapture(0)

# Load Haar Cascade classifier
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Emotion labels
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

while True:
    ret, frame = video.read()
    if not ret:  # Stop if frame is not captured
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for x, y, w, h in faces:
        sub_face_img = gray[y:y+h, x:x+w]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized.astype(np.float32) / 255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        result = model.predict(reshaped, verbose=0)
        label = np.argmax(result, axis=1)[0]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Frame", frame)

    # Check if 'q' is pressed or window is closed
    if cv2.waitKey(1) == ord('q') or cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) < 1:
        break

video.release()
cv2.destroyAllWindows()
