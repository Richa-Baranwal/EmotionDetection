import cv2
import numpy as np
from keras.models import load_model
import time

# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# import trained model
model = load_model('model_file_30epochs.h5')

# import Haar Cascade classifier
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# adding labels in data
labels_dict = {0: 'Angry', 1: 'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}

# x: len(number_of_image), y:image_height, w:image_width, h:channel

frame = cv2.imread("small-face.jpg")
# frame = cv2.imread("data03.jpg")
gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = faceDetect.detectMultiScale(gray, 1.3, 3)

# Add ground truth label for the detected face
ground_truth_label = 0  # Replace this with the actual ground truth label
correct_predictions = 0

for x,y,w,h in faces:
    sub_face_img=gray[y:y+h, x:x+w]
    resized=cv2.resize(sub_face_img,(48,48))
    normalize=resized/255.0
    reshaped = np.reshape(normalize, (1, 48, 48, 1))
    result = model.predict(reshaped) 
    label = np.argmax(result, axis=1)[0]
    if label == ground_truth_label:
        correct_predictions += 1
    print(label)
    print(label)
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 1)
    cv2.rectangle(frame, (x,y), (x+w, y+h), (50, 50, 255), 2)
    cv2.putText(frame, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.8,(255, 255, 255),2)
    
cv2.imshow("Frame",frame)
time.sleep(2)

cv2.waitKey(0)
cv2.destroyAllWindows()