import os
import cv2 as cv
import numpy as np


people = []

# get the list of people from disk
for i in os.listdir(r'type here the location of resource photos'):
    people.append(i)

dir = r"C:\Users\Hagai\PycharmProjects\Machine Learning\projects\recognition\PyPower_face-recognition\dataset"
# loading haar cascade model of face detection
haar_cascade = cv.CascadeClassifier(r"type here the location of haar cascade frontal face detection")

features = []
labels = []

def create_train():
    # loop over the people (represent in a folder)
    for person in people:
        path = os.path.join(dir, person)
        label = people.index(person)
        # loop over the pictures inside the folder
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_arr = cv.imread(img_path)
            gray = cv.cvtColor(img_arr, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, 1.1,4)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print("Training DONE ----------------")

features = np.array(features, dtype='object')
labels = np.array(labels)

# using OpenCV recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# train the recognizer
face_recognizer.train(features, labels)

# saving the training details for the recognition
face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)