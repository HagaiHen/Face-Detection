import numpy as np
import cv2 as cv
import os

people = []

# get the list of people from disk
for i in os.listdir(r'type here the location of resource photos'):
    people.append(i)

# loading haar cascade model of face detection
haar_cascade = cv.CascadeClassifier(r"type here the location of haar cascade frontal face detection")

features = np.load("features.npy", allow_pickle=True)
labels = np.load("labels.npy")

# using OpenCV face recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_trained.yml")    #reading the file after the train

img = cv.imread(r"type here the location of the image for test")

# convert the photo to black & white photo
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# using haar cascade model on the gray img
faces_rect = haar_cascade.detectMultiScale(gray, 1.25, 4)

for (x, y, w, h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+h]

    label, confidence = face_recognizer.predict(faces_roi)  #predict the name of the person by the face zone (faces_roi)
    print(f"label = {people[label]} with a confidence of {confidence}")

    cv.putText(img, str(people[label]), (x,y), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), 2)
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)  #draw a rectangle around the face

cv.imshow("DETECTED", img)
cv.waitKey(0)