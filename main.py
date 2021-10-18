import cv2
import numpy as np

videoCapture = cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if videoCapture.isOpened():
    rval, frame = videoCapture.read()
else:
    rval = False

while rval:
    rval, frame = videoCapture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    facez = faceCascade.detectMultiScale(gray, 1.3, 5)

    if (len(facez) == 2):
        (x1, y1, w1, h1) = facez[0]
        (x2, y2, w2, h2) = facez[1]

        cropImg1 = frame[y1:y1+h1, x1:x1+w1]
        cropImg2 = frame[y2:y2+h2, x2:x2+w2]

        height1 = cropImg1.shape[0]
        width1 = cropImg1.shape[1]

        height2 = cropImg2.shape[0]
        width2 = cropImg2.shape[1]

        resized1 = cv2.resize(cropImg1, (w2, h2), interpolation = cv2.INTER_AREA)
        resized2 = cv2.resize(cropImg2, (w1, h1), interpolation = cv2.INTER_AREA)

        frame[y1:y1+h1, x1:x1+w1] = resized2
        frame[y2:y2+h2, x2:x2+w2] = resized1
    cv2.imshow('Camerax', frame)

    key = cv2.waitKey(20)

    if key == 27:
        break

videoCapture.release()
cv2.destroyWindow("Camera")