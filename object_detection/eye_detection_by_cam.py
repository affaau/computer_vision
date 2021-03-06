#!/usr/bin/python3
'''Simple face detection demo of OpenCV3 under Python 3'''

import cv2
import sys

## training
#cascPath = './training_data/haarcascade_frontalface_default.xml'
cascPath = 'C:\\opencv\\sources\\data\\haarcascades\\haarcascade_eye.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

cv2.namedWindow('Eyes', cv2.WINDOW_AUTOSIZE)
cap = cv2.VideoCapture(0)

while True:
   ## read frame
   ret, frame = cap.read()

   ## face detection
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   faces = faceCascade.detectMultiScale(
      gray,
      scaleFactor=1.1,
      minNeighbors=5,
      minSize=(30,30),
      flags=cv2.CASCADE_SCALE_IMAGE
   )
   
   ## put rectangle box over detected face
   for (x, y, w, h) in faces:
      cv2.circle(frame, (x+w//2,y+h//2), max([w//2,h//2]), (255,0,0), 2)
   
   cv2.imshow('Eyes', frame)
   
   ## press ESC terminate demo
   if cv2.waitKey(10) == 27:
      break

cap.release()
cv2.destroyAllWindows()
