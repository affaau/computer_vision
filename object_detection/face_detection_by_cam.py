#!/usr/bin/python3
'''Simple face detection demo of OpenCV3 under Python 3'''

import cv2
import sys

## training
#cascPath = './training_data/haarcascade_frontalface_default.xml'
#cascPath = 'C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalcatface.xml'
cascPath = 'C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

cv2.namedWindow('Video', cv2.WINDOW_AUTOSIZE)
cap = cv2.VideoCapture(0)

while True:
   ## read frame
   ret, frame = cap.read()

   ## face detection
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   print(gray.shape)
   faces = faceCascade.detectMultiScale(
      gray,
      scaleFactor=1.1,
      minNeighbors=5,
      minSize=(30,30),
      flags=cv2.CASCADE_SCALE_IMAGE
   )
   
   ## put rectangle box over detected face
   for (x, y, w, h) in faces:
      cv2.rectangle(frame, (x,y),(x+w,y+h), (0,255,0),2)
   
   cv2.imshow('Video', frame)
   
   ## press ESC terminate demo
   if cv2.waitKey(1) == 27:
      break

cap.release()
cv2.destroyAllWindows()
