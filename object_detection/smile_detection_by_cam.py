#!/usr/bin/python3
'''Simple face detection demo of OpenCV3 under Python 3'''

import cv2
import sys

## training
facePath = 'training_data\\haarcascade_frontalface_default.xml'
smilePath = 'training_data\\haarcascade_smile.xml'
faceCascade = cv2.CascadeClassifier(facePath)
smileCascade = cv2.CascadeClassifier(smilePath)

cv2.namedWindow('Smile', cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture(0)

while cap.isOpened():
   ## read frame
   ret, frame = cap.read()

   ## face detection
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   faces = faceCascade.detectMultiScale(
      gray,
      scaleFactor=1.05,
      minNeighbors=8,
      minSize=(55,55),
      flags=cv2.CASCADE_SCALE_IMAGE
   )
   
   count = 0
   ## put rectangle box over detected face
   for (x, y, w, h) in faces:
      cv2.rectangle(frame, (x,y),(x+w,y+h), (0,255,0),2)
      cv2.putText(frame, 'face '+str(count), (x, y-4), cv2.FONT_HERSHEY_PLAIN,
                  1, (0,255,0))
      count +=1
      roi_gray=gray[y:y+h, x:x+w]
      roi_color=frame[y:y+h, x:x+w]
      
      smile=smileCascade.detectMultiScale(
         roi_gray,
         scaleFactor=1.7,
         minNeighbors=22,
         minSize=(25, 25),
         flags=cv2.CASCADE_SCALE_IMAGE
      )
      
      for (x,y,w,h) in smile:
         #print("Found", len(smile), "smiles!")
         cv2.rectangle(roi_color, (x, y), (x+w, y+h), (255, 0, 0), 1)
   
   cv2.imshow('Smile', frame)
   
   ## press ESC terminate demo
   if cv2.waitKey(7) == 27:
      break

cap.release()
cv2.destroyAllWindows()
