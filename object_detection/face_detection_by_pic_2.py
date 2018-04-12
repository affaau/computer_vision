#!/usr/bin/python3
'''Simple face detection demo of OpenCV3 under Python 3'''

import cv2
import sys
import numpy as np
from PIL import Image

## training
cascPath = './training_data/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

cv2.namedWindow('detect', cv2.WINDOW_NORMAL)
cv2.namedWindow('gray', cv2.WINDOW_NORMAL)

## read frame
img = cv2.imread(sys.argv[1])

## face detection
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)
#gray = cv2.GaussianBlur(gray,(5,5),0)
faces = faceCascade.detectMultiScale(
    gray,
    
    scaleFactor=1.03,
    minNeighbors=5,
    minSize=(30,30),
    flags=cv2.CASCADE_SCALE_IMAGE
)
   
## put rectangle box over detected face
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x,y),(x+w,y+h), (0,255,0),2)
   
cv2.imshow('detect', img)
cv2.imshow('gray', gray)

cv2.waitKey()
cv2.destroyAllWindows()
