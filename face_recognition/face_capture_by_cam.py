#!/usr/bin/python3
'''Simple face detection demo of OpenCV3 under Python 3'''

import cv2, os
import sys
#from timeit import default_timer as timer
from time import time as timer

def is_similar(previous, present):
   diff = [abs(previous_pos[0]-present[0]), 
           abs(previous_pos[1]-present[1]), 
           abs(previous_pos[2]-present[2]), 
           abs(previous_pos[3]-present[3])]
   if min(diff) < 2:     # small difference implies not moving
      return True
   
   return False

path = './pic_database'
subject = input("Please enter your nickname: ")

## training
cascPath = './training_data/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

cv2.namedWindow('Video', cv2.WINDOW_AUTOSIZE)
cap = cv2.VideoCapture(0)

count = 0
same_count = 0
start = timer()
previous_pos = (0,0,1,1)
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
      cv2.rectangle(frame, (x,y),(x+w,y+h), (0,255,0),2) 
      
      if is_similar(previous_pos, (x,y,w,h)):
         if same_count < 10:
            same_count += 1
            break      
         else:
            img_name = subject+'%02d'%(count,)+'.png'
            count += 1
            cv2.imwrite(os.path.join(path, img_name ), gray[y:y+h, x:x+w])
            print('{}.\ttime lapsed {}'.format(count, timer() - start))
            start = timer()
            same_count = 0
      else:
         previous_pos = (x,y,w,h)
         same_count = 0

   cv2.imshow('Video', frame)
   
## press ESC terminate demo
   if cv2.waitKey(100) == 27:
      break
   
   if count >= 10:
      break 

cap.release()
cv2.destroyAllWindows()
