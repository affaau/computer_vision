#!/usr/bin/python3
'''Simple demo of OpenCV3 with camera'''

import cv2

cv2.namedWindow('test_screen', cv2.WINDOW_AUTOSIZE)
cam = cv2.VideoCapture(0)

while cam.isOpened():
   ret,frame= cam.read()
   cv2.imshow('test_screen', frame)
   
   ## press ESC to terminate demo 
   if cv2.waitKey(5) == 27:
      break

cam.release()
cv2.destroyAlWindows()
