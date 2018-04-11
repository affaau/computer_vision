#!/usr/bin/env python3
'''
Pedestrian detection demo

e.g.:

$ python human_detection_test.py --video \path\video

ref: https://www.pyimagesearch.com/2015/11/09/pedestrian-detection-opencv/
'''

# import the necessary packages
#from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="path to video file")
args = vars(ap.parse_args())
 
vid = cv2.VideoCapture(args["video"])
cv2.namedWindow('Human Detect', cv2.WINDOW_NORMAL)
 
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# loop over the image paths
while True:

    ret, image = vid.read()
    ret, image = vid.read()
    ret, image = vid.read()
    ret, image = vid.read()
    ret, image = vid.read()
    if not ret:
        break
    
    # load the image and resize it to (1) reduce detection time
    # and (2) improve detection accuracy
    image = imutils.resize(image, width=min(400, image.shape[1]))
 
	# detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.03)  # original scale=1.05
 
    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    
    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        
	# show the output images
    cv2.imshow("Human Detect", image)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

vid.release()
cv2.destroyAllWindows()