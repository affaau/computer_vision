#!/usr/bin/python3
'''Very simple implementation but much error and noise'''

import numpy as np
import cv2


def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


def draw_detections(img, rects, thickness = 2):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)


if __name__ == '__main__':
    cv2.namedWindow('feed', cv2.WINDOW_NORMAL)
    cv2.namedWindow('gray & blur', cv2.WINDOW_NORMAL)
    
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )
    #cap=cv2.VideoCapture('pano2.mp4')
    
    #             (start, stop, step)
    for i in range(200,700,5):
        #_,frame=cap.read()
        #frame = cv2.imread(".\\img_series\\{0:06d}.jpg".format(i), cv2.IMREAD_COLOR)
        frame = cv2.imread(".\\pano2\\{0:06d}.jpg".format(i), cv2.IMREAD_COLOR)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # remove noise
        img = cv2.GaussianBlur(gray,(11,11),0)
        #img = cv2.medianBlur(gray, 5)
        
        found,w = hog.detectMultiScale(img, winStride=(8,8), padding=(32,32), scale=1.2)
        draw_detections(frame,found)
        cv2.imshow('feed', frame)
        cv2.imshow('gray & blur', img)
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break
    
    cv2.waitKey()
    cv2.destroyAllWindows()