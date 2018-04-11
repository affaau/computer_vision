'''simple program
to extract video into series of jpg files'''
import cv2
import numpy as np

# manually create path directory first
save_path = ".\\pano2\\"
count = 1

# video to be extracted
cap = cv2.VideoCapture('pano2.mp4')

ret, frame = cap.read()
while ret:
    # named in ascending order from 000001.jpg onwards
    name = save_path + "{0:06d}.jpg".format(count)
    cv2.imwrite(name, frame)
    count = count + 1
    ret, frame = cap.read()

print('extracted %d images'%(count-1))
cap.release()