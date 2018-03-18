All programs tested under following environment:

64-bit Window 10
Python 3.5.4
OpenCV 3.4.0
Numpy 1.13.3
Pillow 4.2.1

ref: https://github.com/vipul-sharma20/gesture-opencv

Test program that could be used in hand gesture application.

Embedded camera is used.
Hand is recognized under a sepecific region in the video.
Contour of hand, vertex (between fingers) and tips of fingers are identified.

Messages are shown according to the number of vertex found.

Limitation:
To allow for correct tips & vertex recognition, background color of captured 
video does matter.

If background is bright or white, cv2.THRESH_BINARY_INV is used to reverse 
color. (see source code).

If set correctly, the hand in the 'Thresholded' image should be 'write'.
