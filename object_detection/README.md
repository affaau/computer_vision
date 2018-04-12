All programs tested under following environment:

64-bit Window 10
Python 3.5.4
OpenCV 3.4.0
Numpy 1.13.3
Pillow 4.2.1

Various object detection are tested in this project.
Including,
-face,
-car,
-eye,
-smile,
-color,
-pedestrian
-hog_svm_human_detection_2 (NEW)

The code is simple and self explanatory.


######
A document scanner project is included.

ref: https://github.com/vipul-sharma20/document-scanner
######


######
(NEW) Another Human Detection program, inside directory 'human-detector'

ref: https://buptldy.github.io/2016/04/01/2016-04-01-Human%20Detection/
ref: https://github.com/BUPTLdy/human-detector

It consists of examples from extract HOG features from training images, training SVM to testing of images.

Functional scripts are inside 'human-detector/object_detector'

Firstly, run
extract_features.py - extracts HOG features of all images in 
human-detector/data/images 
  /pos_person   (consists of 2416 positive images)
  /neg_person   (consists of 4146 negative images)

and store them into

human-detector/data/features
  /pos
  /neg
  
2. Then run
train_svm.py - which learn from all features above and create a model file inside human-detector/data/model
  /svm.model
  
3. Finally, ready to test images which store inside
humna-detector/object_detector/test_image

by running
detector.py - which extract HOG features of each testing images and being classified by the trained model.

One by one, testing images with human are shown with GREEN rectangles (without suppression). Press enter to show clean up image (after non-max suppression).
Press enter again to show next image...till all images in the test_image are finished.

**Unless new training pictures or training methods are changed, (no need to extract or train svm again) just run 'detector.py' to test new images will do.
######