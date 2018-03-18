#!/usr/bin/python3

# Import the required modules
import cv2, os, sys
import numpy as np
from PIL import Image

# For face detection we will use the Haar Cascade provided by OpenCV
cascadePath = "./training_data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# For face recognition we will use the LBPH Face Recognizer 
recognizer = cv2.face.LBPHFaceRecognizer_create()
# Retrieve previouly trained dataset
recognizer.read('./trained_data/trained.yml')  # load() is deprecated!!

# expected valid pic & path input from command line
image_path = sys.argv[1]   

predict_image_pil = Image.open(image_path).convert('L')
predict_image = np.array(predict_image_pil, 'uint8')
faces = faceCascade.detectMultiScale(predict_image)
for (x, y, w, h) in faces:
    nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
    #nbr_actual = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
    #if nbr_actual == nbr_predicted:
    #    print("{} is Correctly Recognized with confidence {}".format(nbr_actual, conf))
    #else:
    #    print("{} is Incorrect Recognized as {}".format(nbr_actual, nbr_predicted))
    print("It is Recognized as {} with confidence {}".format(nbr_predicted, conf))

    cv2.imshow("Recognizing Face", predict_image[y: y + h, x: x + w])
    cv2.waitKey(1000)

cv2.waitKey()
cv2.destroyAllWindows()
