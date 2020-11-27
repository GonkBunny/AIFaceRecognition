import cv2
import sys
import numpy as np
import os

image_path= "Dataset\Gonçalo\GoncaloCoelho.png"
print(os.path.exists(image_path))
print(image_path)



CV_LOAD_IMAGE_COLOR = 1 # set flag to 1 to give colour image
CV_LOAD_IMAGE_COLOR = 0 # set flag to 0 to give a grayscale one
img = cv2.imread(image_path,CV_LOAD_IMAGE_COLOR)
print(img.shape)
cv2.namedWindow('Display Window') ## create window for display
cv2.imshow('Display Window', img) ## Show image in the window
cv2.waitKey(0) ## Wait for keystroke
cv2.destroyAllWindows() ## Destroy all windows