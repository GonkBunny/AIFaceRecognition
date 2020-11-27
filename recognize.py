from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
from imutils import paths
import imutils
import pickle
import time
import cv2
import os

import os.path
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
	help="path to face detector")
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())


print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))


for (i, imagePath) in enumerate(imagePaths):
      print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
      name = imagePath.split(os.path.sep)[-2]
	# load the image, resize it to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image
	# dimensions
      pil_image = np.array(Image.open(imagePath))
      image = cv2.cvtColor(pil_image,cv2.COLOR_RGB2BGR)
      image = imutils.resize(image, width=600)
      (h, w) = image.shape[:2]
      imageBlob = cv2.dnn.blobFromImage(cv2.resize(image,(300,300)),1.0,(300,300),(104.0,177.0,123.0), swapRB=False,crop=False)
 
      detector.setInput(imageBlob)
      detections = detector.forward()
      for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections
            if confidence > args["confidence"]:
                  box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                  (startX, startY, endX, endY) = box.astype("int")
                  # extract the face ROI
                  face = image[startY:endY, startX:endX]
                  (fH, fW) = face.shape[:2]
                  # ensure the face width and height are sufficiently large
                  if fW < 20 or fH < 20:
                        continue
            
                  faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                              (96, 96), (0, 0, 0), swapRB=True, crop=False)
                  embedder.setInput(faceBlob)
                  vec = embedder.forward()
                  # perform classification to recognize the face
                  preds = recognizer.predict_proba(vec)[0]
                  j = np.argmax(preds)
                  proba = preds[j]
                  name = le.classes_[j]
                  # draw the bounding box of the face along with the
                  # associated probability
                  text = "{}: {:.2f}%".format(name, proba * 100)
                  y = startY - 10 if startY - 10 > 10 else startY + 10
                  cv2.rectangle(image, (startX, startY), (endX, endY),
                        (0, 0, 255), 2)
                  cv2.putText(image, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
      cv2.imshow(image)