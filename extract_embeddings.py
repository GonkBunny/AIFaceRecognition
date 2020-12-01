# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
import os.path
from PIL import Image

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--dataset", required=True, help="path to input directory of faces + images")
    ap.add_argument("-e", "--embeddings", required=True, help="path to output serialized db of facial embeddings")
    ap.add_argument("-d", "--detector", required=True, help="path to OpenCV's deep learning face detector")
    ap.add_argument("-m", "--embedding-model", required=True, help="path to OpenCV's deep learning face embedding model")
    ap.add_argument("-c", "--confidence", type=float, default=0.4, help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())
    # load our serialized face detector from disk
    print("[INFO] loading face detector...")
    protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
    modelPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    # load our serialized face embedding model from disk
    print("[INFO] loading face recognizer...")
    embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

    print("[INFO] quantifying faces...")
    imagePaths = list(paths.list_images(args["dataset"]))

    # initialize our lists of extracted facial embeddings and
    # corresponding people names
    knownEmbeddings = []
    knownNames = []
    # initialize the total number of faces processed
    total = 0

    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]
        # load the image, resize it to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image
        # dimensions
        pil_image = np.array(Image.open(imagePath))
        image = cv2.cvtColor(pil_image, cv2.COLOR_RGB2BGR)
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]
        imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
        detector.setInput(imageBlob)
        detections = detector.forward()

        if len(detections) > 0:
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]

            if confidence > args["confidence"]:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

                (startX, startY, endX, endY) = box.astype("int")
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]
                if fW < 20 or fH < 20:
                    continue
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()
                knownNames.append(name)
                knownEmbeddings.append(vec.flatten())
                total += 1
        i += 1

    print("[INFO] serializing {} encodings...".format(total))
    data = {"embeddings": knownEmbeddings, "names": knownNames}
    f = open(args["embeddings"], "wb")
    f.write(pickle.dumps(data))
    f.close()
