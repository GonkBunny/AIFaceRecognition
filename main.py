from subprocess import Popen
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
Dataset = "Dataset"
recognizer = "Output\recognizer.pickle"
le = "Output\le.pickle"
embedding_model = "openface.nn4.small2.v1.t7"
detector = "face_detection_model"
embeddings = "Output\embeddings.pickle"





