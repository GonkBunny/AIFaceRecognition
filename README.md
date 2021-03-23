# AIFaceRecognition
 
Face Detection and Embeddings
extract_embeddings.py -i Dataset -d face_detection_model -e Output\embeddings.pickle -m openface.nn4.small2.v1.t7

Face Recognition
train_model.py -e Output\embeddings.pickle -r  Output\recognizer.pickle -l Output\le.pickle

Webcam
recognize_video.py -r Output\recognizer.pickle -l Output\le.pickle -d face_detection_model -m nn4.small2.v1.t7

==================================================================================================================

To see plots used in the paper

plot_stats.py

Experiment1.py -e Output\embeddings.pickle -r  Output\recognizer.pickle -l Output\le.pickle



Dataset -> Package of Packages identifying the person
Ex:
Dataset
|
|--------Barack Obama-------Obama.png
