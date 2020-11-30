# AIFaceRecognition
 

extract_embeddings.py -i Dataset -d face_detection_model -e Output\embeddings.pickle -m openface.nn4.small2.v1.t7


train_model.py -e Output\embeddings.pickle -r  Output\recognizer.pickle -l Output\le.pickle


recognize_video.py -r Output\recognizer.pickle -l Output\le.pickle -d face_detection_model -m nn4.small2.v1.t7