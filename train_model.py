from os import replace
from modAL.models.learners import Committee
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

import argparse
import pickle
from modAL.models import ActiveLearner
from modAL.uncertainty import entropy_sampling
import numpy as np
from copy import deepcopy
import method_entropy
from method_entropy import vote_uncertain_sampling_entropy
from modAL.disagreement import vote_entropy_sampling

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True,
	help="path to serialized db of facial embeddings")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to output model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to output label encoder")
args = vars(ap.parse_args())


print("[INFO] loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())
# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])
X = data["embeddings"]


X_pool = deepcopy(X)
X_pool = np.asarray(X_pool)
y_pool = deepcopy(labels)
X_train, X_test,y_train,y_test = train_test_split(X_pool,y_pool,test_size=0.2,random_state = 42)

print("[INFO] training model...")
"""
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['linear','rbf', 'poly', 'sigmoid'],'probability': [True]}

grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(data["embeddings"], labels)
"""
kernel = 1.0 * RBF(1.0)
Alg_list = [SVC(C=1,kernel='linear',probability=True),
		RandomForestClassifier(),
		SVC(C=2,probability=True),
		GaussianProcessClassifier(kernel=kernel),
		GaussianNB(),
		MLPClassifier(alpha=1, max_iter=1000),
		]

learner_list = list()

for i in Alg_list:
	learner = ActiveLearner(
		estimator= i,
		X_training= X_train,y_training=y_train
	)
	learner_list.append(learner)

committee = Committee(learner_list=learner_list,
				query_strategy= vote_uncertain_sampling_entropy
				)
y = labels
N_QUERIES = 40
i = 0
minimum_accuracy = 0.6

for idx in range(N_QUERIES):
	try:
		query_idx, query_inst = committee.query(X)

		committee.teach(X=X_pool[query_idx],y=y_pool[query_idx])
		X_pool = np.delete(X_pool, query_idx, axis = 0)
		y_pool = np.delete(y_pool, query_idx)
	except IndexError:
		break



print(committee.score(X_test,y_test))

svc = committee.learner_list[0]

f = open(args["recognizer"], "wb")
f.write(pickle.dumps(committee))
f.close()


f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()