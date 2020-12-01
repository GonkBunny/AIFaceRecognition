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
from method_entropy import vote_disagreement, vote_uncertain_sampling_entropy, random_choice

if __name__ == "__main__":
    ITERATIONS = 10
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--embeddings", required=True, help="path to serialized db of facial embeddings")
    ap.add_argument("-r", "--recognizer", required=True, help="path to output model trained to recognize faces")
    ap.add_argument("-l", "--le", required=True, help="path to output label encoder")
    args = vars(ap.parse_args())

    file_stats = open("iteration_stats.csv", "w")
    file_stats.write("iteration,Uncertainty Sampling Entropy Committee,Random Committee,Solo Active Learner\n")
    for itera in range(ITERATIONS):
        print("STARTING ITERATION %d OUT OF %d" % (itera+1, ITERATIONS))
        print("[INFO] loading face embeddings...")
        data = pickle.loads(open(args["embeddings"], "rb").read())
        # encode the labels
        print("[INFO] encoding labels...")
        le = LabelEncoder()
        labels = le.fit_transform(data["names"])
        X = data["embeddings"]

        # np.random.seed(42)
        X_pool = deepcopy(X)
        X_pool = np.asarray(X_pool)
        y_pool = deepcopy(labels)
        X_train, X_test, y_train, y_test = train_test_split(X_pool, y_pool, test_size=0.2)  # TODO
        X_train_labeled, X_train_unlabeled, y_train_labeled, y_train_unlabeled = train_test_split(X_train, y_train, test_size=0.8)  # TODO

        print("[INFO] training model...")
        """
        param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['linear','rbf', 'poly', 'sigmoid'],'probability': [True]}
        
        grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
        grid.fit(data["embeddings"], labels)
        """
        kernel = 1.0 * RBF(1.0)
        Alg_list = [SVC(C=1, kernel='linear', probability=True),
                    RandomForestClassifier(),
                    SVC(C=2, probability=True),
                    GaussianProcessClassifier(kernel=kernel),
                    GaussianNB(),
                    MLPClassifier(alpha=0.01, max_iter=1000),
                    ]

        learner_list = list()

        for algo in Alg_list:
            learner = ActiveLearner(
                estimator=algo,
                X_training=X_train_labeled, y_training=y_train_labeled
            )
            learner_list.append(learner)

        # só um membro apenas Podia usar um elemento do committee, mas pode causar erro
        activeLearner = ActiveLearner(
            estimator=SVC(C=1, kernel='linear', probability=True),
            X_training=X_train_labeled, y_training=y_train_labeled,
            query_strategy=entropy_sampling
        )

        committee = Committee(learner_list=learner_list, query_strategy=vote_uncertain_sampling_entropy)
        # committee query strategy
        committee2 = Committee(learner_list=learner_list, query_strategy=random_choice)
        y = labels
        N_QUERIES = len(y_train_unlabeled) // 2
        algo = 0
        minimum_accuracy = 0.6
        X_pool = deepcopy(X_train_unlabeled)
        y_pool = deepcopy(y_train_unlabeled)
        X_pool_US = deepcopy(X_train_unlabeled)
        y_pool_US = deepcopy(y_train_unlabeled)
        X_pool_C = deepcopy(X_train_unlabeled)
        y_pool_C = deepcopy(y_train_unlabeled)

        for idx in range(N_QUERIES):
            print("Executing query %d/%d\n" % (idx+1, N_QUERIES))
            try:
                query_idx, query_inst = committee.query(X_pool)
                # print(query_idx)
                committee.teach(X=X_pool[query_idx], y=y_pool[query_idx])
                X_pool = np.delete(X_pool, query_idx, axis=0)
                y_pool = np.delete(y_pool, query_idx)

                query_idx1, query_inst1 = committee2.query(X_pool_C)
                # print(query_idx1)
                committee2.teach(X=X_pool_C[query_idx1], y=y_pool_C[query_idx1])
                X_pool_C = np.delete(X_pool_C, query_idx1, axis=0)
                y_pool_C = np.delete(y_pool_C, query_idx1)

                query_idx2, query_inst2 = activeLearner.query(X_pool_US)
                # print(query_idx2)
                activeLearner.teach(X=X_pool_US[query_idx2], y=y_pool_US[query_idx2])
                X_pool_US = np.delete(X_pool_US, query_idx2, axis=0)
                y_pool_US = np.delete(y_pool_US, query_idx2)
                print("\n")
                try:
                    score1 = committee.score(X_test, y_test)
                except:
                    score1 = float("nan")
                try:
                    score2 = committee2.score(X_test, y_test)
                except:
                    score2 = float("nan")
                try:
                    score3 = activeLearner.score(X_test, y_test)
                except:
                    score3 = float("nan")
                file_stats.write("%d,%f,%f,%f\n" % (idx, score1, score2, score3))
                file_stats.flush()
            except IndexError:
                print("It broke")
                break

        print(committee.score(X_test, y_test))
        print(activeLearner.score(X_test, y_test))
        print(committee2.score(X_test, y_test))

    # svc = committee.learner_list[0]

    f = open(args["recognizer"], "wb")
    f.write(pickle.dumps(committee))
    f.close()

    f = open(args["le"], "wb")
    f.write(pickle.dumps(le))
    f.close()

    file_stats.close()
