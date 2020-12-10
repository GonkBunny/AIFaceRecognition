import itertools
from modAL.models.learners import Committee
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

import argparse
import pickle
from modAL.models import ActiveLearner
import numpy as np
from copy import deepcopy
from method_entropy import vote_disagreement, vote_uncertain_sampling_entropy, random_choice


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--embeddings", required=True, help="path to serialized db of facial embeddings")
    ap.add_argument("-r", "--recognizer", required=True, help="path to output model trained to recognize faces")
    ap.add_argument("-l", "--le", required=True, help="path to output label encoder")
    args = vars(ap.parse_args())
    return args


def run_iteration(args, file_stats, itera, iterations, verbose=True):
    if verbose:
        print("STARTING ITERATION %d OUT OF %d" % (itera + 1, iterations))
        print("[INFO] loading face embeddings...")
    data = pickle.loads(open(args["embeddings"], "rb").read())
    # encode the labels
    if verbose:
        print("[INFO] encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])
    x = data["embeddings"]

    # np.random.seed(42)
    x_pool_us = deepcopy(x)
    x_pool_us = np.asarray(x_pool_us)
    y_pool_us = deepcopy(labels)
    x_train, x_test, y_train, y_test = train_test_split(x_pool_us, y_pool_us, test_size=0.2)  # TODO
    x_train_labeled, x_train_unlabeled, y_train_labeled, y_train_unlabeled = train_test_split(x_train, y_train, test_size=0.8)  # TODO

    print("[INFO] training model...")
    """
    param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['linear','rbf', 'poly', 'sigmoid'],'probability': [True]}

    grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
    grid.fit(data["embeddings"], labels)
    """
    kernel = 1.0 * RBF(1.0)
    alg_list = [SVC(C=1, kernel='linear', probability=True),
                RandomForestClassifier(),
                SVC(C=2, probability=True),
                GaussianProcessClassifier(kernel=kernel),
                GaussianNB(),
                MLPClassifier(alpha=0.01, max_iter=1000),
                ]

    learner_list = list()

    for i, algo in enumerate(alg_list):
        learner = ActiveLearner(
            estimator=algo,
            X_training=x_train_labeled, y_training=y_train_labeled
        )
        learner_list.append(learner)

    learner_list_us = deepcopy(learner_list)
    learner_list_d = deepcopy(learner_list)
    learner_list_r = deepcopy(learner_list)
    committee_us = Committee(learner_list=learner_list_us, query_strategy=vote_uncertain_sampling_entropy)
    committee_d = Committee(learner_list=learner_list_d, query_strategy=vote_disagreement)
    committee_r = Committee(learner_list=learner_list_r, query_strategy=random_choice)

    solo_learner_x_pools = [deepcopy(x_train_unlabeled) for _ in learner_list]
    solo_learner_y_pools = [deepcopy(y_train_unlabeled) for _ in learner_list]
    x_pool_us = deepcopy(x_train_unlabeled)
    y_pool_us = deepcopy(y_train_unlabeled)
    x_pool_d = deepcopy(x_train_unlabeled)
    y_pool_d = deepcopy(y_train_unlabeled)
    x_pool_r = deepcopy(x_train_unlabeled)
    y_pool_r = deepcopy(y_train_unlabeled)

    n_queries = len(y_train_unlabeled) // 2
    for idx in range(n_queries):
        if verbose:
            print("Executing query %d/%d\n" % (idx + 1, n_queries))
        try:
            # COMMITTEE 1
            query_idx, query_inst = committee_us.query(x_pool_us)
            committee_us.teach(X=x_pool_us[query_idx], y=y_pool_us[query_idx])
            x_pool_us = np.delete(x_pool_us, query_idx, axis=0)
            y_pool_us = np.delete(y_pool_us, query_idx)

            # COMMITTEE 2
            query_idx1, query_inst1 = committee_d.query(x_pool_d)
            committee_d.teach(X=x_pool_d[query_idx1], y=y_pool_d[query_idx1])
            x_pool_d = np.delete(x_pool_d, query_idx1, axis=0)
            y_pool_d = np.delete(y_pool_d, query_idx1)

            # COMMITTEE 3
            query_idx2, query_inst2 = committee_r.query(x_pool_r)
            committee_r.teach(X=x_pool_r[query_idx2], y=y_pool_r[query_idx2])
            x_pool_r = np.delete(x_pool_d, query_idx2, axis=0)
            y_pool_r = np.delete(y_pool_r, query_idx2)

            # ACTIVE LEARNERS
            for i in range(len(learner_list)):
                query_idx2, query_inst2 = learner_list[i].query(solo_learner_x_pools[i])
                learner_list[i].teach(X=solo_learner_x_pools[i][query_idx2], y=solo_learner_y_pools[i][query_idx2])
                solo_learner_x_pools[i] = np.delete(solo_learner_x_pools[i], query_idx2, axis=0)
                solo_learner_y_pools[i] = np.delete(solo_learner_y_pools[i], query_idx2)

            file_stats.write("%d" % idx)
            for learner in itertools.chain([committee_us], [committee_d], [committee_r], learner_list):
                try:
                    score = learner.score(x_test, y_test)
                except:
                    score = float("nan")
                file_stats.write(",%f" % score)
            file_stats.write("\n")
            file_stats.flush()
        except IndexError:
            if verbose:
                print("It broke")
            break


def run_experiment(args, iterations=10):
    file_stats = open("iteration_stats.csv", "w")
    file_stats.write("iteration,Uncertainty Sampling Entropy Committee, Voter Disagreement Committee, Random Committee")
    for i in range(1, 7):
        file_stats.write(",Solo Active Learner %d" % i)
    file_stats.write("\n")
    for itera in range(iterations):
        run_iteration(args, file_stats, itera, iterations)

    # svc = committee.learner_list[0]

    # f = open(args["recognizer"], "wb")
    # f.write(pickle.dumps(committee))
    # f.close()
    #
    # f = open(args["le"], "wb")
    # f.write(pickle.dumps(le))
    # f.close()

    file_stats.close()


def main():
    iterations = 1
    args = get_args()
    run_experiment(args, iterations)


if __name__ == "__main__":
    main()
