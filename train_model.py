import itertools
import random as rd
from math import gamma
import os
from modAL.models.learners import Committee
from scipy.sparse.construct import random
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier

import argparse
import pickle
from modAL.models import ActiveLearner
import numpy as np
from copy import deepcopy
from method_entropy import vote_disagreement, vote_uncertain_sampling_entropy, random_choice, vote_uncertain_sampling_entropy_v2

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from ModifiedCommittee import ModifiedCommittee

seed_value = 42


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--embeddings", required=True, help="path to serialized db of facial embeddings")
    ap.add_argument("-r", "--recognizer", required=True, help="path to output model trained to recognize faces")
    ap.add_argument("-l", "--le", required=True, help="path to output label encoder")
    args = vars(ap.parse_args())
    return args


def run_iteration(args, file_stats, itera, iterations, verbose=True, partition=(0.4, 0.6)):
    if sum(partition) != 1:
        print("WRONG PARTITIONING")
        return
    if verbose:
        print("STARTING ITERATION %d OUT OF %d" % (itera + 1, iterations))
        print("[INFO] loading face embeddings...")
    data = pickle.loads(open(args["embeddings"], "rb").read())
    # encode the labels
    if verbose:
        print("[INFO] encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])
    x = np.asarray(data["embeddings"])
    x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=42)
    x_train_labeled, x_train_unlabeled, y_train_labeled, y_train_unlabeled = train_test_split(x_train, y_train, test_size=partition[1], random_state=42)  # TODO

    print("[INFO] training model...")
    """
    param_grid = {'C': [0.1,1, 10, 100,1000,10000], 'gamma': [1,0.1,0.01,0.001],'kernel': ['linear', 'poly','rbf', 'sigmoid'],'probability': [True]}
    params_NB = {'var_smoothing': np.logspace(0,-9, num=100)}
    grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=1,scoring='accuracy')
    grid.fit(data["embeddings"], labels)
    print("%s",grid.best_params_)
    """

    kernel = 1 ** 2 + Matern(length_scale=2, nu=1.5) + WhiteKernel(noise_level=1)

    alg_list = [SVC(C=1, gamma=1, kernel='poly', probability=True, random_state=42),
                RandomForestClassifier(random_state=42),
                SVC(C=1, gamma=1, kernel='rbf', probability=True, random_state=42),
                GaussianProcessClassifier(kernel=kernel, random_state=42),
                GaussianNB(var_smoothing=0.008111308307896872),
                MLPClassifier(alpha=0.001, max_iter=10000, random_state=42)
                ]

    b = deepcopy(alg_list)
    """
    for i in b:
        scores = cross_val_score(i,x,labels)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
"""
    learner_list = list()

    for i, algo in enumerate(alg_list):
        learner = ActiveLearner(
            estimator=algo,
            X_training=x_train_labeled, y_training=y_train_labeled
        )
        learner_list.append(learner)
        learner.name = str(algo)

    learner_list_us = deepcopy(learner_list)
    learner_list_d = deepcopy(learner_list)
    learner_list_r = deepcopy(learner_list)
    learner_list_v2 = deepcopy(learner_list)
    committee_us = ModifiedCommittee(learner_list=learner_list_us, query_strategy=vote_uncertain_sampling_entropy)
    committee_us.name = "vote_uncertain_sampling_entropy"
    committee_d = ModifiedCommittee(learner_list=learner_list_d, query_strategy=vote_disagreement)
    committee_d.name = "vote_disagreement"
    committee_r = ModifiedCommittee(learner_list=learner_list_r, query_strategy=random_choice)
    committee_r.name = "random_choice"
    committee_v2 = ModifiedCommittee(learner_list=learner_list_v2, query_strategy=vote_uncertain_sampling_entropy_v2)
    committee_v2.name = "vote_uncertain_sampling_entropy_v2"

    evaluation_list = deepcopy([committee_v2, committee_us, committee_d, committee_r] + learner_list)
    # evaluation_list = deepcopy(alg_list)
    # evaluation_list = deepcopy(learner_list)

    def evaluation():

        y = label_binarize(labels, classes=np.unique(labels))

        y_train, y_test = train_test_split(y, test_size=0.2, random_state=42)

        for i, algo in enumerate(evaluation_list):
            classifier = OneVsRestClassifier(evaluation_list[i]).fit(x_train, y_train)
            y_pred = classifier.predict_proba(x_test)
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            precision = dict()
            recall = dict()
            average_precision = dict()
            plt.figure()
            for n in range(len(np.unique(y_test))):
                fpr[n], tpr[n], _ = roc_curve(y_test[:, n], y_pred[:, n])
                roc_auc[n] = auc(fpr[n], tpr[n])
                # plt.plot(fpr[n],tpr[n],label = "ROC Curve of class "+ str(le.inverse_transform(np.unique(y_test))[n]))
                precision[n], recall[n], _ = precision_recall_curve(y_test[:, n], y_pred[:, n])
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            plt.plot([0, 1], [0, 1], 'k--')
            precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_pred.ravel())
            average_precision["micro"] = average_precision_score(y_test, y_pred, average="micro")
            plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["micro"]), linestyle=':', linewidth=4)
            print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))
            plt.xlabel('False positive')
            plt.ylabel('True positive')
            plt.title('ROC curve ' + str(algo))
            plt.legend()
            plt.savefig('plots/ROC_curve_%s.png' % str(algo).split("(")[0])
            plt.show()

            plt.figure()
            plt.plot(recall["micro"], precision["micro"], lw=2, label='Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))
            plt.title('Precision score ' + str(algo))
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend()
            plt.savefig('plots/Precision_Recall_%s.png' % str(algo).split("(")[0])
            plt.show()

    evaluation()

    solo_learner_x_pools = [deepcopy(x_train_unlabeled) for _ in learner_list]
    solo_learner_y_pools = [deepcopy(y_train_unlabeled) for _ in learner_list]
    x_pool_us = deepcopy(x_train_unlabeled)
    y_pool_us = deepcopy(y_train_unlabeled)
    x_pool_d = deepcopy(x_train_unlabeled)
    y_pool_d = deepcopy(y_train_unlabeled)
    x_pool_r = deepcopy(x_train_unlabeled)
    y_pool_r = deepcopy(y_train_unlabeled)
    x_pool_v2 = deepcopy(x_train_unlabeled)
    y_pool_v2 = deepcopy(y_train_unlabeled)

    for learner in itertools.chain([committee_v2], [committee_us], [committee_d], [committee_r], learner_list):
        plot_confusion(learner, x_test, y_test, le, "Before Active Learning %s" % learner.name)

    n_queries = len(y_train_unlabeled)
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
            x_pool_r = np.delete(x_pool_r, query_idx2, axis=0)
            y_pool_r = np.delete(y_pool_r, query_idx2)

            # COMMITTEE 4
            query_idx3, query_inst3 = committee_v2.query(x_pool_v2)
            committee_v2.teach(X=x_pool_v2[query_idx3], y=y_pool_v2[query_idx3])
            x_pool_v2 = np.delete(x_pool_v2, query_idx3, axis=0)
            y_pool_v2 = np.delete(y_pool_v2, query_idx3)

            # ACTIVE LEARNERS
            for i in range(len(learner_list)):
                query_idx2, query_inst2 = learner_list[i].query(solo_learner_x_pools[i])
                learner_list[i].teach(X=solo_learner_x_pools[i][query_idx2], y=solo_learner_y_pools[i][query_idx2])
                solo_learner_x_pools[i] = np.delete(solo_learner_x_pools[i], query_idx2, axis=0)
                solo_learner_y_pools[i] = np.delete(solo_learner_y_pools[i], query_idx2)

            file_stats.write("%d" % idx)
            for learner in itertools.chain([committee_v2], [committee_us], [committee_d], [committee_r], learner_list):
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
    for learner in itertools.chain([committee_v2], [committee_us], [committee_d], [committee_r], learner_list):
        plot_confusion(learner, x_test, y_test, le, "After Active Learning %s" % learner.name)

    f = open(args["recognizer"], "wb")
    f.write(pickle.dumps(committee_v2))
    f.close()

    f = open(args["le"], "wb")
    f.write(pickle.dumps(le))
    f.close()


def plot_confusion(learner, x_test, y_test, le, title):
    plt.figure()
    conf_mat = confusion_matrix(y_test, learner.predict(x_test))
    df_cm = pd.DataFrame(conf_mat, index=[i for i in range(len(conf_mat))], columns=[i for i in range(len(conf_mat))])
    sn.heatmap(df_cm, annot=True)
    plt.title(title)
    locs, labels = plt.xticks()
    plt.xticks(locs, le.inverse_transform([int(label._text) for label in labels]), rotation=70)
    locs, labels = plt.yticks()
    plt.yticks(locs, le.inverse_transform([int(label._text) for label in labels]), rotation=0)
    title = title.split("(")[0]
    plt.tight_layout()
    plt.savefig("plots/%s.png" % title)
    plt.show()


def run_experiment(args, iterations=10, partition=(0.4, 0.6)):
    file_stats = open("iteration_stats.csv", "w")
    file_stats.write("iteration,Entropy Committee with Uncertainty Sampling, Voter Entropy Committee,Voter Disagreement Committee, Random Committee")
    for i in range(1, 7):
        file_stats.write(",Solo Active Learner %d" % i)
    file_stats.write("\n")
    for itera in range(iterations):
        run_iteration(args, file_stats, itera, iterations, partition=partition)

    # svc = committee.learner_list[0]

    file_stats.close()


def main():
    iterations = 1
    args = get_args()
    run_experiment(args, iterations, partition=(0.1, 0.9))


if __name__ == "__main__":
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    rd.seed(seed_value)
    np.random.seed(seed_value)
    main()
