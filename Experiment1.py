import argparse
import os
from copy import deepcopy
from math import ceil
import matplotlib.pyplot as plt

import numpy as np
import pandas
from modAL import ActiveLearner
from pandas import Series
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.model_selection import train_test_split
import pickle

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from ModifiedCommittee import ModifiedCommittee
from method_entropy import vote_uncertain_sampling_entropy, vote_disagreement, random_choice, vote_uncertain_sampling_entropy_v2
from plot_stats import plot_evolution, plot_evolution_trend, plot_evolution_speed_trend, plot_best_iteration

'''
This script shall be used to perform an experiment where for each algorithm we perform both active and passive learning and compare the algorithm progression
'''


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--embeddings", required=True, help="path to serialized db of facial embeddings")
    ap.add_argument("-r", "--recognizer", required=True, help="path to output model trained to recognize faces")
    ap.add_argument("-l", "--le", required=True, help="path to output label encoder")
    args = vars(ap.parse_args())
    return args


def setup():
    args = get_args()
    data = pickle.loads(open(args["embeddings"], "rb").read())

    # encode the labels
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])
    x = np.asarray(data["embeddings"])

    # split dataset
    x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2)
    x_train_labeled, x_train_unlabeled, y_train_labeled, y_train_unlabeled = train_test_split(x_train, y_train, test_size=0.8)  # TODO

    # setup learners
    kernel = 1 ** 2 + Matern(length_scale=2, nu=1.5) + WhiteKernel(noise_level=1)
    alg_list = [SVC(C=1, gamma=1, kernel='poly', probability=True),
                RandomForestClassifier(),
                SVC(C=1, gamma=1, kernel='rbf', probability=True),
                GaussianProcessClassifier(kernel=kernel),
                GaussianNB(var_smoothing=0.008111308307896872),
                MLPClassifier(alpha=0.001, max_iter=10000)
                ]
    learner_list = list()
    for i, algo in enumerate(alg_list):
        learner = ActiveLearner(
            estimator=algo,
            X_training=x_train_labeled, y_training=y_train_labeled
        )
        learner_list.append(learner)
        # learner.name = str(algo)
    learner_list[0].name = "SVC_poly"
    learner_list[1].name = "RandomForestClassifier"
    learner_list[2].name = "SVC_rbf"
    learner_list[3].name = "GaussianPC"
    learner_list[4].name = "GaussianNB"
    learner_list[5].name = "MLPClassifier"
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
    learners = [committee_us, committee_d, committee_r, committee_v2]
    learners.extend(learner_list)
    if not os.path.isdir("Experiment1"):
        os.mkdir("Experiment1")
    os.chdir("Experiment1")
    return x_train_unlabeled, y_train_unlabeled, x_test, y_test, learners


def passive_learning(x_train, y_train, x_test, y_test, learner):
    iterations = []
    for x, y in zip(x_train, y_train):
        learner.teach([x], [y])
        iterations.append(learner.score(x_test, y_test))
    return iterations


def active_learning(x_train, y_train, x_test, y_test, learner):
    x_train = deepcopy(x_train)
    y_train = deepcopy(y_train)
    iterations = []
    n_queries = len(y_train)
    for i in range(n_queries):
        try:
            print("%d/%d" % (i, len(y_train)))
            query_idx, query_inst = learner.query(x_train)
            learner.teach(X=x_train[query_idx], y=y_train[query_idx])
            x_train = np.delete(x_train, query_idx, axis=0)
            y_train = np.delete(y_train, query_idx, axis=0)
        except:
            print("An iteration failed")

        iterations.append(learner.score(x_test, y_test))
    return iterations


def calculate_integrals(data, learner):
    try:
        integrals = pandas.read_csv("experiment1-integrals-%s.csv" % learner.name)
        integrals = dict(integrals)
        integrals = {k: list(v) for k, v in integrals.items()}
    except:
        integrals = {
            "First 10": [],
            "First 25": [],
            "First 50": [],
            "100": [],
            "Last 10": [],
            "Last 25": [],
            "Last 50": [],
        }
    integrals["First 10"].append(sum(data["Difference"][0:ceil(len(data["Difference"]) / 10)]))
    integrals["First 25"].append(sum(data["Difference"][0:ceil(len(data["Difference"]) / 4)]))
    integrals["First 50"].append(sum(data["Difference"][0:ceil(len(data["Difference"]) / 2)]))
    integrals["100"].append(sum(data["Difference"][:]))
    integrals["Last 10"].append(sum(data["Difference"][len(data["Difference"]) - ceil(len(data["Difference"]) / 10):]))
    integrals["Last 25"].append(sum(data["Difference"][len(data["Difference"]) - ceil(len(data["Difference"]) / 4):]))
    integrals["Last 50"].append(sum(data["Difference"][len(data["Difference"]) - ceil(len(data["Difference"]) / 2):]))
    integrals.pop('Unnamed: 0', None)
    integrals = pandas.DataFrame(integrals)
    integrals.to_csv("experiment1-integrals-%s.csv" % learner.name)


def experiment(x_train, y_train, x_test, y_test, learners):
    for iteration in range(30):
        # break
        print("----------------------------------------------------------")
        print("Running iteration %d of 30" % (iteration + 1))
        print("----------------------------------------------------------")
        new_learners = deepcopy(learners)
        for learner in new_learners:
            if learner.name == 'GaussianPC' or learner.name == 'GaussianNB':
                continue
            print("Executing experiment on learner %s" % learner.name)
            try:
                print("Loading data...")
                data = pandas.read_csv("experiment1-%s-%d.csv" % (learner.name, iteration))
            except:
                print("Couldn't load data, executing training")
                active = deepcopy(learner)
                print("Training passive learner")
                pl = np.array(passive_learning(x_train, y_train, x_test, y_test, learner))
                print("Training active learner")
                al = np.array(active_learning(x_train, y_train, x_test, y_test, active))
                diff = al - pl
                data = pandas.DataFrame({
                    "Passive Learning": pl,
                    "Active Learning": al,
                    "Difference": diff
                })
                print("Saving data to file")
                data.to_csv("experiment1-%s-%d.csv" % (learner.name, iteration))
                print("Loading data...")  # for some reason this fucks up if it's not loaded from the csv
                data = pandas.read_csv("experiment1-%s-%d.csv" % (learner.name, iteration))
            # plot_evolution(data)
            # plot_evolution_trend(data)
            # plot_evolution_speed_trend(data)
            # plot_best_iteration(data)
            calculate_integrals(data, learner)
    conclusions()


def conclusions():
    for stat in ["First 10", "First 25", "First 50", "100", "Last 10", "Last 25", "Last 50"]:
        x = []
        y = []
        fig = plt.figure()
        for file in os.listdir():
            parts = file.split("-")
            if parts[1] == "integrals":
                integrals = pandas.read_csv(file)
                integrals = dict(integrals)
                integrals = {k: list(v) for k, v in integrals.items()}
                x.append(insert_newlines(parts[-1].split(".")[0], every=16))
                y.append(integrals[stat])
        plt.title("Integral of Active and Passive Learning\nQuality Difference Over\n%s%% Iterations" % stat)
        plt.boxplot(y)
        plt.xticks([i + 1 for i, _ in enumerate(x)], x, rotation=90)

        plt.tight_layout()
        plt.show()


def insert_newlines(string, every=64):
    return '\n'.join(string[i:i + every] for i in range(0, len(string), every))


if __name__ == "__main__":
    x_train, y_train, x_test, y_test, learners = setup()
    experiment(x_train, y_train, x_test, y_test, learners)
    os.chdir("..")
