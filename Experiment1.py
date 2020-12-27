import argparse
import os
from copy import deepcopy
from math import ceil
import matplotlib.pyplot as plt

import numpy as np
import pandas
from modAL import ActiveLearner
from pandas import Series
from scipy.stats import shapiro, ttest_ind, bartlett, kstest
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.model_selection import train_test_split
import pickle

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from statsmodels.formula.api import ols
from statsmodels.graphics.factorplots import interaction_plot
from statsmodels.stats.anova import anova_lm

from ModifiedCommittee import ModifiedCommittee
from method_entropy import vote_uncertain_sampling_entropy, vote_disagreement, random_choice, vote_uncertain_sampling_entropy_v2
from plot_stats import plot_evolution, plot_evolution_trend, plot_evolution_speed_trend, plot_best_iteration
from termcolor import colored

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
    integrals["First 10"].append(sum(data["Difference"][0:ceil(len(data["Difference"]) / 10)]) / 10)
    integrals["First 25"].append(sum(data["Difference"][0:ceil(len(data["Difference"]) / 4)]) / 25)
    integrals["First 50"].append(sum(data["Difference"][0:ceil(len(data["Difference"]) / 2)]) / 50)
    integrals["100"].append(sum(data["Difference"][:]) / 100)
    integrals["Last 10"].append(sum(data["Difference"][len(data["Difference"]) - ceil(len(data["Difference"]) / 10):]) / 10)
    integrals["Last 25"].append(sum(data["Difference"][len(data["Difference"]) - ceil(len(data["Difference"]) / 4):]) / 25)
    integrals["Last 50"].append(sum(data["Difference"][len(data["Difference"]) - ceil(len(data["Difference"]) / 2):]) / 50)
    integrals.pop('Unnamed: 0', None)
    integrals = pandas.DataFrame(integrals)
    integrals.to_csv("experiment1-integrals-%s.csv" % learner.name)


def experiment(x_train, y_train, x_test, y_test, learners):
    for iteration in range(30):
        break  # TODO comment this out
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
    # PLOTS
    two_way_anova("Section", "Integral_of_Differences")
    for stat in ["First 25", "First 50", "Last 25", "Last 50", "100"]:
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
        plt.title("Active and Passive Learning\n Mean Quality Difference\nOver %s%% Iterations" % stat)
        plt.boxplot(y)
        plt.xticks([i + 1 for i, _ in enumerate(x)], x, rotation=90)
        plt.tight_layout()
        plt.show()

    # STATISTICAL TESTS
    check_if_better_evolution_on_initial_iterations()
    check_if_better_evolution_than_random_choice()


def check_if_better_evolution_than_random_choice():
    print("check_if_better_evolution_than_random_choice")
    random_learner_csv = pandas.read_csv("experiment1-integrals-random_choice.csv")
    baseline = random_learner_csv["100"]
    if not my_shapiro(baseline):
        colored_text = colored("Failed to perform test", 'yellow')
        print(colored_text)
    else:
        for file in os.listdir():
            parts = file.split("-")
            if parts[1] == "integrals":
                learner_name = parts[-1].split(".")[0]
                if learner_name == "random_choice":
                    continue
                csv = pandas.read_csv(file)
                colored_text = colored("Testing if first learning was better than random choice for learner: %s" % learner_name, 'blue')
                print(colored_text)
                if not my_shapiro(csv["100"]):
                    colored_text = colored("Failed to perform test", 'yellow')
                    print(colored_text)
                else:
                    if t_test(csv["100"], baseline, alternative='greater', equal_var=my_bartlett(csv["100"], baseline)):
                        print("H0 is retained")
                    else:
                        print("H0 is rejected")


def check_if_better_evolution_on_initial_iterations():
    print("check_if_better_evolution_on_initial_iterations")
    for file in os.listdir():
        parts = file.split("-")
        if parts[1] == "integrals":
            learner_name = parts[-1].split(".")[0]
            csv = pandas.read_csv(file)
            colored_text = colored("Testing if first 25%% iterations present better progress than last 25%% iterations for learner: %s" % learner_name, 'blue')
            print(colored_text)
            if my_shapiro(csv["First 25"]) and my_shapiro(csv["Last 25"]):
                if t_test(csv["First 25"], csv["Last 25"], alternative='greater', equal_var=my_bartlett(csv["First 25"], csv["Last 25"])):
                    print("H0 is retained")
                else:
                    print("H0 is rejected")
            else:
                if t_test(csv["First 25"], csv["Last 25"], alternative='greater', equal_var=my_bartlett(csv["First 25"], csv["Last 25"])):
                    print("H0 is retained")
                else:
                    print("H0 is rejected")
                colored_text = colored("Failed to perform test", 'yellow')
                print(colored_text)

            colored_text = colored("Testing if first 50%% iterations present better progress than last 50%% iterations for learner: %s" % learner_name, 'blue')
            print(colored_text)
            if my_shapiro(csv["First 50"]) and my_shapiro(csv["Last 50"]):
                if t_test(csv["First 50"], csv["Last 50"], alternative='greater', equal_var=my_bartlett(csv["First 50"], csv["Last 50"])):
                    print("H0 is retained")
                else:
                    print("H0 is rejected")
            else:
                colored_text = colored("Failed to perform test", 'yellow')
                print(colored_text)


def t_test(x, y, alternative='both-sided', equal_var=True):
    statistic, double_p = ttest_ind(x, y, equal_var=equal_var)
    if equal_var:
        sd = np.sqrt((len(x) - 1) * np.var(x) + (len(y) - 1) * np.var(y))
        effect_size = abs((np.mean(y) - np.mean(x)) / sd)
    else:
        effect_size = abs((np.mean(y) - np.mean(x)) / np.std(y))

    if alternative == 'both-sided':
        pval = double_p
    elif alternative == 'greater':
        if np.mean(x) > np.mean(y):
            pval = double_p / 2.
        else:
            pval = 1.0 - double_p / 2.
    elif alternative == 'less':
        if np.mean(x) < np.mean(y):
            pval = double_p / 2.
        else:
            pval = 1.0 - double_p / 2.
    else:
        pval = None
    if pval > 0.05:
        colored_text = colored("T test statistic: %f\tpvalue: %f\teffect size: %f" % (statistic, pval, effect_size), 'red')
        print(colored_text)
        return False
    else:
        colored_text = colored("T test statistic: %f\tpvalue: %f\teffect size: %f" % (statistic, pval, effect_size), 'green')
        print(colored_text)
        return True


def my_bartlett(array1, array2):
    statistic, pvalue = bartlett(array1, array2)
    if pvalue > 0.01:
        print("Bartlett test statistic: %f\tpvalue: %f\tEqual variance: True" % (statistic, pvalue))
        return True
    else:
        colored_text = colored("Bartlett test statistic: %f\tpvalue: %f\tEqual variance: False" % (statistic, pvalue), 'red')
        print(colored_text)
        return False


def my_shapiro(array):
    statistic, pvalue = shapiro(array)
    if pvalue > 0.01:
        print("Shapiro test statistic: %f\tpvalue: %f\tIs normal: True" % (statistic, pvalue))
        return True
    else:
        colored_text = colored("Shapiro test statistic: %f\tpvalue: %f\tIs normal: False" % (statistic, pvalue), 'red')
        print(colored_text)
        return False


def two_way_anova(input_variable, output_variable):
    # interaction plot
    data = {
        "learner": [],
        input_variable: [],
        output_variable: []
    }
    for file in os.listdir():
        parts = file.split("-")
        if parts[1] == "integrals":
            csv = pandas.read_csv(file)
            learner_name = insert_newlines(parts[-1].split(".")[0], every=16)
            for stat in ["First 25", "First 50", "Last 25", "Last 50"]:  # ["First 10", "First 25", "First 50", "100", "Last 10", "Last 25", "Last 50"]
                length = len(csv[stat])
                data["learner"].extend([learner_name] * length)
                data[input_variable].extend([stat] * length)
                data[output_variable].extend(csv[stat])
                statistic, pvalue = shapiro(csv[stat])
                if pvalue > 0.01:
                    print("Distribution of %s for %s with %s = %s:\n\tShapiro test statistic: %f\n\tpvalue: %f\n\tIs normal: True" % (stat, output_variable, input_variable, stat, statistic, pvalue))
                else:
                    colored_text = colored("Distribution of %s for %s with %s = %s:\n\tShapiro test statistic: %f\n\tpvalue: %f\n\tIs normal: False" % (stat, output_variable, input_variable, stat, statistic, pvalue), 'red')
                    print(colored_text)

    data = pandas.DataFrame(data)
    fig = interaction_plot(data["learner"], data[input_variable], data[output_variable])
    plt.xticks(rotation=90)
    plt.title("Interaction of %s and algorithms on %s" % (input_variable, output_variable))
    plt.tight_layout()
    fig.show()

    formula = '%s ~ C(learner) + C(%s) + C(learner):C(%s)' % (output_variable, input_variable, input_variable)
    model = ols(formula, data).fit()
    aov_table = anova_lm(model, typ=2)
    omega_squared(aov_table)
    eta_squared(aov_table)

    pandas.set_option('display.max_columns', None)
    pandas.set_option('display.width', 200)
    print(aov_table)

    if aov_table["PR(>F)"][0] > 0.05:
        print("\nNo difference in means due to algorithm")
    else:
        print("\nDifference in means due to algorithm")
    if aov_table["PR(>F)"][1] > 0.05:
        print("No difference in means due to %s" % input_variable)
    else:
        print("Difference in means due to %s" % input_variable)
    if aov_table["PR(>F)"][2] > 0.05:
        print("No algorithm and %s interaction" % input_variable)
    else:
        print("Significant algorithm and %s interaction" % input_variable)


def omega_squared(aov):
    mse = aov['sum_sq'][-1] / aov['df'][-1]
    aov['omega_sq'] = 'NaN'
    aov['omega_sq'] = (aov[:-1]['sum_sq'] - (aov[:-1]['df'] * mse)) / (sum(aov['sum_sq']) + mse)
    return aov


def eta_squared(aov):
    aov['eta_sq'] = 'NaN'
    aov['eta_sq'] = aov[:-1]['sum_sq'] / sum(aov['sum_sq'])
    return aov


def insert_newlines(string, every=64):
    return '\n'.join(string[i:i + every] for i in range(0, len(string), every))


if __name__ == "__main__":
    x_train, y_train, x_test, y_test, learners = setup()
    experiment(x_train, y_train, x_test, y_test, learners)
    os.chdir("..")
