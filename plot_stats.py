import math
import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error


def choose_best_p(x, y):
    p = np.polyfit(x, y, 0)
    py = np.polyval(p, x)
    best = (0, mean_squared_error(y, py))
    for i in range(1, 10):
        p = np.polyfit(x, y, i)
        py = np.polyval(p, x)
        test = (i, mean_squared_error(y, py))
        best = test if test[1] < best[1] else best
    p = np.polyfit(x, y, best[0])
    print("best: %d" % best[0])
    return p


def plot_evolution(info):
    plt.figure()
    plt.title("Algorithms score at each iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Score [0; 1]")
    for i in range(1, len(info.columns)):
        plt.plot('iteration', info.columns[i], data=info)
    plt.legend(fontsize='xx-small')
    plt.tight_layout()
    plt.show()


def plot_evolution_trend(info):
    plt.figure()
    plt.title("Algorithms score at each iteration approximated by a trend line")
    plt.xlabel("Iteration")
    plt.ylabel("Score [0; 1]")
    for i in range(1, len(info.columns)):
        p = choose_best_p(info['iteration'], info[info.columns[i]])
        x = info['iteration']
        y = np.polyval(p, x)
        plt.plot(x, y, label=info.columns[i])
    plt.legend(fontsize='xx-small')
    plt.tight_layout()
    plt.show()


def plot_evolution_speed_trend(info):
    plt.figure()
    plt.title("Derivative of algorithms score at each iteration approximation")
    plt.xlabel("Iteration")
    plt.ylabel("Derivative of Score [0; 1]")
    for i in range(1, len(info.columns)):
        p = choose_best_p(info['iteration'], info[info.columns[i]])
        x = info['iteration']
        y = np.polyval(p, x)
        dy = np.diff(y)
        dx = np.diff(x)
        dydx = dy / dx
        plt.plot(x[0:len(x) - 1], dydx, label=info.columns[i])
    plt.legend(fontsize='xx-small')
    plt.tight_layout()
    plt.show()


def plot_best_iteration(info):
    plt.figure()
    plt.title("Algorithms score at their best iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Score [0; 1]")
    best_iterations = {k: (np.argmax(info[k]), np.amax(info[k])) for k in info.columns if k != "iteration"}
    for k, v in best_iterations.items():
        plt.scatter(v[0], v[1], label=k)
    plt.legend(fontsize='xx-small')
    plt.tight_layout()
    plt.show()


def main():
    info = pandas.read_csv("iteration_stats.csv")
    info.dropna(inplace=True)
    plot_evolution(info)
    plot_evolution_trend(info)
    plot_evolution_speed_trend(info)
    plot_best_iteration(info)


if __name__ == "__main__":
    main()
