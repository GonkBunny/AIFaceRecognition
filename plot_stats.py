import math
import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error


def choose_best_p(x, y):
    p = np.polyfit(x, y, 0)
    py = np.polyval(p, x)
    best = (0, mean_squared_error(y, py))
    for i in range(1, 4):
        p = np.polyfit(x, y, i)
        py = np.polyval(p, x)
        test = (i, mean_squared_error(y, py))
        best = test if test[1] < best[1] else best
    p = np.polyfit(x, y, best[0])
    # print("best: %d" % best[0])
    return p


def plot_evolution(info):
    plt.figure()
    plt.title("Algorithms score at each iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Score [0; 1]")
    for i in range(1, len(info.columns)):
        plt.plot(info.columns[0], info.columns[i], data=info)
    plt.legend(fontsize='xx-small')
    plt.tight_layout()
    plt.show()


def plot_evolution_trend(info):
    plt.figure()
    plt.title("Algorithms score at each iteration approximated by a trend line")
    plt.xlabel("Iteration")
    plt.ylabel("Score [0; 1]")
    for i in range(1, len(info.columns)):
        p = choose_best_p(info[info.columns[0]], info[info.columns[i]])
        x = info[info.columns[0]]
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
        x = info[info.columns[0]]
        p = choose_best_p(x, info[info.columns[i]])
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
    best_iterations = {k: (np.argmax(info[k]), np.amax(info[k])) for k in info.columns if k != "iteration" and k != info.columns[0]}
    for k, v in best_iterations.items():
        plt.scatter(v[0], v[1], label=k)
    plt.legend(fontsize='xx-small')
    plt.tight_layout()
    plt.show()


def main():
    al = pandas.read_csv("active_learning_iterations.csv")
    al.dropna(inplace=True)
    plot_evolution(al)
    plot_evolution_trend(al)
    plot_evolution_speed_trend(al)
    plot_best_iteration(al)

    pl = pandas.read_csv("passive_learning_iterations.csv")
    pl.dropna(inplace=True)
    plot_evolution(pl)
    plot_evolution_trend(pl)
    plot_evolution_speed_trend(pl)
    plot_best_iteration(pl)

    diff = {k: al[k] - pl[k] if k != "iteration" and k != al.columns[0] else al[k] for k in al.columns}
    diff = pandas.DataFrame(diff)
    diff.dropna(inplace=True)
    plot_evolution(diff)
    plot_evolution_trend(diff)
    plot_evolution_speed_trend(diff)
    plot_best_iteration(diff)


if __name__ == "__main__":
    main()
