import math
import pandas
import matplotlib.pyplot as plt
import numpy as np


def main():
    info = pandas.read_csv("iteration_stats.csv")
    info.dropna(inplace=True)
    plt.figure(0)
    for i in range(1, len(info.columns)):
        plt.plot('iteration', info.columns[i], data=info)
    plt.legend()
    plt.show()

    plt.figure(1)
    for i in range(1, len(info.columns)):
        p = np.polyfit(info['iteration'], info[info.columns[i]], 2)
        x = info['iteration']
        y = np.polyval(p, x)
        plt.plot(x, y, label=info.columns[i])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
