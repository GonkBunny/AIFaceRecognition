import math

import pandas
import matplotlib.pyplot as plt


def main():
    info = pandas.read_csv("iteration_stats.csv")
    info.dropna(inplace=True)
    plt.scatter('iteration', info.columns[1], data=info, marker='o')
    plt.scatter('iteration', info.columns[2], data=info, marker='D')
    plt.scatter('iteration', info.columns[3], data=info, marker='+')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
