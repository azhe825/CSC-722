from __future__ import division, print_function


import matplotlib.pyplot as plt
import numpy as np
from demos import cmd
from sklearn import svm
from random import random
from collections import Counter
from pdb import set_trace
from mpl_toolkits.mplot3d import Axes3D


def scatter_points():
    x_p = [0,2]
    x_n = [1]
    y_p = [2,0]
    y_n = [1]


    plt.figure(0)
    plt.scatter(x_p,y_p,marker='o', s=250, edgecolors='blue', linewidths=3)
    plt.scatter(x_n,y_n,marker='x', s=250, edgecolors='red', linewidths=3)

    plt.savefig("../figure/1a.png")

    x_p = [0,2]
    x_n = [2]
    y_p = [2,0]
    y_n = [2]

    plt.figure(1)
    plt.scatter(x_p,y_p,marker='o', s=250, edgecolors='blue', linewidths=3)
    plt.scatter(x_n,y_n,marker='x', s=250, edgecolors='red', linewidths=3)
    plt.plot([-1,4],[4,-1],color='black')

    plt.xlim([-1,4])
    plt.ylim([-1,4])

    plt.savefig("../figure/1b.png")

    x_p = [0,2]
    x_n = [0,2]
    y_p = [2,0]
    y_n = [0,2]

    plt.figure(2)
    plt.scatter(x_p,y_p,marker='o', s=250, edgecolors='blue', linewidths=3)
    plt.scatter(x_n,y_n,marker='x', s=250, edgecolors='red', linewidths=3)

    plt.savefig("../figure/2.png")


    x_p = [0,2]
    x_n = [0,2]
    y_p = [2,0]
    y_n = [0,2]
    z_p = [0,0]
    z_n = [1,1]

    fig=plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_p,y_p,z_p, marker='o', s=250, edgecolors='blue', linewidths=3)
    ax.scatter(x_n,y_n,z_n, marker='x', s=250, edgecolors='red', linewidths=3)

    plt.savefig("../figure/3a.png")

    x_p = [0,2,2]
    x_n = [0]
    y_p = [2,0,2]
    y_n = [0]
    z_p = [0,0,0]
    z_n = [1]

    fig=plt.figure(4)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_p,y_p,z_p, marker='o', s=250, edgecolors='blue', linewidths=3)
    ax.scatter(x_n,y_n,z_n, marker='x', s=250, edgecolors='red', linewidths=3)

    plt.savefig("../figure/3b.png")

    x_p = [0,2]
    x_n = [0,2,1]
    y_p = [2,0]
    y_n = [0,2,1]
    z_p = [0,0]
    z_n = [1,1,0]

    fig=plt.figure(5)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_p,y_p,z_p, marker='o', s=250, edgecolors='blue', linewidths=3)
    ax.scatter(x_n,y_n,z_n, marker='x', s=250, edgecolors='red', linewidths=3)

    plt.savefig("../figure/4.png")


if __name__ == "__main__":
    eval(cmd())