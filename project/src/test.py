from __future__ import division, print_function

<<<<<<< HEAD
a=[[1,2,3,4],[2,3]]
for x in a:
    if len(x)<3:
        x.append(10)
=======
>>>>>>> ecbf0fb9ee1919af322353083004c4f88600dfb0

import matplotlib.pyplot as plt
import numpy as np
from demos import cmd
from sklearn import svm
from random import random,seed
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


def scatter_points2(s):
    min=0
    max=6
    seed(s)
    num=100

    x_p = []
    x_n = []
    x_na = []
    y_na = []
    y_p = []
    y_n = []
    can = []

    for i in xrange(num):
        can.append([random()*max,random()*max])
        if sum(can[i])>2:
            x_n.append(can[i][0])
            y_n.append(can[i][1])
        else:
            x_p.append(can[i][0])
            y_p.append(can[i][1])


    plt.figure(1)
    plt.scatter(x_p,y_p,marker='+', s=250, edgecolors='blue', linewidths=3)
    plt.scatter(x_n,y_n,marker='x', s=250, edgecolors='red', linewidths=3)
    plt.plot([0,2],[2,0],color='black', linewidth=3)

    plt.xlim([min,max])
    plt.ylim([min,max])

    plt.savefig("../figure/train.png")

    pos=len(x_p)
    a=[sum(c) for c in can]
    for id in np.argsort(a)[::-1][:pos]:
        x_na.append(can[id][0])
        y_na.append(can[id][1])

    plt.figure(2)
    plt.scatter(x_p,y_p,marker='+', s=250, edgecolors='blue', linewidths=3)
    plt.scatter(x_na,y_na,marker='x', s=250, edgecolors='red', linewidths=3)
    plt.plot([0,6],[6,0],color='black', linewidth=3)

    plt.xlim([min,max])
    plt.ylim([min,max])

    plt.savefig("../figure/train_a.png")

def scatter_points3(s):
    min=0
    max=6
    seed(s)
    num=100

    x_p = []
    x_n = []
    y_p = []
    y_n = []

    for i in xrange(num):
        can=[random()*max,random()*max]
        x_n.append(can[0])
        y_n.append(can[1])


    plt.figure(1)
    plt.scatter(x_n,y_n,marker='.', s=250, color='green', linewidths=3)
    plt.plot([0,2],[2,0],color='black', linewidth=3)

    plt.xlim([min,max])
    plt.ylim([min,max])

    plt.savefig("../figure/test.png")

    plt.figure(2)
    plt.scatter(x_n,y_n,marker='.', s=250, color='green', linewidths=3)
    plt.plot([0,6],[6,0],color='black', linewidth=3)

    plt.xlim([min,max])
    plt.ylim([min,max])

    plt.savefig("../figure/test_a.png")

def scatter_pointsR(s):
    min=0
    max=6
    seed(s)
    num=100

    can=[]
    y=[]

    x_p = []
    x_n = []
    y_p = []
    y_n = []

    for i in xrange(num):
        tmp=[random()*max,random()*max]
        can.append(tmp)
        if random()<np.exp(-0.8*sum(tmp)+1):
            y.append(1)
            x_p.append(tmp[0])
            y_p.append(tmp[1])
        else:
            y.append(0)
            x_n.append(tmp[0])
            y_n.append(tmp[1])



    clf=svm.SVC(kernel='linear', probability=True)

    clf.fit(can, y)
    w=clf.coef_[0]
    b=clf.intercept_[0]
    p1=-b/w[0]
    p2=-b/w[1]

    poses = np.where(np.array(y) == 1)[0]
    negs = np.where(np.array(y) == 0)[0]
    train_dist = clf.decision_function(np.array(can)[negs])
    negs_sel = np.argsort(np.abs(train_dist))[::-1][:len(poses)]
    sample = poses.tolist() + negs[negs_sel].tolist()
    clf.fit(np.array(can)[sample],np.array(y)[sample])

    ww=clf.coef_[0]
    bb=clf.intercept_
    pp1=-bb/ww[0]
    pp2=-bb/ww[1]

    x_nn=[x_n[i] for i in negs_sel]
    y_nn=[y_n[i] for i in negs_sel]

    plt.figure(1)
    plt.scatter(x_p,y_p,marker='+', s=250, edgecolors='blue', linewidths=3)
    plt.scatter(x_n,y_n,marker='x', s=250, edgecolors='red', linewidths=3)
    plt.plot([0,p1],[p2,0],color='black', linewidth=3)
    plt.xlim([min,max])
    plt.ylim([min,max])

    plt.savefig("../figure/train.png")

    plt.figure(2)
    plt.scatter(x_p,y_p,marker='+', s=250, edgecolors='blue', linewidths=3)
    plt.scatter(x_nn,y_nn,marker='x', s=250, edgecolors='red', linewidths=3)
    plt.plot([0,pp1],[pp2,0],color='black', linewidth=3)

    plt.xlim([min,max])
    plt.ylim([min,max])

    plt.savefig("../figure/train_a.png")


    x_nt=[]
    y_nt=[]
    for i in xrange(num):
        tmp=[random()*max,random()*max]
        x_nt.append(tmp[0])
        y_nt.append(tmp[1])


    plt.figure(3)
    plt.scatter(x_nt,y_nt,marker='.', s=250, color='green', linewidths=3)
    plt.plot([0,p1],[p2,0],color='black', linewidth=3)

    plt.xlim([min,max])
    plt.ylim([min,max])

    plt.savefig("../figure/test.png")

    plt.figure(4)
    plt.scatter(x_nt,y_nt,marker='.', s=250, color='green', linewidths=3)
    plt.plot([0,pp1],[pp2,0],color='black', linewidth=3)

    plt.xlim([min,max])
    plt.ylim([min,max])

    plt.savefig("../figure/test_a.png")



if __name__ == "__main__":
    eval(cmd())