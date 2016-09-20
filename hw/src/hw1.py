from __future__ import division, print_function


import matplotlib.pyplot as plt
import numpy as np
from demos import cmd
from sklearn import svm
from random import random
from collections import Counter
from pdb import set_trace

def scatter_points():
    x_p = [2,4,-6]
    x_n = [-4,-2,8]
    y_p = [2,-8,-2]
    y_n = [8,-8,0]


    plt.figure(0)
    plt.scatter(x_p,y_p,marker='+', s=250, edgecolors='red', linewidths=3)
    plt.scatter(x_n,y_n,marker='_', s=250, edgecolors='blue', linewidths=3)

    plt.savefig("../figure/hw1_scatter0.eps")
    plt.savefig("../figure/hw1_scatter0.png")

    plt.figure(1)
    plt.scatter(x_p,y_p,marker='+', s=250, edgecolors='red', linewidths=3)
    plt.scatter(x_n,y_n,marker='_', s=250, edgecolors='blue', linewidths=3)
    plt.plot([-8,10],[9.67,-2.33],color='black')

    plt.xlim([-8,10])
    plt.ylim([-10,10])

    plt.savefig("../figure/hw1_scatter1.eps")
    plt.savefig("../figure/hw1_scatter1.png")


def linear_separable():
    x= [[2,2],[4,-8],[-6,-2],[-4,8],[-2,-8],[8,0]]
    y= [1,1,1,-1,-1,-1]
    classifier = svm.SVC(kernel='linear')

def kNN_draw(k,n=100000):
    x= np.array([[2,2],[4,-8],[-6,-2],[-4,8],[-2,-8],[8,0]])
    y= np.array([1,1,1,-1,-1,-1])
    xlim = np.array([-10,10])
    ylim = np.array([-10,10])

    xp=[]
    xn=[]
    xd=[]
    yp=[]
    yn=[]
    yd=[]

    for i in xrange(n):
        rx=[random()*(xlim[1]-xlim[0])+xlim[0], random()*(ylim[1]-ylim[0])+ylim[0]]
        labels = y[kNN(x,rx)[:k]]
        classes = Counter(labels)
        if classes[1]>classes[-1]:
            label = 1
            xp.append(rx[0])
            yp.append(rx[1])
        elif classes[1]<classes[-1]:
            label = -1
            xn.append(rx[0])
            yn.append(rx[1])
        else:
            label = 0
            xd.append(rx[0])
            yd.append(rx[1])

    plt.figure(0)
    plt.scatter(xp,yp, s=1, edgecolors='green')
    plt.scatter(xn,yn, s=1, edgecolors='brown')
    plt.scatter(xd,yd, s=1, edgecolors='gray')

    x_p = [2,4,-6]
    x_n = [-4,-2,8]
    y_p = [2,-8,-2]
    y_n = [8,-8,0]

    plt.scatter(x_p,y_p,marker='+', s=250, edgecolors='red', linewidths=3)
    plt.scatter(x_n,y_n,marker='_', s=250, edgecolors='blue', linewidths=3)

    plt.xlim([-10,10])
    plt.ylim([-10,10])

    plt.savefig("../figure/hw1_"+str(k)+"NN.eps")
    plt.savefig("../figure/hw1_"+str(k)+"NN.png")


def kNN_group(k, n=100000):
    x= np.array([[2,2],[4,-8],[-6,-2],[-4,8],[-2,-8],[8,0]])
    y= np.array([1,1,1,-1,-1,-1])
    xlim = np.array([-10,10])
    ylim = np.array([-10,10])

    group={}


    for i in xrange(n):
        rx=[random()*(xlim[1]-xlim[0])+xlim[0], random()*(ylim[1]-ylim[0])+ylim[0]]
        labels = tuple(np.sort(kNN(x,rx)[:k]))
        try:
            group[labels].append(rx)
        except:
            group[labels] = [rx]

    colors=['green',"brown",'purple','pink','yellow','orange','gray','magenta','cyan']
    plt.figure(0)
    for key in group:
        points = group[key]

        xs=[]
        ys=[]
        for point in points:
            xs.append(point[0])
            ys.append(point[1])

        plt.scatter(xs,ys, s=1, edgecolors=colors.pop())

    x_p = [2,4,-6]
    x_n = [-4,-2,8]
    y_p = [2,-8,-2]
    y_n = [8,-8,0]

    plt.scatter(x_p,y_p,marker='+', s=250, edgecolors='red', linewidths=3)
    plt.scatter(x_n,y_n,marker='_', s=250, edgecolors='blue', linewidths=3)

    plt.xlim([-10,10])
    plt.ylim([-10,10])

    plt.savefig("../figure/hw1_"+str(k)+"NN_group.eps")
    plt.savefig("../figure/hw1_"+str(k)+"NN_group.png")


    plt.figure(1)
    for key in group:
        points = group[key]

        xs=[]
        ys=[]
        for point in points:
            xs.append(point[0])
            ys.append(point[1])
        if cmp(key,tuple([0,3]))==0 or cmp(key,tuple([0,5]))==0:
            color = 'brown'
        else:
            color = 'green'
        plt.scatter(xs,ys, s=1, edgecolors=color)

    x_p = [2,4,-6]
    x_n = [-4,-2,8]
    y_p = [2,-8,-2]
    y_n = [8,-8,0]

    plt.scatter(x_p,y_p,marker='+', s=250, edgecolors='red', linewidths=3)
    plt.scatter(x_n,y_n,marker='_', s=250, edgecolors='blue', linewidths=3)

    plt.xlim([-10,10])
    plt.ylim([-10,10])

    plt.savefig("../figure/hw1_"+str(k)+"NN_final.eps")
    plt.savefig("../figure/hw1_"+str(k)+"NN_final.png")

def kNN(x,r):
    dis = [np.linalg.norm(xx-r) for xx in x]
    order = np.argsort(dis)
    return order



def loo(k=1):
    x= np.array([[2,2],[4,-8],[-6,-2],[-4,8],[-2,-8],[8,0]])
    y= np.array([1,1,1,-1,-1,-1])
    true = 0
    for i in xrange(len(y)):
        tmp=range(len(y))
        tmp.pop(i)
        order = np.array(tmp)[kNN(x[tmp],x[i])]
        classes = Counter(y[order[:k]])
        if classes[1]>classes[-1]:
            label = 1
        elif classes[1]<classes[-1]:
            label = -1
        else:
            label = 0
        if label==0:
            true=true+1
        elif y[i]==label:
            true=true +1
    accuracy=true/len(y)
    print(accuracy)





if __name__ == "__main__":
    eval(cmd())