from __future__ import print_function
from __future__ import absolute_import, division
from pdb import set_trace
from models import MLP,SVM
from demos import cmd
import numpy as np

def loadData():
    with open("../../data/biodeg.csv",'r') as f:
        data=[]
        label=[]
        for line in f.readlines():
            data.append(map(float,line.strip().split(";")[:-1]))
            label.append(line.strip().split(";")[-1])
    data=np.array(data)
    label=np.array(label)
    return data,label

def exp_folds(model):
    if model=="SVM":
        model=SVM
    elif model=="MLP":
        model=MLP
    data,label=loadData()
    folds=[10,25,100]
    best={}
    F1={}
    mlp=model()
    for fold in folds:
        best[fold]={}
        F1[fold]={}

        repeats=10
        result=[]
        for i in xrange(repeats):
            tmp=mlp.crossval(data,label,fold=fold)
            result.append(tmp)
        best[fold]=result
    set_trace()



def exp_mlp():
    data, label = loadData()
    folds = 100
    activations = ['identity', 'logistic', 'tanh', 'relu']
    max_iters = [200, 500, 1000, 2000]
    hidden_layer = [(5, 2), (10, 4), (10), (20), (30), (40), (80)]

    Fbest=0
    for act in activations:
        for epoch in max_iters:
            for hid in hidden_layer:
                mlp=MLP(activation=act,max_iter=epoch, hidden_layer_sizes=hid)
                F1=mlp.crossval(data,label,folds)
                if F1>Fbest:
                    Fbest=F1
                    best=(act,epoch,hid)
    set_trace()

def exp_svm():
    data, label = loadData()
    folds = 100
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    max_iters = [200, 500, 1000, 2000, -1]
    Cs = [1, 0.5, 0.2, 0.1, 2, 5, 10]

    Fbest = 0
    for kernel in kernels:
        for epoch in max_iters:
            for C in Cs:
                svm = SVM(kernel=kernel, max_iter=epoch, C=C)
                F1 = svm.crossval(data, label, folds)
                if F1 > Fbest:
                    Fbest = F1
                    best = (kernel, epoch, C)
    set_trace()





if __name__ == "__main__":
    eval(cmd())