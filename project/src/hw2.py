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

def exp_mlp():
    data,label=loadData()
    folds=[10,25,100]
    activations=['identity', 'logistic', 'tanh', 'relu']
    max_iters=[200,500,1000,2000]
    hidden_layer=[(100),(5,2)]
    best={}
    F1={}
    mlp=MLP()
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
    set_trace()





    mlp=MLP()
    svm=SVM(max_iter=-1)
    result_mlp=mlp.crossval(data,label)
    result_svm=svm.crossval(data,label)
    set_trace()

if __name__ == "__main__":
    eval(cmd())