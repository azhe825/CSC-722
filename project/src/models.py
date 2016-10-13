from __future__ import print_function
from __future__ import absolute_import, division
from sklearn.neural_network import MLPClassifier
from random import uniform,randint,random,seed,shuffle
from time import time
import numpy as np
from pdb import set_trace
import pickle
from sklearn import svm
from ABCD import ABCD

class Cross_exp(object):

    def split_cross(self, num, fold, i):
        x = range(num)
        size = int(num / fold)
        test = x[i * size:i * size + size]
        train = list(set(x) - set(test))
        train = np.array(train)
        test = np.array(test)
        return train, test

    def crossval(self, data, label, fold=5):
        num = len(label)
        tmp = range(num)
        shuffle(tmp)
        label = label[tmp]
        data = data[tmp]
        result=[]
        for i in xrange(fold):
            train, test = self.split_cross(num, fold, i)
            self.clf.fit(data[train], label[train])
            prediction = self.clf.predict(data[test])
            abcd = ABCD(before=label[test], after=prediction)
            F1 = [k.stats()['F1'] for k in abcd()][0]
            result.append(F1)
        return sum(result)/fold

    def crossval_pr(self, data, label, fold=5):
        num = len(label)
        tmp = range(num)
        shuffle(tmp)
        label = label[tmp]
        data = data[tmp]
        result_p = []
        result_r = []
        for i in xrange(fold):
            train, test = self.split_cross(num, fold, i)
            self.clf.fit(data[train], label[train])
            prediction = self.clf.predict(data[test])
            abcd = ABCD(before=label[test], after=prediction)
            prec = [k.stats()['Prec'] for k in abcd()][0]
            rec = [k.stats()['Sen'] for k in abcd()][0]
            result_p.append(prec)
            result_r.append(rec)
        result=[sum(result_p) / fold, sum(result_r) / fold]
        return result

class MLP(Cross_exp):
    def __init__(self,activation='relu',max_iter=200, hidden_layer_sizes=(5, 2)):
        self.clf=MLPClassifier(activation=activation,max_iter=max_iter,hidden_layer_sizes=hidden_layer_sizes)

class SVM(Cross_exp):
    def __init__(self, kernel='linear', max_iter=200, C=1):
        self.clf = svm.SVC(kernel=kernel, max_iter=max_iter, C=C)

