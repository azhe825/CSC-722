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



class Model(object):
    def any(self):
        while True:
            for i in range(0,self.decnum):
                self.dec[i]=uniform(self.bottom[i],self.top[i])
            if self.check(): break
        return self

    def __init__(self):
        self.bottom=[0]
        self.top=[0]
        self.decnum=0
        self.objnum=0
        self.dec=[]
        self.lastdec=[]
        self.obj=[]
        self.any()

    def eval(self):
        return sum(self.getobj())

    def copy(self,other):
        self.dec=other.dec[:]
        self.lastdec=other.lastdec[:]
        self.obj=other.obj[:]
        self.bottom=other.bottom[:]
        self.top=other.top[:]
        self.decnum=other.decnum
        self.objnum=other.objnum

    def getobj(self):
        return []

    def getdec(self):
        return self.dec

    def check(self):
        for i in range(0,self.decnum):
            if self.dec[i]<self.bottom[i] or self.dec[i]>self.top[i]:
                return False
        return True




"Models:"
class DTLZ1(Model):
    def __init__(self,n=10,m=2):
        self.bottom=[0]*n
        self.top=[1]*n
        self.decnum=n
        self.objnum=m
        self.dec=[0]*n
        self.lastdec=[]
        self.obj=[]
        self.any()

    def getobj(self):
        if self.dec==self.lastdec:
            return self.obj
        f=[]
        g=self.decnum-self.objnum+1
        for x in self.dec[self.objnum-1:]:
            g=g+np.square(x-0.5)-np.cos((x-0.5)*20*np.pi)
        g=g*100
        for i in xrange(self.objnum):
            tmp=0.5*(1+g)
            for x in self.dec[:self.objnum-1-i]:
                tmp=tmp*x
            if not i==0:
                tmp=tmp*(1-self.dec[self.objnum-i])
            f.append(tmp)
        self.lastdec=self.dec
        self.obj=f
        return f

class DTLZ3(Model):
    def __init__(self,n=10,m=2):
        self.bottom=[0]*n
        self.top=[1]*n
        self.decnum=n
        self.objnum=m
        self.dec=[0]*n
        self.lastdec=[]
        self.obj=[]
        self.any()

    def getobj(self):

        if self.dec==self.lastdec:
            return self.obj

        f=[]
        g=self.decnum-self.objnum+1
        for x in self.dec[self.objnum-1:]:
            g=g+np.square(x-0.5)-np.cos((x-0.5)*20*np.pi)
        g=g*100
        for i in xrange(self.objnum):
            tmp=1+g
            for x in self.dec[:self.objnum-1-i]:
                tmp=tmp*np.cos(x*np.pi/2)
            if not i==0:
                tmp=tmp*np.sin(self.dec[self.objnum-i]*np.pi/2)
            f.append(tmp)
        self.lastdec=self.dec
        self.obj=f
        return f

class DTLZ5(Model):
    def __init__(self,n=10,m=2):
        self.bottom=[0]*n
        self.top=[1]*n
        self.decnum=n
        self.objnum=m
        self.dec=[0]*n
        self.lastdec=[]
        self.obj=[]
        self.any()

    def getobj(self):

        if self.dec==self.lastdec:
            return self.obj

        f=[]
        g=0
        for x in self.dec[self.objnum-1:]:
            g=g+np.square(x-0.5)
        theta=[np.pi*self.dec[0]/2]
        for x in self.dec[1:self.objnum-1]:
            theta.append((1+2*g*x)*np.pi/(4*(1+g)))
        for i in xrange(self.objnum):
            tmp=1+g
            for x in theta[:self.objnum-1-i]:
                tmp=tmp*np.cos(x*np.pi/2)
            if not i==0:
                tmp=tmp*np.sin(theta[self.objnum-i-1]*np.pi/2)
            f.append(tmp)
        self.lastdec=self.dec
        self.obj=f
        return f

class DTLZ7(Model):
    def __init__(self,n=10,m=2):
        self.bottom=[0]*n
        self.top=[1]*n
        self.decnum=n
        self.objnum=m
        self.dec=[0]*n
        self.lastdec=[]
        self.obj=[]
        self.any()

    def getobj(self):
        if self.dec==self.lastdec:
            return self.obj
        f=[]
        g=1+9/(self.decnum-self.objnum+1)*np.sum(self.dec[self.objnum-1:])
        h=self.objnum
        for i in xrange(self.objnum-1):
            f.append(self.dec[i])
            h=h-f[i]/(1+g)*(1+np.sin(3*np.pi*f[i]))
        f.append((1+g)*h)
        self.lastdec=self.dec
        self.obj=f
        return f

class Tunee(Model):
    def __init__(self,**kwargs):
        self.decname=[]
        self.decnum=0
        self.objnum=1
        self.bottom=[]
        self.top=[]
        self.kw=kwargs
        self.dec=[]
        for key in kwargs:
            if type(kwargs[key])==type([]):
                if len(kwargs[key])==3:
                    self.decname.append(key)
                    self.decnum=self.decnum+1
                    self.bottom.append(kwargs[key][1])
                    self.top.append(kwargs[key][2])
                    self.dec.append(kwargs[key][0])

        self.lastdec=[]
        self.obj=[]

    def getobj(self):
        if self.dec==self.lastdec:
            return self.obj
        for i,key in enumerate(self.decname):
            self.kw[key]=self.dec[i]
        f=optimize(**self.kw)
        self.lastdec=self.dec
        self.obj=[f]
        return [f]

    def copy(self,other):
        self.decname=other.decname[:]
        self.kw.update(other.kw)
        self.dec=other.dec[:]
        self.lastdec=other.lastdec[:]
        self.obj=other.obj[:]
        self.bottom=other.bottom[:]
        self.top=other.top[:]
        self.decnum=other.decnum
        self.objnum=other.objnum

def optimize(**kwargs):
    min=kwargs['min']
    max=kwargs['max']
    optimizer=kwargs['optimizer']
    del kwargs['min']
    del kwargs['max']
    del kwargs['optimizer']
    return hve(optimizer(**kwargs),min,max,100000)




class Cross_tunee(Model):

    def loadData(self):
        with open("../../data/biodeg.csv",'r') as f:
            data=[]
            label=[]
            for line in f.readlines():
                data.append(map(float,line.strip().split(";")[:-1]))
                label.append(line.strip().split(";")[-1])
        self.data=np.array(data)
        self.label=np.array(label)

    def split_cross(self, num, fold, i):
        x = range(num)
        size = int(num / fold)
        test = x[i * size:i * size + size]
        train = list(set(x) - set(test))
        train = np.array(train)
        test = np.array(test)
        return train, test

    def crossval(self, fold=5):
        num = len(self.label)
        tmp = range(num)
        shuffle(tmp)
        label = self.label[tmp]
        data = self.data[tmp]
        result=[]
        for i in xrange(fold):
            train, test = self.split_cross(num, fold, i)
            self.clf.fit(data[train], label[train])
            prediction = self.clf.predict(data[test])
            abcd = ABCD(before=label[test], after=prediction)
            F1 = [k.stats()['F1'] for k in abcd()][0]
            result.append(F1)
        return sum(result)/fold

class MLP_tunee(Cross_tunee):

    def any(self):
        while True:
            for i in range(0,self.decnum):
                self.dec[i]=int(uniform(self.bottom[i],self.top[i]))
            if self.check(): break
        return self

    def __init__(self):
        self.bottom=[0,200,10]
        self.top=[4,2000,80]
        self.decnum=3
        self.objnum=1
        self.dec=[0,0,0]
        self.lastdec=[]
        self.obj=[0]
        self.label=[]
        self.activation=['identity', 'logistic', 'tanh', 'relu']
        self.any()

    def getobj(self):
        self.dec=map(int,self.dec)
        if self.dec==self.lastdec:
            return self.obj

        f=[]
        self.clf=MLPClassifier(activation=self.activation[self.dec[0]],max_iter=self.dec[1],hidden_layer_sizes=(self.dec[2]))
        if len(self.label)==0:
            self.loadData()
        f.append(self.crossval())
        self.lastdec=self.dec
        self.obj=f
        return f


class SVM_tunee(Cross_tunee):

    def any(self):
        while True:
            for i in range(0,self.decnum):
                self.dec[i]=int(uniform(self.bottom[i],self.top[i]))
            if self.check(): break
        return self

    def __init__(self):
        self.bottom=[0,200,-3]
        self.top=[4,2000,3]
        self.decnum=3
        self.objnum=1
        self.dec=[0,0,0]
        self.lastdec=[]
        self.obj=[0]
        self.label=[]
        self.kernels=['linear', 'poly', 'rbf', 'sigmoid']
        self.any()

    def getobj(self):
        self.dec=map(int,self.dec)
        if self.dec==self.lastdec:
            return self.obj
        f=[]
        self.clf=svm.SVC(kernel=self.kernels[self.dec[0]],max_iter=self.dec[1],C=2**self.dec[2])
        if len(self.label)==0:
            self.loadData()
        f.append(self.crossval())
        self.lastdec=self.dec
        self.obj=f
        return f


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

