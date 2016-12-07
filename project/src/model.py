from __future__ import print_function
from __future__ import absolute_import, division

from random import uniform,randint,random,seed
from time import time
import numpy as np
from HVE import Hve
from models import MLP,SVM
from pdb import set_trace

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

    def new(self):
        x=self.__class__()
        x.copy(self)
        return x




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
    hve_cal = Hve(optimizer, 100000, min_max=[min, max])
    return hve_cal.hve(optimizer(**kwargs))

##
class Mlp_exp(Model):

    def any(self):
        while True:
            for i in range(0, self.decnum):
                self.dec[i] = int(uniform(self.bottom[i], self.top[i]))
            if self.check(): break
        return self

    def __init__(self, data=[], label=[], bottom=[5,0,8,5], top=[101,4,12,81]):
        self.data=data
        self.label=label
        self.decnum = 4
        self.objnum = 2
        self.bottom = bottom
        self.top = top
        self.dec = [0] * self.decnum
        self.lastdec = []
        self.obj = []
        self.any()

    def copy(self, other):
        self.dec = other.dec[:]
        self.lastdec = other.lastdec[:]
        self.obj = other.obj[:]
        self.bottom = other.bottom[:]
        self.top = other.top[:]
        self.decnum = other.decnum
        self.objnum = other.objnum
        self.data = other.data
        self.label = other.label

    def check(self):
        for i in range(0, self.decnum):
            if self.dec[i] < self.bottom[i] or self.dec[i] >= self.top[i]:
                return False
        return True

    def getobj(self):
        if self.dec == self.lastdec:
            return self.obj
        activations = ['identity', 'logistic', 'tanh', 'relu']
        mlp = MLP(activation=activations[self.dec[1]], max_iter=2**self.dec[2], hidden_layer_sizes=(self.dec[3]))
        self.lastdec = self.dec
        self.obj = mlp.crossval_pr(self.data, self.label, self.dec[0])
        return self.obj

class Svm_exp(Model):

    def any(self):
        while True:
            for i in range(0, self.decnum):
                self.dec[i] = int(uniform(self.bottom[i], self.top[i]))
            if self.check(): break
        return self

    def __init__(self, data=[], label=[], bottom=[5,0,8,-7], top=[101,4,13,8]):
        self.data=data
        self.label=label
        self.decnum = 4
        self.objnum = 2
        self.bottom = bottom
        self.top = top
        self.dec = [0] * self.decnum
        self.lastdec = []
        self.obj = []
        self.any()

    def copy(self, other):
        self.dec = other.dec[:]
        self.lastdec = other.lastdec[:]
        self.obj = other.obj[:]
        self.bottom = other.bottom[:]
        self.top = other.top[:]
        self.decnum = other.decnum
        self.objnum = other.objnum
        self.data = other.data
        self.label = other.label

    def check(self):
        for i in range(0, self.decnum):
            if self.dec[i] < self.bottom[i] or self.dec[i] >= self.top[i]:
                return False
        return True

    def getobj(self):
        if self.dec == self.lastdec:
            return self.obj
        kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        if self.dec[2]>11:
            iter=-1
        else:
            iter=2**self.dec[2]
        svm = SVM(kernel=kernels[self.dec[1]], max_iter=iter, C=2**self.dec[3])
        self.lastdec = self.dec
        self.obj = svm.crossval_pr(self.data, self.label, self.dec[0])
        return self.obj