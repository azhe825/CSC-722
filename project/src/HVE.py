from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division
from random import uniform
import numpy as np
from pdb import set_trace


"Hyper volumn estimation"
"RAISE LAB"

class Hve(object):

    def __init__(self,model,num=100000,min_max=[]):
        self.model=model
        self.decnum=model.decnum
        self.objnum=model.objnum
        self.num=num
        if len(min_max)==2:
            self.min=min_max[0]
            self.max=min_max[1]
        else:
            self.init_min_max()
        # self.init_pebble()

    def init_min_max(self):
        can=[self.model.new().any() for _ in xrange(self.num)]
        self.max=[np.max([c.getobj()[i] for c in can]) for i in range(self.objnum)]
        self.min=[np.min([c.getobj()[i] for c in can]) for i in range(self.objnum)]

    # def init_pebble(self):
    #     self.pebbles=[[uniform(self.min[k],self.max[k]) for k in xrange(self.objnum)] for i in xrange(self.num)]


    "is a binary dominate b? smaller is better"
    def is_bd(self,a,b):
        try:
            obj_a=a.getobj()
        except:
            obj_a=a
        try:
            obj_b=b.getobj()
        except:
            obj_b=b
        if obj_a==obj_b:
            return False
        for i in xrange(a.objnum):
            if obj_b[i]<obj_a[i]:
                return False
        return True

    "is the peddle inside the hyper volumn"
    def inbox(self,pebble,frontier):
        for candidate in frontier:
            if self.is_bd(candidate,pebble):
                return True
        return False


    "estimate hyper volumn of frontier"
    def hve(self,frontier):
        count=0
        for x in xrange(self.num):
            pebble = [uniform(self.min[k], self.max[k]) for k in xrange(self.objnum)]
            if self.inbox(pebble,frontier):
                count=count+1
        return count/(self.num)

