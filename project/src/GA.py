from __future__ import print_function
from __future__ import absolute_import, division
from random import uniform,randint,random,seed
from time import time
import numpy as np
from pdb import set_trace
import pickle
from sklearn import svm
from demos import cmd


from HVE import Hve


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

"DE, maximization"
def differential_evolution(model,**kwargs):

    def mutate(candidates,f,cr,xbest):
        for i in xrange(len(candidates)):
            tmp=range(len(candidates))
            tmp.remove(i)
            while True:
                abc=np.random.choice(tmp,3)
                a3=[candidates[tt] for tt in abc]
                xold=candidates[i]
                r=randint(0,xold.decnum-1)
                xnew=model(**kwargs)
                xnew.any()
                for j in xrange(xold.decnum):
                    if random()<cr or j==r:
                        xnew.dec[j]=a3[0].dec[j]+f*(a3[1].dec[j]-a3[2].dec[j])
                    else:
                        xnew.dec[j]=xold.dec[j]
                if xnew.check(): break
            if xnew.eval()>xbest.eval():
                xbest.copy(xnew)
                print("!",end="")
            elif xnew.eval()>xold.eval():
                print("+",end="")
            else:
                xnew=xold
                print(".",end="")
            yield xnew


    nb=10
    maxtries=10
    f=0.75
    cr=0.3
    xbest=model(**kwargs)
    candidates=[xbest]
    for i in range(1,nb):
        x=model(**kwargs)
        x.any()
        candidates.append(x)
        if x.eval()>xbest.eval():
            xbest.copy(x)
    for tries in range(maxtries):
        print(", Retries: %2d, : Best solution: %s, " %(tries,xbest.dec),end="")
        candidates=[xnew for xnew in mutate(candidates,f,cr,xbest)]
        print("")
    print("Best solution: %s, " %xbest.dec,"hv: %s, " %xbest.getobj(),
          "evals: %s * %s" %(nb,maxtries))
    try:
        return xbest.kw
    except:
        return xbest.dec

"""
def better_gen(pf_new,pf,rate):
    count=0
    for a in pf:
        for b in pf_new:
            if is_bd(b,a):
                count=count+1
    whole=len(pf)*len(pf_new)
    if count>-int(whole*rate):
        return True
    else:
        return False
"""

"is a binary dominate b? smaller is better"
def is_bd(a,b):
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

def crossover(a,b,baby):
    life=len(a.dec)
    while True:
        x=randint(1,len(a.dec)-1)
        baby.dec=list(np.array(a.dec)[:x])+list(np.array(b.dec)[x:])
        if baby.check():
            return baby
        life=life-1
        if life<0:
            return baby.any()

def mutate(baby):
    return baby.any()

"Update pf_best"
def compete(pf_best,pf_new):
    tmp=[]
    for a in pf_new:
        for b in pf_best:
            if is_bd(a,b):
                tmp.append(a)
                pf_best.remove(b)
    if tmp:
        pf_best.extend(tmp)
        return True
    else:
        return False

""




def update_pf(pf,can):
    if type(can)==type([]):
        flag=False
        for x in can:
            if update_pf(pf,x):
                flag=True
        return flag
    else:
        if not pf:
            pf.append(can)
        else:
            for elem in pf:
                if is_bd(elem,can):
                    return False
                if is_bd(can,elem):
                    pf.remove(elem)
            pf.append(can)
        return True

def train_learner(can,pf,kernel='rbf'):
    x=[]
    y=[]
    for data in can:
        x.append(data.getdec())
        if data in pf:
            y.append("pos")
        else:
            y.append("neg")
    learner=svm.SVC(kernel=kernel,probability=True)
    learner.fit(x,y)
    return learner


def GeneticAlgorithm_active(hve,Model=DTLZ1,decnum=10,objnum=2,the_seed=-1,candidates=100,generations=1000,mutation_rate=0.05,lifes=5,kernal="linear"):
    if the_seed>=0:
        seed(the_seed)
    candidates=int(candidates)
    lifes=int(lifes)
    can=[Model(decnum,objnum) for _ in xrange(candidates*10)]

    pf=[]
    update_pf(pf,can)
    learner=train_learner(can,pf,kernel=kernal)

    hv=[]
    hv.append(hve.hve(pf))

    print("Generation: %d. Hypervolume: %f" %(0,hv[0]))

    life=0
    for i in xrange(generations):
        can_new=[]

        for aa in pf:
            for bb in pf:
                if bb==aa:
                    continue
                baby=Model(decnum,objnum)
                crossover(aa,bb,baby)
                can_new.append(baby)
        for j in xrange(len(can_new)):
            baby=Model(decnum,objnum)
            can_new.append(baby.any())

        data=[x.getdec() for x in can_new]

        pos_at = list(learner.classes_).index("pos")
        proba=learner.predict_proba(data)
        proba = proba[:, pos_at]
        order = np.argsort(proba)[::-1][:candidates]
        can_kept = [can_new[x] for x in order]
        can.extend(can_kept)
        learner=train_learner(can,pf,kernel=kernal)

        change=update_pf(pf,can_kept)
        if change:
            life=0
        else:
            life=life+1
        if life==lifes:
            break
        #print("Frontier num: "+str(len(pf_best)))
        hv.append(hve.hve(pf))
        print("Generation: %d. Hypervolume: %f" %(i+1,hv[i+1]))
    return hv

def GeneticAlgorithm(hve,Model=DTLZ1,decnum=10,objnum=2,the_seed=-1,candidates=100,generations=1000,mutation_rate=0.05,lifes=5):
    if the_seed>=0:
        seed(the_seed)
    candidates=int(candidates)
    lifes=int(lifes)
    can=[Model(decnum,objnum) for _ in xrange(candidates*10)]

    pf=[]
    update_pf(pf,can)
    life=0

    hv=[]
    hv.append(hve.hve(pf))

    print("Generation: %d. Hypervolume: %f" %(0,hv[0]))

    for i in xrange(generations):
        can_new=[]
        for j in xrange(candidates):
            baby=Model(decnum,objnum)
            pick=np.random.choice(len(pf),2,replace=False)
            crossover(pf[pick[0]],pf[pick[1]],baby)
            if random()<mutation_rate*(2**life):
                mutate(baby)
            can_new.append(baby)

        change=update_pf(pf,can_new)
        if change:
            life=0
        else:
            life=life+1
        if life==lifes:
            break
        #print("Frontier num: "+str(len(pf_best)))
        hv.append(hve.hve(pf))
        print("Generation: %d. Hypervolume: %f" %(i+1,hv[i+1]))

    return hv






def simple():
    candidates=100
    generations=1000
    mutation_rate=0.05
    lifes=5
    model=DTLZ7
    objs=3
    decs=30
    hv_num=300000

    hve_cal = Hve(model,decs,objs,hv_num)
    hv_1=GeneticAlgorithm_active(hve_cal,Model=model,decnum=decs,objnum=objs,candidates=candidates,generations=generations,mutation_rate=mutation_rate,lifes=lifes,kernal="linear")
    hv_2=GeneticAlgorithm(hve_cal,Model=model,decnum=decs,objnum=objs,candidates=candidates,generations=generations,mutation_rate=mutation_rate,lifes=lifes)
    with open("../dump/simple.pickle","w") as handle:
        pickle.dump(hv_1, handle)
        pickle.dump(hv_2, handle)

if __name__ == "__main__":
    eval(cmd())



