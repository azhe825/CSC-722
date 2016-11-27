from __future__ import print_function
from __future__ import absolute_import, division
from random import uniform,randint,random,seed
from time import time
import numpy as np
from pdb import set_trace
import pickle
from sklearn import svm
from demos import cmd
from models import *

from HVE import Hve


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

    if model == "SVM_tunee":
        model=SVM_tunee
    elif model == "MLP_tunee":
        model=MLP_tunee
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
    print("Best solution: %s, " %xbest.dec,"obj: %s, " %xbest.getobj(),
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



