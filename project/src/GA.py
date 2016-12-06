from __future__ import print_function
from __future__ import absolute_import, division
from random import uniform,randint,random,seed
from time import time
import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace
import pickle
from sklearn import svm
from demos import cmd


# from HVE import Hve
from model import *





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
                xnew=model.new()
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
    xbest=model.new()
    candidates=[xbest]
    for i in range(1,nb):
        x=model.new()
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

def crossover(a,b):
    baby=a.new()
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

def train_learner(can,pf,kernel='rbf',aggressive=False):
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
    if aggressive:
        poses = np.where(np.array(y) == "pos")[0]
        negs = np.where(np.array(y) == "neg")[0]
        train_dist = learner.decision_function(np.array(x)[negs])
        negs_sel = np.argsort(np.abs(train_dist))[::-1][:len(poses)]
        sample = poses.tolist() + negs[negs_sel].tolist()

        learner.fit(np.array(x)[sample], np.array(y)[sample])
    return learner


def GeneticAlgorithm_active(hve,Model,the_seed=-1,candidates=100,init=10,generations=1000,mutation_rate=0.05,lifes=5,kernal="linear",aggressive=False):
    if the_seed>=0:
        seed(the_seed)
    candidates=int(candidates)
    lifes=int(lifes)
    can=[Model.new().any() for _ in xrange(candidates*init)]

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
                baby=crossover(aa,bb)
                can_new.append(baby)
        # for j in xrange(int(mutation_rate*(2**life)*len(can_new))):
        #     baby=Model.new().any()
        #     can_new.append(baby)

        data=[x.getdec() for x in can_new]

        pos_at = list(learner.classes_).index("pos")
        proba=learner.predict_proba(data)
        proba = proba[:, pos_at]
        order = np.argsort(proba)[::-1][:candidates]
        can_kept = [can_new[x] for x in order]
        for x in can_kept:
            if random() < mutation_rate * (2 ** life):
                x.any()
        can.extend(can_kept)
        change = update_pf(pf, can_kept)
        learner=train_learner(can,pf,kernel=kernal,aggressive=aggressive)
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


def GeneticAlgorithm(hve,Model,the_seed=-1,candidates=100,init=10,generations=1000,mutation_rate=0.05,lifes=5):
    if the_seed>=0:
        seed(the_seed)
    candidates=int(candidates)
    lifes=int(lifes)
    can=[Model.new().any() for _ in xrange(candidates*init)]

    pf=[]
    update_pf(pf,can)
    life=0

    hv=[]
    hv.append(hve.hve(pf))

    print("Generation: %d. Hypervolume: %f" %(0,hv[0]))

    for i in xrange(generations):
        can_new=[]
        for j in xrange(candidates):
            pick=np.random.choice(len(pf),2,replace=False)
            baby=crossover(pf[pick[0]],pf[pick[1]])
            if random()<mutation_rate*(2**life):
                baby.any()
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

def run_exp(setting):

    hve_cal = Hve(setting['model'], setting['hv_num'])

    hv_1 = GeneticAlgorithm(hve_cal, the_seed=setting['seed'], Model=setting['model'], candidates=setting['candidates'], init=setting['init'],
                            generations=setting['generations'], mutation_rate=setting['mutation_rate'], lifes=setting['lifes'])
    hv_2 = GeneticAlgorithm_active(hve_cal, the_seed=setting['seed'], Model=setting['model'], candidates=setting['candidates'], init=setting['init'],
                            generations=setting['generations'], mutation_rate=setting['mutation_rate'], lifes=setting['lifes'], kernal="linear",
                                   aggressive=False)
    hv_3 = GeneticAlgorithm_active(hve_cal, the_seed=setting['seed'], Model=setting['model'], candidates=setting['candidates'], init=setting['init'],
                            generations=setting['generations'], mutation_rate=setting['mutation_rate'], lifes=setting['lifes'], kernal="linear",
                                   aggressive=True)
    hv_4 = GeneticAlgorithm_active(hve_cal, the_seed=setting['seed'], Model=setting['model'], candidates=setting['candidates'], init=setting['init'],
                            generations=setting['generations'], mutation_rate=setting['mutation_rate'], lifes=setting['lifes'], kernal="rbf",
                                   aggressive=False)
    hv_5 = GeneticAlgorithm_active(hve_cal, the_seed=setting['seed'], Model=setting['model'], candidates=setting['candidates'], init=setting['init'],
                            generations=setting['generations'], mutation_rate=setting['mutation_rate'], lifes=setting['lifes'], kernal="rbf",
                                   aggressive=True)
    hvs = {"GA": hv_1, "linearN": hv_2, "linearA": hv_3, "rbfN": hv_4, "rbfA": hv_5}
    return hvs

def simple(id):
    candidates=100
    generations=20
    mutation_rate=0.1
    init=10
    lifes=5
    objs=3
    decs=10
    model=DTLZ7(n=decs,m=objs)
    hv_num=100000
    seed=int(id)
    setting = {"candidates": candidates, "generations": generations, "mutation_rate": mutation_rate, "init": init,
               "hv_num": hv_num,'seed': seed, "lifes": lifes, "model": model}

    hvs=run_exp(setting)
    with open("../dump/simple"+str(id)+".pickle","w") as handle:
        pickle.dump(hvs, handle)


def loadData(file="../../data/biodeg.csv"):
    with open(file, 'r') as f:
        data = []
        label = []
        for line in f.readlines():
            data.append(map(float, line.strip().split(";")[:-1]))
            label.append(line.strip().split(";")[-1])
    data = np.array(data)
    label = np.array(label)
    return data, label

def tune_mlp(file="../../data/biodeg.csv"):
    data,label=loadData(file)
    candidates = 10
    generations = 20
    mutation_rate = 0.05
    init=5
    lifes = 5
    model = Mlp_exp(data=data,label=label)
    hv_num = 100000
    seed = 10

    setting = {"candidates": candidates, "generations": generations, "mutation_rate": mutation_rate, "init": init,
               "hv_num": hv_num, 'seed': seed, "lifes": lifes, "model": model}


    hvs = run_exp(setting)
    with open("../dump/tune_mlp.pickle", "w") as handle:
        pickle.dump(hvs, handle)

def tune_svm(file="../../data/biodeg.csv"):
    data, label = loadData(file)
    candidates = 10
    generations = 20
    init=5
    mutation_rate = 0.05
    lifes = 5
    model = Svm_exp(data=data, label=label)
    hv_num = 100000
    seed = 10

    setting = {"candidates": candidates, "generations": generations, "mutation_rate": mutation_rate, "init": init,
               "hv_num": hv_num, 'seed': seed, "lifes": lifes, "model": model}


    hvs = run_exp(setting)
    with open("../dump/tune_svm.pickle", "w") as handle:
        pickle.dump(hv_1, handle)
        pickle.dump(hv_2, handle)


## Drawings ##

def draw_simple(id):
    font = {'family': 'cursive',
            'weight': 'bold',
            'size': 20}


    plt.rc('font', **font)
    paras = {'lines.linewidth': 4, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
             'figure.autolayout': True, 'figure.figsize': (16, 6)}
    plt.rcParams.update(paras)

    with open("../dump/simple"+str(id)+".pickle", "r") as handle:
        hvs = pickle.load(handle)

    x=np.min([len(hvs[k]) for k in hvs])
    xa=[1000+s*100 for s in xrange(x)]

    plt.figure(1)
    for key in hvs:
        plt.plot(xa, hvs[key][:x], label=key)

    plt.ylabel("Hyper Volume")
    plt.xlabel("Number of Evaluations")
    plt.legend(bbox_to_anchor=(0.9, 0.60), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../figure/simple"+str(id)+".eps")
    plt.savefig("../figure/simple"+str(id)+".png")


if __name__ == "__main__":
    eval(cmd())



