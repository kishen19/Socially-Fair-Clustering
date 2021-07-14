from code.classes import Point, Center, Dataset
from code import datagen
from code.utilities import gen_rand_centers,plot
from code.solve import solve_clustering
from code.lloyd import lloyd
from coresets import coresets

import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import sys
import time


def get_sens(dataset,flag):
    if dataset=="credit":
        data = pd.read_csv("./data/" + dataset + "/" + dataset + flag + ".csv",index_col=0)
        svar,data,groups = data.iloc[:,0],data.iloc[:,1:],{0:"Higher Education",1:"Lower Education"}
    elif dataset=="adult":
        data = pd.read_csv("./data/" + dataset + "/" + dataset + flag + ".csv",index_col=0)
        svar,data,groups = data.iloc[:,0],data.iloc[:,1:],{0:"Female",1:"Male"}
    return svar,data,groups

def get_result(args):
    algo,dataset,name,k,num_iters,flag = args
    try:
        print(algo + ": Start: k="+str(k))
        sys.stdout.flush()
        svar,data,groups = get_sens(dataset,flag)
        if algo=="ALGO":
            solve_clustering(dataset,name,data,svar,groups,k,2,num_iters)
        else:
            lloyd(dataset,name,data,svar,groups,k,2,num_iters)
    except ValueError as e:
        print(algo + ": Failed: k="+str(k))
        print(e)
        sys.stdout.flush()
        return
    except ArithmeticError as e:
        print(algo + ": Failed: k="+str(k))
        print(e)
        sys.stdout.flush()
        return
    print(algo + ": Done: k="+str(k))
    sys.stdout.flush()

def init_dataset(dataset,name,num_inits,coreset_sizes,k,isPCA=False):
    init_centers = []
    svar,data,groups = get_sens(dataset,"N")
    for init in range(num_inits):
        mask = gen_rand_centers(data.shape[0],k)
        centers = [Center(data.iloc[mask[i],:],i) for i in range(k)]
        init_centers.append(centers)
    resultsk = Dataset(name+"_k="+str(k), data,svar,groups)

    flag = "P_k="+str(k) if isPCA else "N"
    svar,data,groups = get_sens(dataset,flag)
    n = data.shape[0]
    ell = len(groups)
    if isPCA:
        resultsk.add_PCA_data(data)
    for coreset_size in coreset_sizes:
        coreset = []
        rem = coreset_size
        _coreset_time = 0
        for ind,group in enumerate(groups):
            data_group = data[np.asarray(svar)==group]
            coreset_gen = coresets.KMeansCoreset(data_group,n_clusters=k)
            coreset_group_size = int(data_group.shape[0]*coreset_size/n) if ind<ell-1 else rem
            rem-=coreset_group_size
            _st = time.time()
            coreset_group, weights = coreset_gen.generate_coreset(coreset_group_size)
            _ed = time.time()
            _coreset_time += (_ed - _st)
            coreset += [Point(coreset_group[i],group,weights[i]) for i in range(coreset_group_size)]
        resultsk.add_coreset(k,coreset)

    for init in range(num_inits):
        for group in groups:
            for cor_num in range(len(coreset_sizes)):
                resultsk.add_new_result("ALGO"+" ("+ groups[group] + ")",k,init,cor_num,0,0,0,0,_coreset_time,init_centers[init])
    for init in range(num_inits):
        for group in groups:
            resultsk.add_new_result("Lloyd"+" ("+ groups[group] + ")",k,init,0,0,0,0,0,_coreset_time,init_centers[init])

    f = open("./results/"+dataset+"/" + name+"_k="+str(k) + "_picklefile","wb")
    pickle.dump(resultsk,f)
    f.close()

def main():
    dataset = "adult"
    datagen.dataNgen(dataset)
    isPCA = False
    if isPCA:
        for k in range(4,17,2):
            datagen.dataPgen(dataset,k)

    namesuf="_wPCA" if isPCA else "_woPCA"
    name = dataset + namesuf    

    # Generate Init_centers
    k_vals = range(4,17,2)
    algos = ['Lloyd','ALGO']
    num_inits = 10
    num_iters = 100
    coreset_sizes = [1000,2000,3000,4000,5000]
    z = 2

    for k in k_vals:
        init_dataset(dataset, name, num_inits, coreset_sizes, k, isPCA)
    
    

    for algo in algos:
        if algo == 'Lloyd':
            pool = mp.Pool(mp.cpu_count() + 4)
            jobs = []
            for k in k_vals:
                flag = "P_K="+str(k) if isPCA else "N"
                job = pool.apply_async(get_result,([algo,dataset,name,k,num_iters,flag],))
                jobs.append(job)
            
            for job in jobs:
                job.get()
            pool.close()
            pool.join()
        else:
            for iter in tqdm(range(1,num_iters+1)):
                solve_clustering(dataset,name,k_vals,z,iter)
                

if __name__=='__main__':
    main()