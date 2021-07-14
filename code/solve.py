import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import pickle
import multiprocessing as mp
import sys

from code.classes import Point,Center,Subspace,Affine
from code.algo import LinearProjClustering, run
from coresets import coresets
from code.utilities import gen_rand_partition, Socially_Fair_Clustering_Cost, wSocially_Fair_Clustering_Cost

def update(results,q,mdict):
    # global results
    while 1:
        m = q.get()
        if m==[]:
            mdict['output'] = results
            break
        sys.stdout.flush()
        algo,k,cor_num,init,new_centers,runtime = m
        results[k].result[algo][k][cor_num][init]["centers"] = new_centers
        results[k].result[algo][k][cor_num][init]["num_iters"] += 1
        results[k].result[algo][k][cor_num][init]["running_time"] += runtime


def process(args,q):
    k,cor_num,init,algos,coreset,d,ell,z,centers = args
    try:
        new_centers, cpcost, runtime = run(coreset,centers,k,d,ell,z)
    except ValueError as e:
        print("ALGO: Failed: k="+str(k))
        print(e)
        sys.stdout.flush()
        return
    except ArithmeticError as e:
        print("ALGO: Failed: k="+str(k))
        print(e)
        sys.stdout.flush()
        return
    for algo in algos:
        q.put([algo,k,cor_num,init,new_centers,runtime])


def solve_clustering(dataset,name,k_vals,z,iter,is_PCA=0):
   
    results = {}
    for k in k_vals:
        f = open("./results/"+dataset+"/" + name+"_k="+str(k) + "_picklefile","rb")
        results[k] = pickle.load(f)
        f.close()
    
    manager = mp.Manager()
    mdict = manager.dict()
    q = manager.Queue()
    pool = mp.Pool(mp.cpu_count() + 4)
    watcher = pool.apply_async(update, (results,q,mdict))
    jobs = []

    for k in k_vals:
        print("ALGO: Start: k="+str(k))
        n,d,ell = results[k].get_params()
        algos = [algo for algo in results[k].result if algo[:4]=="ALGO"]
        for cor_num in results[k].result[algos[0]][k]:
            coreset = results[k].coresets[k][cor_num]
            for init in results[k].result[algos[0]][k][cor_num]:
                if results[k].result[algos[0]][k][cor_num][init]["num_iters"]==iter-1:              
                    # Step 2: Fair-Lloyd's Algorithm
                    centers = results[k].result[algos[0]][k][cor_num][init]["centers"]
                    job = pool.apply_async(process,([k,cor_num,init,algos,coreset,d,ell,z,centers],q)) 
                    jobs.append(job)
    for job in jobs:
        job.get()

    q.put([])
    watcher.get()
    results = mdict['output']
    pool.close()
    pool.join()


    for k in k_vals:
        f = open("./results/"+dataset+"/" + name+"_k="+str(k) + "_picklefile","wb")
        pickle.dump(results[k],f)
        f.close()

def solve_projective_linear(data,svar,groups,k,J,z,num_iters):
    ell = len(groups)
    # Step 1: Compute Coreset
    coreset = []
    for group in groups:
        data_group = data[np.asarray(svar)==group]
        coreset_gen = coresets.ProjectiveClusteringCoreset(data_group,n_clusters=k, J=J)
        coreset_size = data_group.shape[0]//500
        print(coreset_size)
        coreset_group,weights = coreset_gen.generate_coreset(coreset_size)
        coreset_group = pd.DataFrame(coreset_group,columns=data.columns)
        coreset += [Point(coreset_group.iloc[i],group,weights[i]) for i in range(coreset_group.shape[0])]
        # print(weights)

    # Step 2: Fair-Lloyd's Algorithm
    n = len(coreset)
    print(n)
    solver = LinearProjClustering(coreset,ell,k,J,z)
    solver.run(num_iters,gen_rand_partition(n,k))
    return solver.centers, solver.cost