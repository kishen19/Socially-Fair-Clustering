import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import multiprocessing as mp
import sys

from utils.classes import Point
from utils.utilities import gen_rand_partition
from code.algos import run_algo, run_lloyd, run_fair_lloyd, LinearProjClustering
from coresets import coresets


#----------------------------------------------------------------------#
# Multiprocessing target functions
#----------------------------------------------------------------------#
# Updating results
def update(results,q,mdict):
    while 1:
        m = q.get()
        if m==[]:
            mdict['output'] = results
            break
        sys.stdout.flush()
        algo,k,cor_num,init_num,iter,new_centers,time_taken = m
        results[k].add_new_result(algo, k, cor_num, init_num, time_taken, new_centers, iter)

# Processing each Input
def process(args,q):
    algo,k,cor_num,init_num,iter,data,coreset,d,ell,z,centers = args
    try:
        if algo == "ALGO":
            new_centers, time_taken = run_algo(coreset,k,d,ell,z,centers)
        elif algo=="Lloyd":
            new_centers, time_taken = run_lloyd(data,k,d,ell,z,centers)
        elif algo=="Fair-Lloyd":
            new_centers, time_taken = run_fair_lloyd(data,k,d,ell,z,centers)
        q.put([algo,k,cor_num,init_num,iter,new_centers,time_taken])
    except ValueError as e:
        print("ALGO: Failed: k="+str(k),"cor_num="+str(cor_num),"init="+str(init_num))
        print(e)
        sys.stdout.flush()
    except ArithmeticError as e:
        print("ALGO: Failed: k="+str(k),"cor_num="+str(cor_num),"init="+str(init_num))
        print(e)
        sys.stdout.flush()

#----------------------------------------------------------------------#
# Main Function
#----------------------------------------------------------------------#
def solve_clustering(dataset,name,k_vals,z,iter):
    results = {}
    for k in k_vals:
        f = open("./results/"+dataset+"/" + name+"_k="+str(k) + "_picklefile","rb")
        results[k] = pickle.load(f)
        f.close()
    
    # Multiprocessing Part
    manager = mp.Manager()
    mdict = manager.dict()
    q = manager.Queue()
    pool = mp.Pool(mp.cpu_count()+4)
    watcher = pool.apply_async(update, (results,q,mdict))
    jobs = []

    for k in k_vals:
        if results[k].iters == iter-1:
            data = results[k].get_data()
            for algo in results[k].result:
                print(algo+"> Start: k="+str(k))
                n,d,ell = results[k].get_params()
                for cor_num in results[k].result[algo][k]:
                    coreset = results[k].coresets[k][cor_num]["data"]
                    for init_num in results[k].result[algo][k][cor_num]:
                        if results[k].result[algo][k][cor_num][init_num]["num_iters"] == iter-1:
                            centers = results[k].result[algo][k][cor_num][init_num]["centers"]
                            job = pool.apply_async(process,([algo,k,cor_num,init_num,iter,data,coreset,d,ell,z,centers],q))
                            jobs.append(job)
    # Closing Multiprocessing Pool
    for job in tqdm(jobs):
        job.get()    
    q.put([])
    watcher.get()
    results = mdict['output']
    pool.close()
    pool.join()
    # Dumping Output to Pickle file
    for k in k_vals:
        results[k].iters = max(results[k].iters, iter)
        f = open("./results/"+dataset+"/" + name+"_k="+str(k) + "_picklefile","wb")
        pickle.dump(results[k],f)
        f.close()


#----------------------------------------------------------------------#
# Projective Clustering
#----------------------------------------------------------------------#
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