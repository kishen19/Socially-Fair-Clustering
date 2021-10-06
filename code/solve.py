import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import multiprocessing as mp
import sys

from utils.classes import Point
from utils.utilities import gen_rand_partition
from code.algos import run_algo,run_algo2, run_lloyd, run_fair_lloyd, run_kmedoids, run_algo_proj, run_algo_proj2, run_PCA
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
        algo,k,J,cor_num,init_num,iter,new_centers,time_taken = m
        results.add_new_result(algo, k, J,cor_num, init_num, time_taken, new_centers, iter)

# Processing each Input
def process(args,q):
    algo,k,J,cor_num,init_num,iter,data,dataGC,distmatrix,groups,coreset,d,ell,z,centers,n_samples,sample_size = args
    try:
        if algo == "ALGO":
            if J==0:
                new_centers, time_taken = run_algo(coreset,k,d,ell,z,centers=centers)
            else:
                if iter==1:
                    new_centers, time_taken = run_algo_proj(coreset,k,d,ell,z,J,init_partition=centers)
                else:
                    new_centers, time_taken = run_algo_proj(coreset,k,d,ell,z,J,centers=centers)
        elif algo == "ALGO2":
            if J==0:
                new_centers, time_taken = run_algo2(data,groups,k,d,ell,z,centers=centers,n_samples=n_samples,sample_size=sample_size)
            else:
                if iter==1:
                    new_centers, time_taken = run_algo_proj2(data,dataGC,groups,k,d,ell,z,J,n_samples=n_samples,sample_size=sample_size,init_partition=centers)
                else:
                    new_centers, time_taken = run_algo_proj2(data,dataGC,groups,k,d,ell,z,J,n_samples=n_samples,sample_size=sample_size,centers=centers)
        elif algo == "ALGO3":
            if J==0:
                new_centers, time_taken = run_algo(data,groups,k,d,ell,z,centers=centers)
            else:
                if iter==1:
                    new_centers, time_taken = run_algo_proj(data,dataGC,k,d,ell,z,J,init_partition=centers)
                else:
                    new_centers, time_taken = run_algo_proj(data,dataGC,k,d,ell,z,J,centers=centers)
        elif algo=="Lloyd":
            new_centers, time_taken = run_lloyd(data,k,d,ell,z,centers=centers)
        elif algo=="Fair-Lloyd":
            new_centers, time_taken = run_fair_lloyd(data,k,d,ell,z,centers=centers)
        elif algo=="PCA":
            if iter==1:
                new_centers, time_taken = run_PCA(data,dataGC,k,d,ell,z,J,init_partition=centers)
            else:
                new_centers, time_taken = run_PCA(data,dataGC,k,d,ell,z,J,centers=centers)
        elif algo=="KMedoids":
            new_centers, time_taken = run_kmedoids(data,distmatrix,k,d,ell,z,centers=centers)
        q.put([algo,k,J,cor_num,init_num,iter,new_centers,time_taken])        
    except ValueError as e:
        print(algo+": Failed: k="+str(k),"cor_num="+str(cor_num),"init="+str(init_num))
        print(e)
        sys.stdout.flush()
    except ArithmeticError as e:
        print(algo+": Failed: k="+str(k),"cor_num="+str(cor_num),"init="+str(init_num))
        print(e)
        sys.stdout.flush()
    except TypeError as e:
        print(algo+": Failed: k="+str(k),"cor_num="+str(cor_num),"init="+str(init_num))
        print(e)
        sys.stdout.flush()

#----------------------------------------------------------------------#
# Main Function
#----------------------------------------------------------------------#
def solve(iter, ALGOS, DATASET, dt_string, NAME, K_VALS, Z, J_VALS, ALGO2_N_SAMPLES, ALGO2_SAMPLE_SIZE):
    f = open("./results/"+ DATASET +"/" + dt_string + "/" + NAME + "_iters="+str(iter-1),"rb")
    results = pickle.load(f)
    f.close()
    if results.iters == iter-1:
        # Multiprocessing Part
        manager = mp.Manager()
        mdict = manager.dict()
        q = manager.Queue()
        pool = mp.Pool(mp.cpu_count()+4)
        watcher = pool.apply_async(update, (results,q,mdict))
        jobs = []

        for k in K_VALS:
            for J in J_VALS:
                data = results.get_data(k)
                distmatrix = results.get_distmatrix(k)
                if J>0:
                    dataGC = results.get_groups_centered() 
                else:
                    dataGC = None
                for algo in ALGOS:
                    if ('ALGO' not in algo)  or ('ALGO' in algo  and iter <= 20):
                        print(algo+"> Start: k="+str(k))
                        n,d,ell = results.get_params(k)
                        for cor_num in results.result[algo][k][J]:
                            if algo == "ALGO":
                                coreset = results.coresets[k][J][cor_num]["data"]
                            else:
                                coreset = []
                            for init_num in results.result[algo][k][J][cor_num]:
                                if results.result[algo][k][J][cor_num][init_num]["num_iters"] == iter-1:
                                    centers = results.result[algo][k][J][cor_num][init_num]["centers"]
                                    job = pool.apply_async(process,([algo,k,J,cor_num,init_num,iter,data,dataGC,distmatrix,results.groups,coreset,d,ell,Z,centers,ALGO2_N_SAMPLES,ALGO2_SAMPLE_SIZE],q))
                                    jobs.append(job)
        # Closing Multiprocessing Pool
        for job in tqdm(jobs):
            job.get()    
        q.put([])
        watcher.get()
        results = mdict['output']
        pool.close()
        pool.join()
        f = open("./results/"+DATASET+"/" + dt_string + "/" + NAME+"_iters="+str(iter),"rb")
        results1 = pickle.load(f)
        f.close()
        results1.result["Fair-Lloyd"] = results.result['Fair-Lloyd']
        # Dumping Output to Pickle file
        results.iters = max(results.iters, iter)
        f = open("./results/"+DATASET+"/" + dt_string + "/" + NAME+"_iters="+str(iter),"wb")
        pickle.dump(results1,f)
        f.close()