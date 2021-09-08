import pickle, yaml
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

from code.algos import run_algo, run_algo2, run_fair_lloyd, run_lloyd
from utils import cluster_assign
from utils.utilities import Socially_Fair_Clustering_Cost, plot
from utils.classes import Point,Dataset
from utils.preprocess import get_data

def update(results,q,mdict):
    while 1:
        m = q.get()
        if m==[]:
            mdict['output'] = results
            break
        algo,k,J,cor_num,init_num,costs,coreset_costs = m
        results.add_new_cost(algo,k,J,cor_num,init_num,costs,coreset_costs)
    
def PCA_cost(data,groups,centers):
    costs = {groups[group]:0 for group in groups}
    num = {groups[group]:0 for group in groups}
    for i,p in enumerate(data):
        costs[groups[p.group]] += (np.linalg.norm(p.cx)**2 - np.linalg.norm(centers[0].basis[i])**2)
        num[groups[p.group]] += 1
    for group in costs:
        costs[group]/=num[group]
    return costs

def process(args,q):
    algo,k,J,z,cor_num,init_num,data,groups,coreset,centers = args
    if J > 0 and algo =="PCA":
        costs = PCA_cost(data,groups,centers)
    else:
        costs = Socially_Fair_Clustering_Cost(data,groups,centers,J,z)
    if coreset and J==0:
        coreset_costs = Socially_Fair_Clustering_Cost(coreset,groups,centers,J,z)
    else:
        coreset_costs = {group:0 for group in costs}
    coreset_costs = {group:0 for group in costs}
    q.put([algo,k,J,cor_num,init_num,costs,coreset_costs])

def processPCA(args,q):
    algo,k,J,z,cor_num,init_num,data,groups,dataP,centers = args
    n = len(data)
    d = len(data[0].cx)
    ell = len(groups)
    if J==0:
        assign = cluster_assign.cluster_assign(np.asarray([x.cx for x in dataP]),np.asarray([c.cx for c in centers]))
        for i in range(n):
            data[i].cluster = assign[i]
        if "ALGO" in algo:
            new_centers,time_taken = run_algo(data,k,d,ell,z,centers=None)
        elif algo == "Lloyd":
            new_centers,time_taken = run_lloyd(data,k,d,ell,z)
        elif algo == "Fair-Lloyd":
            new_centers,time_taken = run_fair_lloyd(data,k,d,ell,z)
        costs = Socially_Fair_Clustering_Cost(data,groups,new_centers,J,z)
        coreset_costs = {group:0 for group in costs}
        q.put([algo,k,J,cor_num,init_num,costs,coreset_costs])

def compute_costs(results,k_vals,J_vals,algos,Z):
    for k in tqdm(k_vals):
        manager = mp.Manager()
        mdict = manager.dict()
        q = manager.Queue()
        pool = mp.Pool(mp.cpu_count() + 4)
        watcher = pool.apply_async(update, (results,q,mdict))
        jobs = []
        for J in J_vals:
            for algo in algos:
                for cor_num in results.result[algo][k][J]:
                    if 'ALGO' in results.result:
                        coreset = results.coresets[k][J][cor_num]["data"]
                    else:
                        coreset = []
                    for init_num in results.result[algo][k][J][cor_num]:
                        centers = results.result[algo][k][J][cor_num][init_num]["centers"]
                        if results.isPCA:
                            job = pool.apply_async(processPCA,([algo,k,J,Z,cor_num,init_num,results.data,results.groups,results.dataP,centers],q))
                        else:
                            job = pool.apply_async(process,([algo,k,J,Z,cor_num,init_num,results.data,results.groups,coreset,centers],q))
                        jobs.append(job)
        for job in tqdm(jobs):
            job.get()

        q.put([])
        watcher.get()
        results = mdict['output']
        pool.close()
        pool.join()

    return results        

def main():
    # Importing Parameters:
    path = open("./config.yaml", 'r')
    params = yaml.load(path)
    
    # Reading Parameters:
    Z = params["Z"]
    DATASET = params["DATASET"]
    ATTR = params["ATTR"]
    K_VALS = params["K_VALS"]
    J_VALS = params["J_VALS"]
    ALGOS = params["ALGOS"]
    ISPCA = params["ISPCA"]
    DT_STRING = params["DT_STRING"]
    ITER_NUM = params["ITER_NUM"]
    ONLY_PLOT = params["ONLY_PLOT"]
    NAMESUF = "_wPCA" if ISPCA else "_woPCA"
    NAME = ATTR + NAMESUF
    
    if not ONLY_PLOT:
        f = open("./results/"+DATASET+"/" + DT_STRING + "/" + NAME + "_iters="+str(ITER_NUM),"rb")
        results = pickle.load(f)
        f.close()

        results = compute_costs(results, K_VALS, J_VALS, ALGOS, Z)

        f = open("./results/"+DATASET+"/" + DT_STRING + "/" + NAME + "_iters="+str(ITER_NUM),"wb")
        pickle.dump(results,f)
        f.close()
    
    f = open("./results/"+DATASET+"/" + DT_STRING + "/" + NAME + "_iters="+str(ITER_NUM),"rb")
    results = pickle.load(f)
    f.close()
    
    for algo in results.result:
        for k in results.result[algo]:
            for J in results.result[algo][k]:
                for cor_num in results.result[algo][k][J]:
                    for init_num in results.result[algo][k][J][cor_num]:
                        print(algo+"> k="+str(k),"J="+str(J),"cor_num="+str(cor_num),"init="+str(init_num),"->")
                        for group in results.result[algo][k][J][cor_num][init_num]["cost"]:
                            print("\t"+group+":",results.result[algo][k][J][cor_num][init_num]["cost"][group],end=" ")
                        print()
                        # for group in results.result[algo][k][cor_num][init_num]["coreset_cost"]:
                        #     print("\t"+group+":",results.result[algo][k][cor_num][init_num]["coreset_cost"][group],end=" ")
                        # print()
    if 0 in J_VALS:
        param="k"
    else:
        param = "J"
    plot([results], 'cost',param)
    plot([results], 'running_time',param)
    # plot([results], 'coreset_cost',param)
    plot([results], 'cost_ratio',param)


if __name__=="__main__":
    main()