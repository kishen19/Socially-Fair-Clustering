import pickle
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

from code.algos import run_algo, run_fair_lloyd, run_lloyd
from utils import cluster_assign
from utils.utilities import Socially_Fair_Clustering_Cost, plot, plot_ratios
from utils.classes import Dataset

def update(results,q,mdict):
    while 1:
        m = q.get()
        if m==[]:
            mdict['output'] = results
            break
        algo,k,cor_num,init_num,costs,coreset_costs = m
        results.add_new_cost(algo,k,cor_num,init_num,costs,coreset_costs)
    
def process(args,q):
    algo,k,cor_num,init_num,data,svar,groups,coreset,weights,coreset_svar,centers,z = args
    costs = Socially_Fair_Clustering_Cost(data,svar,np.ones(data.shape[0]),groups,centers,z)
    coreset_costs = Socially_Fair_Clustering_Cost(coreset,coreset_svar,weights,groups,centers,z)
    q.put([algo,k,cor_num,init_num,costs,coreset_costs])

def processPCA(args,q):
    algo,k,cor_num,init_num,data,svar,groups,dataP,centers,z = args
    n,d = data.shape[0]
    ell = len(groups)
    assign = cluster_assign.cluster_assign(dataP,centers)
    if algo == "ALGO":
        new_centers,time_taken = run_algo(data,svar,np.ones(n),k,d,ell,z,clustering=assign)
    elif algo == "Lloyd":
        new_centers,time_taken = run_lloyd(data,svar,k,d,ell,z,clustering=assign)
    elif algo == "Fair-Lloyd":
        new_centers,time_taken = run_fair_lloyd(data,svar,k,d,ell,z,clustering=assign)
    costs = Socially_Fair_Clustering_Cost(data,svar,groups,new_centers,z)
    coreset_costs = {group:0 for group in costs}
    q.put([algo,k,cor_num,init_num,costs,coreset_costs])

def compute_costs(results,z):
    manager = mp.Manager()
    mdict = manager.dict()
    q = manager.Queue()
    pool = mp.Pool(mp.cpu_count() + 4)
    watcher = pool.apply_async(update, (results,q,mdict))
    jobs = []
    for algo in results.result:
        for k in results.result[algo]:
            for cor_num in results.result[algo][k]:
                for init_num in results.result[algo][k][cor_num]:
                    if results.isPCA:
                        centers = results.result[algo][k][cor_num][init_num]["centers"]
                        job = pool.apply_async(processPCA,([algo,k,cor_num,init_num,results.data,results.svar,results.groups,results.dataP,centers,z],q))
                        jobs.append(job)
                    else:
                        centers = results.result[algo][k][cor_num][init_num]["centers"]
                        job = pool.apply_async(process,([algo,k,cor_num,init_num,results.data,results.svar,results.groups,results.coresets[k][cor_num]["data"],results.coresets[k][cor_num]["weights"],results.coresets[k][cor_num]["svar"],centers,z],q)) 
                        jobs.append(job)
    for job in jobs:
        job.get()

    q.put([])
    watcher.get()
    results = mdict['output']
    pool.close()
    pool.join()

    return results        

def main():
    # dataset="adult"
    # attr = "RACE"
    dataset="credit"
    attr = "EDUCATION"
    isPCA = False
    namesuf= "_wPCA" if isPCA else "_woPCA"
    name = dataset+"_"+attr+namesuf
    algos = ["Lloyd","Fair-Lloyd","ALGO"]
    k_vals = range(4,5)
    results = Dataset(name,np.zeros([100,100]),np.zeros(100),{0:0,1:1},algos)
    for k in tqdm(k_vals):
        f = open("./results/"+dataset+"/" + name+"_k="+str(k) + "_picklefile","rb")
        resultsk = pickle.load(f)
        f.close()
        resultsk = compute_costs(resultsk, 2)
        f = open("./results/"+dataset+"/" + name+"_k="+str(k) + "_picklefile","wb")
        pickle.dump(resultsk,f)
        f.close()
        for algo in resultsk.result:
            results.result[algo][k] = resultsk.result[algo][k]

    f = open("./results/"+dataset+"/" + name + "_picklefile","wb")
    pickle.dump(results,f)
    f.close()
    
    f = open("./results/"+dataset+"/" + name + "_picklefile","rb")
    results = pickle.load(f)
    f.close()
    
    for algo in results.result:
        for k in results.result[algo]:
            for cor_num in results.result[algo][k]:
                for init_num in results.result[algo][k][cor_num]:
                    print(algo+"> k="+str(k),"cor_num="+str(cor_num),"init="+str(init_num),"->")
                    for group in results.result[algo][k][cor_num][init_num]["cost"]:
                        print("\t"+group+":",results.result[algo][k][cor_num][init_num]["cost"][group],end=" ")
                    print()
                    for group in results.result[algo][k][cor_num][init_num]["coreset_cost"]:
                        print("\t"+group+":",results.result[algo][k][cor_num][init_num]["coreset_cost"][group],end=" ")
                    print()
    
    # plot([results], 'cost')
    # plot([results], 'running_time')
    # plot([results], 'coreset_cost')
    # plot_ratios([results], 'cost_ratio')
    # plot_ratios([results], 'coreset_cost_ratio')


if __name__=="__main__":
    main()