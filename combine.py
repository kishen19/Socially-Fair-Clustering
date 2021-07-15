import pickle
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

from code.algo import run_final
from code.utilities import plot
from code.classes import Dataset, Point
from code.utilities import Socially_Fair_Clustering_Cost, wSocially_Fair_Clustering_Cost

def update(results,q,mdict):
    while 1:
        m = q.get()
        if m==[]:
            mdict['output'] = results
            break
        algo,k,cor_num,init,cost,coreset_cost = m
        results.result[algo][k][cor_num][init]["cost"] = cost
        results.result[algo][k][cor_num][init]["coreset_cost"] = coreset_cost
    
def process(args,q):
    k,cor_num,init,alg,z,centers,data,svar,groups,coresets = args
    costs = Socially_Fair_Clustering_Cost(data,svar,groups,centers,z)
    corcosts = wSocially_Fair_Clustering_Cost(coresets,groups,centers,z)
                        
    for group in groups:
        q.put([alg+" (" + groups[group] + ")",k,cor_num,init,costs[group],corcosts[group]])

def processPCA(args,q):
    k,cor_num,init,alg,n,d,ell,z,centers,data,svar,groups,dataP = args
    svar=np.asarray(svar)
    X = [Point(data.iloc[i,:],int(svar[i])) for i in range(n)]
    for i in range(n):
        best = np.inf
        for center in centers:
            new_dist = center.distance_(dataP.iloc[i,:])
            if new_dist < best:
                X[i].cluster = center.cluster
                best = new_dist
    new_centers,cpcost,runtime = run_final(X,k,d,ell,z)
    costs = Socially_Fair_Clustering_Cost(data,svar,groups,new_centers,z)
    for group in groups:
        q.put([alg+" (" + groups[group] + ")",k,cor_num,init,costs[group],0])

def compute_costs(results,z):
    algos = sorted(set([algo[:5].strip() for algo in results.result]))
    manager = mp.Manager()
    mdict = manager.dict()
    q = manager.Queue()
    pool = mp.Pool(mp.cpu_count() + 4)
    watcher = pool.apply_async(update, (results,q,mdict))
    jobs = []
    for alg in algos:
        algo = alg+" (" + results.groups[0] + ")"
        for k in results.result[algo]:
            for cor_num in results.result[algo][k]:
                for init in results.result[algo][k][cor_num]:
                    if results.isPCA:
                        centers = results.result[algo][k][cor_num][init]["centers"]
                        job = pool.apply_async(processPCA,([k,cor_num,init,alg,results.n,results.d,results.ell,z,centers,results.data,results.svar,results.groups,results.dataP],q))
                        jobs.append(job)
                    else:
                        centers = results.result[algo][k][cor_num][init]["centers"]
                        job = pool.apply_async(process,([k,cor_num,init,alg,z,centers,results.data,results.svar,results.groups, results.coresets[k][cor_num]],q)) 
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
    dataset="credit"
    isPCA = True
    namesuf= "_wPCA" if isPCA else "_woPCA"
    name = dataset + namesuf
    k_vals = range(4,17,2)
    results = Dataset(name,np.random.randint(0,100,(10,10)),np.random.randint(0,100,10),{})
    for k in tqdm(k_vals):
        f = open("./results/"+dataset+"/" + name+"_k="+str(k) + "_picklefile","rb")
        resultsk = pickle.load(f)
        f.close()
        resultsk = compute_costs(resultsk, 2)
        f = open("./results/"+dataset+"/" + name+"_k="+str(k) + "_picklefile","wb")
        pickle.dump(resultsk,f)
        f.close()
        for algo in resultsk.result:
            if algo not in results.result:
                results.result[algo] = {}
            results.result[algo][k] = resultsk.result[algo][k]

    f = open("./results/"+dataset+"/" + name + "_picklefile","wb")
    pickle.dump(results,f)
    f.close()
    
    f = open("./results/"+dataset+"/" + name + "_picklefile","rb")
    results = pickle.load(f)
    f.close()
    algos = sorted(algo for algo in results.result)
    print(algos)
    
    for i in range(0,len(algos),2):
        for k in results.result[algos[i]]:
            for cor_num in results.result[algos[i]][k]:
                for init in results.result[algos[i]][k][cor_num]:
                    print("k=",k,"cor_num=",cor_num,"iters=",results.result[algos[i+1]][k][cor_num][init]["num_iters"],algos[i][:5],"cost=",results.result[algos[i]][k][cor_num][init]["cost"],algos[i][:5],"cost=",results.result[algos[i+1]][k][cor_num][init]["cost"], "runtime=",results.result[algos[i+1]][k][cor_num][init]["running_time"])
    
    plot([results,results], 'cost')
    plot([results,results], 'running_time')
    plot([results,results], 'coreset_cost') 

if __name__=="__main__":
    main()