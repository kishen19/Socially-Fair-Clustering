from code.classes import Point, Center, Dataset
from code import datagen
from code.utilities import gen_rand_centers,plot
from code.solve import solve_clustering
from code.lloyd import lloyd

import pickle
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import sys


def get_sens(dataset,flag):
    if dataset=="credit":
        data = pd.read_csv("./data/" + dataset + "/" + dataset + flag + ".csv",index_col=0)
        svar,data,groups = data.iloc[:,0],data.iloc[:,1:],{0:"Higher Education",1:"Lower Education"}
    return svar,data,groups

def pickle_data(q,flag):
    while 1:
        m = q.get()
        if m == []:
            break
        f = open("./results/"+m[0]+"/picklefile"+flag,"rb")
        resultsP = pickle.load(f)
        f.close()
        resultsP.add_new_result(*m[1:])
        f = open("./results/"+m[0]+"/picklefile"+flag,"wb")
        pickle.dump(resultsP,f)
        f.close()

def get_result(args,q):
    algo,dataset,k,init_centers,num_iters,flag = args
    try:
        print("Start: k="+str(k))
        sys.stdout.flush()
        svar,data,groups = get_sens(dataset,flag)
        if algo=="ALGO":
            costs, coreset_cost, running_time, coreset_time = solve_clustering(data,svar,groups,k,2,num_iters,init_centers)
        else:
            costs, running_time = lloyd(data,svar,groups,k,2,num_iters,init_centers)
            coreset_cost, coreset_time = 0,0
    except ArithmeticError:
        print("Failed: k="+str(k))
        print(e)
        sys.stdout.flush()
        return
    
    for group in groups:
        q.put([dataset,algo+" ("+ groups[group] + ")",k,num_iters,costs[group],running_time,coreset_cost,coreset_time])
    print("Done: k="+str(k))
    sys.stdout.flush()

def main():
    dataset = "credit"
    datagen.dataNgen(dataset)
    # for k in range(2,17):
        # datagen.dataPgen(dataset,k)
    flag = "N"
    # flag = "P" # PCA
    name="_woPCA"
    # name="_wPCA" # PCA

    # Generate Init_centers
    init_centers = {k:[] for k in range(2,17)}
    num_inits = 200
    k_vals = range(2,17)
    for k in k_vals:
        flag1 = "N"
        # flag1 = "P_K="+str(k) # PCA
        svar,data,groups = get_sens(dataset,flag1)
        for init in range(num_inits):
            mask = gen_rand_centers(data.shape[0],k)
            centers = [Center(data.iloc[mask[i],:],i) for i in range(k)]
            init_centers[k].append(centers)

    results = Dataset(dataset+name, data.shape[0],data.shape[1],len(groups),init_centers)
    f = open("./results/"+dataset+"/picklefile"+flag,"wb")
    pickle.dump(results,f)
    f.close()


    num_iters = 200

    f = open("./results/"+dataset+"/picklefile"+flag,"rb")
    results = pickle.load(f)
    f.close()

    for algo in ['ALGO', 'lloyd']:
        manager = mp.Manager()
        q = manager.Queue()    
        pool = mp.Pool(mp.cpu_count() + 2)
        watcher = pool.apply_async(pickle_data, (q,flag))
        jobs = []
        for k in k_vals:
            flag1 = "N"
            # flag1 = "P_K="+str(k) # PCA
            job = pool.apply_async(get_result,([algo,dataset,k,results.init_centers[k],num_iters,flag1],q))
            jobs.append(job)
        
        for job in jobs:
            job.get()

        q.put([])
        pool.close()
        pool.join()

def mainp():
    dataset="credit"
    f = open("./results/"+dataset+"/picklefile"+"N","rb")
    results = pickle.load(f)
    f.close()
    plot([results,results], 'cost')
    plot([results,results], 'coreset_cost')
    plot([results,results], 'running_time')
    # f = open("./results/"+dataset+"/picklefile"+"P","rb")
    # resultsP = pickle.load(f)
    # f.close()
    # print(resultsP.result)
    # plot([resultsP,resultsP], 'coreset_cost')
    # plot([resultsP,resultsP], 'cost')
    # plot([resultsP,resultsP], 'running_time')


if __name__=='__main__':
    main()
    mainp()