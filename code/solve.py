import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import pickle

from code.classes import Point,Center,Subspace,Affine
from code.algo import LinearProjClustering, run
from coresets import coresets
from code.utilities import gen_rand_partition, Socially_Fair_Clustering_Cost, wSocially_Fair_Clustering_Cost


def solve_clustering(dataset,name,data,svar,groups,k,z,num_iters,is_PCA=0):
    f = open("./results/"+dataset+"/" + name+"_k="+str(k) + "_picklefile","rb")
    results = pickle.load(f)
    f.close()
    n,d = data.shape
    ell = len(groups)
    algos = [algo for algo in results.result if algo[:4]=="ALGO"]
    store_iters = set([1,5,10,20,50,100,num_iters])
    store_iters = [num_iters]
    for iter in tqdm(range(1,num_iters+1)):
        for cor_num in results.result[algos[0]][k]:
            coreset = results.coresets[k][cor_num]
            for init in results.result[algos[0]][k][cor_num]:
                if results.result[algos[0]][k][cor_num][init]["num_iters"]==iter-1:              
                    # Step 2: Fair-Lloyd's Algorithm
                    centers = results.result[algos[0]][k][cor_num][init]["centers"]
                    new_centers, cpcost, runtime = run(coreset,centers,k,d,ell,z)
                    for algo in algos:
                        results.result[algo][k][cor_num][init]["centers"] = new_centers
                        results.result[algo][k][cor_num][init]["num_iters"] += 1
                        results.result[algo][k][cor_num][init]["running_time"] += runtime

        if iter in store_iters:
            f = open("./results/"+dataset+"/" + name+"_k="+str(k) + "_picklefile","wb")
            pickle.dump(results,f)
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