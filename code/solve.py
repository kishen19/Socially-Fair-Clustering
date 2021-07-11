import pandas as pd
import numpy as np
import time
from tqdm import tqdm

from code.classes import Point,Center,Subspace,Affine
from code.algo import KZClustering, LinearProjClustering
from coresets import coresets
from code.utilities import gen_rand_partition, gen_rand_centers, Socially_Fair_Clustering_Cost


def solve_clustering(dataset,data,svar,groups,k,z,num_iters,init_centers,q):
    ell = len(groups)
    n = data.shape[0]
    # Step 1: Compute Coreset
    coreset = []
    coreset_size = 1000
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

    # Step 2: Fair-Lloyd's Algorithm
    num_inits = len(init_centers)
    solvers = [KZClustering(coreset,ell,k,z,init_centers[i]) for i in range(num_inits)]
    runtimes = [0]*num_inits
    cor_cost = [0]*num_inits
    costs = [{group:0 for group in groups} for i in range(num_inits)]
    store_iters = set([1,5,10,20,50,100,num_iters])
    for iter in tqdm(range(1,num_iters+1)):
        for init in range(num_inits):
            centersi, cor_costi, runtimei = solvers[init].run()
            costi = Socially_Fair_Clustering_Cost(data,svar,groups,centersi,z)
            runtimes[init] += runtimei
            cor_cost[init] = cor_costi
            for group in groups:
                costs[init][group] = costi[group]
        if iter in store_iters:
            for init in range(num_inits):
                for group in groups:
                    q.put([dataset,"ALGO"+" ("+ groups[group] + ")",k,init,iter,costs[init][group],runtimes[init],cor_cost[init],_coreset_time])

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