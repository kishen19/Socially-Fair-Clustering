import pandas as pd
import numpy as np
import time
from tqdm import tqdm

from code.classes import Point,Center,Subspace,Affine
from code.algo import KZClustering, LinearProjClustering
from coresets import coresets
from code.utilities import gen_rand_partition, gen_rand_centers, Socially_Fair_Clustering_Cost


def solve_clustering(data,svar,groups,k,z,num_iters,init_centers):
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
    solver = KZClustering(coreset,ell,k,z)
    num_trials = len(init_centers)
    runtime = 0
    cor_cost = 0
    costs = {group:0 for group in groups}

    for trial in tqdm(range(num_trials)):
        incenters = init_centers[trial]
        centersi, cor_costi, runtimei = solver.run(num_iters,incenters)
        costi = Socially_Fair_Clustering_Cost(data,svar,groups,centersi,z)
        runtime += runtimei
        cor_cost += cor_costi

        for group in groups:
            costs[group] += costi[group]

    for group in groups:
        costs[group] /= num_trials

    return costs, cor_cost/num_trials, runtime/num_trials, _coreset_time

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