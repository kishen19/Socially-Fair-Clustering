from random import randint
import numpy as np

def gen_rand_partition(n,k):
    return [randint(0,k-1) for i in range(n)]

def compute_cost(data,centers,z):
    cost = 0
    n = data.shape[0]
    for i in range(n):
        best = np.inf
        for center in centers:
            best = min(best,np.linalg.norm(data.iloc[i] - center.cx)**z)
        cost += best
    return cost/n

def Socially_Fair_Clustering_Cost(data,svar,groups,centers,z):
    cost = 0
    for group in groups:
        data_group = data[np.asarray(svar)==group]
        group_cost = compute_cost(data_group,centers,z)
        cost = max(cost,group_cost)
    return cost
