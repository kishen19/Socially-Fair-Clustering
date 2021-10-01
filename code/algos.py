import numpy as np
import heapq
from random import choice
import time
from sklearn.decomposition import PCA

from utils.classes import Center,Subspace
from utils import cluster_assign
from code.convex_prog import kzclustering,linearprojclustering
from code.fair_lloyd import solve_kmeans_clustering
from code.k_medoids import solve_kmedian_clustering


def reassign(data,centers):
    assign = cluster_assign.cluster_assign(np.asarray([x.cx for x in data]),np.asarray([c.cx for c in centers]))
    for i,p in enumerate(data):
        p.cluster = assign[i]

def assign_subspace(data,dataGC,centers):
    
    assign = [-1 for i in range(len(data))]
    for i,x in enumerate(data):
        best = np.inf
        for j,center in enumerate(centers):
            dist = center.distance(x)
            if best < dist:
                assign[i] = j
                best = dist
    id = [0 for i in range(len(dataGC))]
    for i,p in enumerate(data):
        p.cluster = assign[i]
        dataGC[p.group][id[p.group]].cluster = assign[i]
        id[p.group] += 1

#-----------------------------------------------------#
# Our ALGO

def run_algo(data,k,d,ell,z,centers=None):
    if centers is not None:
        reassign(data,centers)
    _st = time.time()
    new_centers,cost_ = kzclustering(data,k,d,ell,z,centers) # Call Convex Program
    _ed = time.time()
    new_centers = [Center(new_centers[i],i) for i in range(k)]
    return new_centers, _ed-_st

#-----------------------------------------------------#
# Our ALGO2 - Sample in each Cluster \cap Group
# 4000 -> 2x10 = 20: 200 per sub-group (Pij)
# Sample in Pij 200: prob in sample - 200/|Pij|, wts: |Pij|/200 per point sampled
# E[cost(Sj)] = Sum_i E[Sij]
# if |Pij|<200, all

def run_algo2(data,groups,k,d,ell,z,centers=None, n_samples = 5, sample_size = 100): # Reminder
    if centers is not None:
        reassign(data,centers)
    n = len(data)
    best_cost = np.inf
    best_centers = []
    runtime = 0
    data_groupwise = {i:{group:[x for x in data if (x.group==group and x.cluster == i)] for group in groups} for i in range(k)}
    flag = 0
    error = ''
    for _ in range(n_samples):
        try:
            sampled_data = []
            for i in range(k):
                for group in groups:
                    if len(data_groupwise[i][group]) < sample_size:
                        selected = np.asarray(range(len(data_groupwise[i][group])))
                    else:
                        selected = np.random.choice(range(len(data_groupwise[i][group])), size=sample_size, replace=False)
                    group_data = [data_groupwise[i][group][ind] for ind in selected]
                    for x in group_data:
                        x.weight = len(data_groupwise[i][group])/min(sample_size,len(data_groupwise[i][group]))
                    sampled_data += group_data
            _st = time.time()
            new_centers,cost_ = kzclustering(sampled_data,k,d,ell,z,centers) # Call Convex Program
            _ed = time.time()
            if cost_ < best_cost:
                best_cost = cost_
                best_centers = new_centers
            runtime += _ed-_st
            flag = 1
        except ValueError as e:
            error = e
        except ArithmeticError as e:
            error = e
    if flag==0:
        raise ValueError(error)
    best_centers = [Center(best_centers[i],i) for i in range(k)]
    return best_centers, runtime/n_samples

#-----------------------------------------------------#
# Our ALGO4 - Random samples

def run_algo4(data,groups,k,d,ell,z,centers=None, n_samples = 5, sample_size = 1000):
    if centers is not None:
        reassign(data,centers)
    n = len(data)
    best_cost = np.inf
    best_centers = []
    runtime = 0
    data_groupwise = {group:[x for x in data if x.group==group] for group in groups}
    flag = 0
    error = ''
    for _ in range(n_samples):
        try:
            sampled_data = []
            rem = sample_size
            for ind,group in enumerate(groups):
                group_sample = rem if ind == len(groups)-1 else int(len(data_groupwise[group])*sample_size/n)
                rem -= group_sample
                selected = np.random.choice(range(len(data_groupwise[group])), size=group_sample, replace=False)
                group_data = [data_groupwise[group][i] for i in selected]
                for x in group_data:
                    x.weight = len(data_groupwise[group])/group_sample
                sampled_data += group_data
            _st = time.time()
            new_centers,cost_ = kzclustering(sampled_data,k,d,ell,z,centers) # Call Convex Program
            _ed = time.time()
            if cost_ < best_cost:
                best_cost = cost_
                best_centers = new_centers
            runtime += _ed-_st
            flag = 1
        except ValueError as e:
            error = e
        except ArithmeticError as e:
            error = e
    if flag==0:
        raise ValueError(error)
    best_centers = [Center(best_centers[i],i) for i in range(k)]
    return best_centers, runtime/n_samples

#-----------------------------------------------------#
# Lloyd's Algorithm

def run_lloyd(data,k,d,ell,z,centers=None):
    if centers is not None:
        reassign(data,centers)
    new_centers = np.zeros((k,d))
    num = np.zeros(k)
    _st = time.time()
    for p in data:
        new_centers[p.cluster] += p.cx
        num[p.cluster] += 1
    for i in range(k):
        new_centers[i] = 0 if num[i] ==0 else new_centers[i]/num[i]
    _ed = time.time()
    new_centers = [Center(new_centers[i],i) for i in range(k)]
    return new_centers, _ed-_st

#----------------------------------------------------#
# Ghadiri et al.'s Fair Lloyd's Algorithm
# Reference: Mehrdad Ghadiri, Samira Samadi, and Santosh Vempala. 2021. Socially Fair k-Means Clustering. 
#            In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency (FAccT '21).

def run_fair_lloyd(data,k,d,ell,z,centers=None):
    if centers is not None:
        reassign(data,centers)
    _st = time.time()
    new_centers,cost = solve_kmeans_clustering(data,k,d,ell,z) # Call Convex Program
    _ed = time.time()
    new_centers = [Center(new_centers[i],i) for i in range(k)]
    return new_centers, _ed-_st

#-----------------------------------------------------#
# K-Medoids
def run_kmedoids(data,distmatrix,k,d,ell,z,centers=None):
    _st = time.time()
    new_centers,cost = solve_kmedian_clustering(data,distmatrix,k,d,ell,z,centers) # Call Convex Program
    _ed = time.time()
    final_centers = [Center(data[new_centers[i]].cx,i,index=new_centers[i]) for i in range(k)]
    return final_centers, _ed-_st

#---------------------------------------------------#
# Our Algorithm for Socially Fair Projective Clustering

def run_algo_proj(data,k,d,ell,z,J,centers=None,init_partition=None):
    if centers is not None:
        assign_subspace(data,centers)
    else:
        if init_partition is not None:
            for i,p in enumerate(data):
                p.cluster = init_partition[i]
    _st = time.time()
    new_centers,cost = linearprojclustering(data,k,J,d,ell,z) # Call Convex Program
    _ed = time.time()
    new_centers = [Subspace(DTV10rounding(c,d,J),i) for i,c in enumerate(new_centers)]
    return new_centers, _ed - _st

def run_algo_proj2(data,dataGC,groups,k,d,ell,z,J,centers=None,init_partition=None,sample_size = 1000, n_samples = 5):
    if centers is not None:
        assign_subspace(data,dataGC,centers)
    else:
        if init_partition is not None:
            id = [0 for i in range(len(dataGC))]
            for i,p in enumerate(data):
                p.cluster = init_partition[i]
                dataGC[p.group][id[p.group]].cluster = init_partition[i]
                id[p.group] += 1
    n = len(data)
    best_cost = np.inf
    best_centers = []
    runtime = 0
    
    # ### 1. ALGO on whole centered data
    # data_groupwise = {group:[x for x in data if x.group==group] for group in groups}
    # dataGC = data_groupwise

    ### 2. ALGO on group centered data
    # do nothing


    flag = 0
    error = ''
    for _ in range(n_samples):
        try:
            sampled_data = []
            rem = sample_size
            for ind,group in enumerate(groups):
                group_sample = rem if ind == len(groups)-1 else int(len(dataGC[group])*sample_size/n)
                rem -= group_sample
                selected = np.random.choice(range(len(dataGC[group])), size=group_sample, replace=False)
                group_data = [dataGC[group][i] for i in selected]
                for x in group_data:
                    x.weight = len(dataGC[group])/group_sample
                sampled_data += group_data
            _st = time.time()
            new_centers,cost_ = linearprojclustering(sampled_data,k,J,d,ell,z) # Call Convex Program
            _ed = time.time()
            if cost_ < best_cost:
                best_cost = cost_
                best_centers = new_centers
            runtime += _ed-_st
            flag = 1
        except ValueError as e:
            error = e
        except ArithmeticError as e:
            error = e
    if flag==0:
        raise ValueError(error)
    centers = [Subspace(DTV10rounding(c,d,J),i) for i,c in enumerate(best_centers)]
    return centers, runtime/n_samples

def DTV10rounding(X,d,J):
        # X is psd
        s,x = np.linalg.eigh(X)
        s,x = s[::-1],x[:,::-1]
        assert np.all(s >= -1e-5)
        s = s[s>=1e-5]
        x = x[:,:len(s)]
        r = len(s)
        w = d - J
        y = [np.zeros(d) for i in range(w)]
        h = [[0,i] for i in range(w)]
        heapq.heapify(h)
        for i in range(r):
            bi = choice([1,-1])
            cur = heapq.heappop(h)
            cur[0] += s[i]
            y[cur[1]] += bi*np.sqrt(s[i])*x[:,i]
            heapq.heappush(h,cur)
        Z = np.asarray([y[i]/np.linalg.norm(y[i]) for i in range(w)]).T
        return Z

#---------------------------------------------------#
# PCA for Subspace Approximation

def run_PCA(data,dataGC,k,d,ell,z,J,centers=None,init_partition=None):
    if centers is not None:
        assign_subspace(data,dataGC,centers)
    else:
        if init_partition is not None:
            for i,p in enumerate(data):
                p.cluster = init_partition[i]
    _st = time.time()
    # d x (d-J): 23 x 13
    pca = PCA(n_components = J) # 30000 x 13
    
    # ### (1) projecting data points on pca
    # dataP = pca.fit_transform(np.asarray([p.cx for p in data])) # 30000 x 10

    # ### (2) projecting group-centered points on group wise pca
    # dataP = np.array(pca.fit_transform(np.asarray([p.cx for p in dataGC[0]])))
    # for j in range(1,len(dataGC)):
    #     dataP = np.concatenate((dataP, pca.fit_transform(np.asarray([p.cx for p in dataGC[j]]))))
    # dataP = np.asarray(dataP)

    ### (3) projecting group-centered points on pca on whole data
    pca.fit(np.asarray([p.cx for p in data])) # 30000 x 10
    dataP = np.array(pca.transform(np.asarray([p.cx for p in dataGC[0]])))
    for j in range(1,len(dataGC)):
        dataP = np.concatenate((dataP, pca.transform(np.asarray([p.cx for p in dataGC[j]]))))
    dataP = np.asarray(dataP)


    # dataP = (pca.singular_values_ * pca.components_.T) # 13 x 23
    # Q, R = np.linalg.qr(dataP)
    new_centers = Subspace(dataP)
    _ed = time.time()
    return [new_centers], _ed - _st