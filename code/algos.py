import numpy as np
import heapq
from random import choice
import time

from utils.classes import Center,Subspace
from utils import cluster_assign
from code.convex_prog import kzclustering,linearprojclustering
from code.fair_lloyd import solve_kmeans_clustering

def reassign(data,centers):
    assign = cluster_assign.cluster_assign(np.asarray([x.cx for x in data]),np.asarray([c.cx for c in centers]))
    for i,p in enumerate(data):
        p.cluster = assign[i]

#-----------------------------------------------------#
# Our ALGO

def run_algo(data,k,d,ell,z,centers=None):
    if centers is not None:
        reassign(data,centers)
    _st = time.time()
    new_centers,cost_ = kzclustering(data,k,d,ell,z) # Call Convex Program
    _ed = time.time()
    new_centers = [Center(new_centers[i],i) for i in range(k)]
    return new_centers, _ed-_st

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


#---------------------------------------------------#
# Our Algorithm for Socially Fair Subspace Approximation

class Base:
    def __init__(self,data,num_groups,k,z):
        self.data = data
        self.n = len(self.data)
        self.d = len(self.data[0].cx)
        self.k = k
        self.z = z
        self.ell = num_groups

    def init_partition(self,start_partition):
        for i,x in enumerate(self.data):
            x.cluster = start_partition[i]

    def reassign(self,centers):
        for x in self.data:
            x.reset()
            for center in centers:
                new_dist = center.distance(x)
                if new_dist < x.dist:
                    if x.center:
                        x.center.size-=1
                    center.size+=1
                    x.center = center
                    x.cluster = center.cluster
                    x.dist = new_dist


class LinearProjClustering(Base):
    def __init__(self,data,num_groups,k,J,z):
        self.J = J
        super().__init__(data,num_groups,k,z)

    def run(self,num_iters,start_partition):
        self.init_partition(start_partition)
        for iter_num in range(num_iters):
            new_centers,cost = linearprojclustering(self.data,self.k,self.J,self.d,self.ell,self.z) # Call Convex Program
            new_centers = [Subspace(self.DTV10rounding(c),i) for i,c in enumerate(new_centers)]
            self.reassign(new_centers)
        self.centers = new_centers
        self.cost = cost
    
    def DTV10rounding(self,X):
        # X is psd
        s,x = np.linalg.eigh(X)
        s,x = s[::-1],x[:,::-1]
        assert np.all(s >= -1e-5)
        s = s[s>=1e-5]
        x = x[:,:len(s)]
        r = len(s)
        w = self.d - self.J
        y = [np.zeros(self.d) for i in range(w)]
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