from code.classes import Point, Center, Subspace, Affine
from code.convex_opt import kzclustering,linearprojclustering
import numpy as np
import heapq
from random import choice
import time

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


class KZClustering(Base):
    def __init__(self,data,num_groups,k,z):
        super().__init__(data,num_groups,k,z)

    def run(self,num_iters,init_centers):
        _st = time.time()
        self.reassign(init_centers)
        for iter_num in range(num_iters):
            new_centers,cost = kzclustering(self.data,self.k,self.d,self.ell,self.z) # Call Convex Program
            new_centers = [Center(c,i) for i,c in enumerate(new_centers)]
            self.reassign(new_centers)
        _ed = time.time()
        return new_centers, cost, _ed-_st


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