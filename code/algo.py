from random import randint
from tqdm import tqdm
import numpy as np

from code.classes import Point, Center, Subspace, Affine
from code.convex_opt import clustering


class ALGO:
    def __init__(self,data,k,num_groups,z):
        self.data = data
        self.n = len(self.data)
        self.d = len(self.data[0].cx)
        self.k = k
        self.z = z
        self.ell = num_groups
        self.costs = []
        self.centers = []

    def run(self,num_iters):
        self.gen_partition()
        for iter_num in range(num_iters):
            P,wts = self.categorize_data()
            new_centers,cost = clustering(self.k,self.d,self.ell,P,wts) # Call Convex Program
            new_centers = [Center(c,i) for i,c in enumerate(new_centers)]
            self.costs.append(cost)
            self.reassign(new_centers)
        self.centers = new_centers

    def gen_partition(self):
        rand_no = [randint(0,self.k-1) for i in range(self.n)]
        for i,x in enumerate(self.data):
            x.cluster = rand_no[i]

    def reassign(self,centers):
        for x in self.data:
            for center in centers:
                new_dist = center.distance(x)
                if new_dist < x.dist:
                    if x.center:
                        x.center.size-=1
                    center.size+=1
                    x.center = center
                    x.cluster = center.cluster
                    x.dist = new_dist
    
    def categorize_data(self):
        P = [[[] for i in range(self.k)] for j in range(self.ell)]
        wts = [[[] for i in range(self.k)] for j in range(self.ell)]
        for x in self.data:
            P[x.group][x.cluster].append(x.cx)
            wts[x.group][x.cluster].append(x.weight)
        return np.asarray(P),np.asarray(wts)