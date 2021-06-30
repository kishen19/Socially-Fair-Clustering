from random import randint
from tqdm import tqdm

class ALGO:
    def __init__(self,data,k,z):
        self.data = data
        self.n = len(self.data)
        self.k = k
        self.z = z
        self.costs = []
        self.centers = []

    def run(self,num_iters):
        self.gen_partition()
        for iter_num in tqdm(range(num_iters)):
            new_centers,cost = [] # Call Convex Program
            self.costs.append(cost)
            self.reassign(new_centers)
        self.centers = new_centers

    def gen_partition(self):
        rand_no = [randint(1,self.k) for i in range(self.n)]
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

    