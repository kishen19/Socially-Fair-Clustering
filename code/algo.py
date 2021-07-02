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

        self.cost = None
        self.centers = []

    def run(self,num_iters,start_partition):
        self.init_partition(start_partition)
        for iter_num in range(num_iters):
            new_centers,cost = clustering(self.data,self.k,self.d,self.ell) # Call Convex Program
            new_centers = [Center(c,i) for i,c in enumerate(new_centers)]
            self.reassign(new_centers)
        self.centers = new_centers
        self.cost = cost

    def init_partition(self,start_partition):
        for i,x in enumerate(self.data):
            x.cluster = start_partition[i]

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