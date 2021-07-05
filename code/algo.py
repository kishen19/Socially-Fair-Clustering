from code.classes import Point, Center, Subspace, Affine
from code.convex_opt import kzclustering,linearprojclustering

class Base:
    def __init__(self,data,num_groups,k,z):
        self.data = data
        self.n = len(self.data)
        self.d = len(self.data[0].cx)
        self.k = k
        self.z = z
        self.ell = num_groups

        self.cost = None
        self.centers = []

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


class KZClustering(Base):
    def __init__(self,data,num_groups,k,z):
        super().__init__(data,num_groups,k,z)

    def run(self,num_iters,start_partition):
        self.init_partition(start_partition)
        for iter_num in range(num_iters):
            new_centers,cost = kzclustering(self.data,self.k,self.d,self.ell,self.z) # Call Convex Program
            new_centers = [Center(c,i) for i,c in enumerate(new_centers)]
            self.reassign(new_centers)
        self.centers = new_centers
        self.cost = cost


class LinearProjClustering(Base):
    def __init__(self,data,num_groups,k,J,z):
        self.J = J
        super().__init__(data,num_groups,k,z)

    def run(self,num_iters,start_partition):
        self.init_partition(start_partition)
        for iter_num in range(num_iters):
            new_centers,cost = linearprojclustering(self.data,self.k,self.J,self.d,self.ell,self.z) # Call Convex Program
            new_centers = [Subspace(c,i) for i,c in enumerate(new_centers)]
            self.reassign(new_centers)
        self.centers = new_centers
        self.cost = cost