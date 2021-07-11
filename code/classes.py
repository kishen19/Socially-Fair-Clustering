import numpy as np

class Point:
    def __init__(self,coordinates,group,weight=1,center=None,cluster=None):
        self.cx = np.asarray(coordinates)
        self.group = group
        self.weight = weight
        self.center = center
        self.cluster = cluster
        self.dist = np.inf

    def reset(self):
        self.dist = np.inf
        self.center = None
        self.cluster = None

class Center:
    def __init__(self,coordinates,cluster=None):
        self.cx= np.asarray(coordinates)
        self.size = 0
        self.cluster = cluster

    def add_point(self,point):
        self.size+=1
        point.center = self
        point.cluster = self.cluster

    def distance(self,point):
        return np.linalg.norm(self.cx-point.cx)


class Subspace:
    def __init__(self,basis,cluster=None):
        self.basis = basis
        self.cluster = cluster
        self.size = 0

    def add_point(self,point):
        self.size+=1
        point.center = self
        point.cluster = self.cluster

    def distance(self,point):
        return np.linalg.norm(np.reshape(point.cx, (1,point.cx.shape[0])) @ self.basis)

class Affine:
    def __init__(self,basis,translation,cluster=None):
        self.basis = basis
        self.b = translation
        self.cluster = cluster

    def add_point(self,point):
        self.size+=1
        point.center = self
        point.cluster = self.cluster

    def distance(self,point):
        pass
    
class Dataset:
    def __init__(self, name, n, d, ell, init_centers):
        self.name = name
        self.n = n # no. of points
        self.d = d
        self.ell = ell
        self.result = {}
        self.init_centers = init_centers

    def add_new_result(self, algorithm, k, init_num, num_iters, cost, running_time, coreset_cost, coreset_time):
        if algorithm not in self.result:
            self.result[algorithm] = {}
        if k not in self.result[algorithm]:
            self.result[algorithm][k] = {}
        self.result[algorithm][k][init_num] = {'cost': cost, 
                        'coreset_cost': coreset_cost,
                        'running_time': running_time, 
                        'coreset_time': coreset_time,
                        'num_iters': num_iters}

    def time_per_iteration(self, algorithm, k):
        return self.result[algorithm][k]['running_time']/self.result[algorithm][k]['num_iters']
    
    def cost(self, algorithm, k):
        return self.result[algorithm][k]['cost']

    def coreset_cost(self, algorithm, k):
        return self.result[algorithm][k]['coreset_cost']

    def k_vs_val(self, algorithm, val):
        ks = sorted(self.result[algorithm].keys())
        vals = [np.sum([self.result[algorithm][k][init_num][val] for init_num in range(len(self.init_centers))])/len(self.init_centers) for k in self.result[algorithm]]
        return ks, vals