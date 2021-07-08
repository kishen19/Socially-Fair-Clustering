import numpy as np

class Point:
    def __init__(self,coordinates,group,weight=1,center=None,cluster=None):
        self.cx = np.asarray(coordinates)
        self.group = group
        self.weight = weight
        self.center = center
        self.cluster = cluster
        self.dist = np.inf


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

    def add_point(self,point):
        self.size+=1
        point.center = self
        point.cluster = self.cluster

    def distance(self,point):
        return np.linalg.norm(np.reshape(point.cx.reshape, (1,point.cx.shape)) @ self.basis)

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
    def __init__(self, name, n, m, ell):
        self.name = name
        self.n = n
        self.m = m # no. of points
        self.ell = ell
        self.result = {}

    
    def add_new_result(self, algorithm, k, cost, coreset_cost, running_time, num_iters):
        if algorithm not in self.result.keys():
            self.result[algorithm] = {}

        self.result[algorithm][k] = {'cost': cost, 
                        'coreset_cost': coreset_cost,
                        'running_time': running_time, 
                        'num_iters': num_iters}

    def time_per_iteration(self, algorithm, k):
        return self.result[algorithm][k]['running_time']/self.result[algorithm][k]['num_iters']
    
    def cost(self, algorithm, k):
        return self.result[algorithm][k]['cost']

    def coreset_cost(self, algorithm, k):
        return self.result[algorithm][k]['coreset_cost']    

    def k_vs_val(self, algorithm, y):
        return self.result[algorithm].keys(), [v[y] for v in self.result[algorithm].values()]    
