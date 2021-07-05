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
        return np.linalg.norm(self.basis @ point.cx)

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