import numpy as np

class Point:
    def __init__(self,coordinates,group,weight=1,center=None,cluster=None):
        self.cx = np.asarray(coordinates)
        self.group = group
        self.weight = weight
        self.cluster = cluster

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
        return np.linalg.norm(self.cx-point)


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
    def __init__(self, name, data, svar, groups):
        self.name = name
        self.data = data
        self.n,self.d = data.shape # no. of points
        self.svar = svar
        self.groups = groups
        self.ell = len(groups)
        self.dataP = [] # For PCA
        self.coresets = {}
        self.result = {}
        self.isPCA = False

    def add_PCA_data(self,data):
        self.isPCA = True
        self.dataP = data
        self.PCA_d = data.shape[1]

    
    def get_params(self):
        if self.isPCA:
            return self.n, self.PCA_d, self.ell
        else:
            return self.n, self.d, self.ell
    
    def get_data(self):
        if self.isPCA:
            return self.dataP
        else:
            return self.data

    def add_coreset(self,k,coreset):
        if k not in self.coresets:
            self.coresets[k] = []
        self.coresets[k].append(coreset)

    def add_new_result(self, algorithm, k, init_num, coreset_num, num_iters, cost, running_time, coreset_cost, coreset_time, centers):
        if algorithm not in self.result:
            self.result[algorithm] = {}
        if k not in self.result[algorithm]:
            self.result[algorithm][k] = {}
        if coreset_num not in self.result[algorithm][k]:
            self.result[algorithm][k][coreset_num] = {}
        self.result[algorithm][k][coreset_num][init_num] = {'cost': cost, 
                        'running_time': running_time,
                        'coreset_cost': coreset_cost,
                        'coreset_time': coreset_time,
                        'num_iters': num_iters,
                        'centers':centers}

    
    def k_vs_val(self, algorithm, val):
        ks = sorted(self.result[algorithm].keys())
        if val=="running_time":
            vals = []
            for k in self.result[algorithm]:
                runtime = []
                for cor_num in self.result[algorithm][k]:
                    for init in self.result[algorithm][k][cor_num]:
                        runtime.append(self.result[algorithm][k][cor_num][init][val])
                vals.append(np.mean(runtime))
        elif val=="cost" or val=="coreset_cost":
            algos = sorted([algo for algo in self.result if algo[:5]==algorithm[:5]])
            w = algos.index(algorithm)
            vals = []
            for k in self.result[algorithm]:
                cost = np.asarray([np.inf for algo in algos])
                for cor_num in self.result[algorithm][k]:
                    for init in self.result[algorithm][k][cor_num]:
                        if max(cost) > max([self.result[algo][k][cor_num][init][val] for algo in algos]):
                            cost = [self.result[algo][k][cor_num][init][val] for algo in algos]
                vals.append(cost[w])
        elif val=="cost_ratio" or val=="coreset_cost_ratio":
            algos = sorted([algo for algo in self.result if algo[:5]==algorithm[:5]])
            algorithm = algos[0]
            vals = []
            for k in self.result[algorithm]:
                best = np.inf
                for cor_num in self.result[algorithm][k]:
                    for init in self.result[algorithm][k][cor_num]:
                        max_ratio = 0
                        for algo in algos:
                            for algo1 in algos:
                                if algo!=algo1:
                                    max_ratio = max(max_ratio,self.result[algo][k][cor_num][init][val[:-6]]/self.result[algo1][k][cor_num][init][val[:-6]])
                        best = min(best,max_ratio)
                vals.append(best)
        else:
            print("Error")
            exit(1)
        return ks, vals
