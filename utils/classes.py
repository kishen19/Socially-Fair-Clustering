import numpy as np

class Point:
    def __init__(self,coordinates,group,weight=1,cluster=None):
        self.cx = np.asarray(coordinates)
        self.group = group
        self.weight = weight
        self.cluster = cluster

class Center:
    def __init__(self,coordinates,cluster):
        self.cx= np.asarray(coordinates)
        self.cluster = cluster

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
    def __init__(self, name, data, groups, algos):
        self.name = name
        self.data = data # Original Data
        self.groups = groups # Original Groups

        # Parameters
        self.n,self.d = len(data),len(data[0].cx) # no. of points and dimension
        self.ell = len(groups)
        self.isPCA = False
        
        self.iters = 0
        self.coresets = {}
        self.result = {algo:{} for algo in algos}

    def add_PCA_data(self,data):
        self.isPCA = True
        self.dataP = data
        self.PCA_d = len(data[0].cx)
    
    def get_params(self):
        if self.isPCA:
            return self.n, self.PCA_d, self.ell
        else:
            return self.n, self.d, self.ell

    def add_coreset(self,k,coreset,ctime):
        if k not in self.coresets:
            self.coresets[k] = []
        self.coresets[k].append({"data":coreset,"time":ctime})

    def add_new_result(self, algorithm, k, coreset_num, init_num, running_time, centers, iters):
        if k not in self.result[algorithm]:
            self.result[algorithm][k] = {}
        if coreset_num not in self.result[algorithm][k]:
            self.result[algorithm][k][coreset_num] = {}
        if init_num not in self.result[algorithm][k][coreset_num]:
            self.result[algorithm][k][coreset_num][init_num] = {}
        self.result[algorithm][k][coreset_num][init_num] = {
                                                            'running_time': running_time,
                                                            'centers': centers,
                                                            'num_iters':iters,
                                                            'cost':{},
                                                            'coreset_cost':{},
                                                        }

    def add_new_cost(self,algorithm, k,coreset_num, init_num, costs, coreset_costs):
        for group in costs:
            self.result[algorithm][k][coreset_num][init_num]["cost"][group] = costs[group]
            self.result[algorithm][k][coreset_num][init_num]["coreset_cost"][group] = coreset_costs[group]

    def get_data(self):
        if self.isPCA:
            return self.dataP
        else:
            return self.data

    def get_centers(self,algorithm,k,coreset_num,init_num):
        return self.result[algorithm][k][coreset_num][init_num]['centers']

    def k_vs_val(self, algorithm, val):
        ks = []
        output = []
        index = []
        if val=="running_time":
            vals = []
            ks = [sorted(self.result[algorithm].keys())]
            for k in ks[0]:
                runtime = []
                for cor_num in self.result[algorithm][k]:
                    for init_num in self.result[algorithm][k][cor_num]:
                        runtime.append(self.result[algorithm][k][cor_num][init_num][val])
                vals.append(np.mean(runtime))
            output.append(vals)
            index.append(algorithm)
        elif val=="cost" or val=="coreset_cost":
            groups = sorted(self.groups.values())
            ks = [sorted(self.result[algorithm].keys()) for group in groups]
            vals = [[] for group in groups]
            for k in ks[0]:
                cost = np.asarray([np.inf for i in range(self.ell)])
                for cor_num in self.result[algorithm][k]:
                    for init_num in self.result[algorithm][k][cor_num]:
                        cur_cost = [self.result[algorithm][k][cor_num][init_num][val][group] for group in groups]
                        if max(cost) > max(cur_cost):
                            cost = cur_cost
                for i,group in enumerate(groups):
                    vals[i].append(cost[i])
            output = vals
            index = [algorithm+" ("+group+")" for group in groups]
            
        elif val=="cost_ratio" or val=="coreset_cost_ratio":
            groups = sorted(self.groups.values())
            ks = [sorted(self.result[algorithm].keys())]
            vals = []
            for k in ks[0]:
                best = np.inf
                for cor_num in self.result[algorithm][k]:
                    for init_num in self.result[algorithm][k][cor_num]:
                        max_ratio = 0
                        for g1 in groups:
                            for g2 in groups:
                                if g1!=g2:
                                    max_ratio = max(max_ratio,self.result[algorithm][k][cor_num][init_num][val[:-6]][g1]/self.result[algorithm][k][cor_num][init_num][val[:-6]][g2])
                        best = min(best,max_ratio)
                vals.append(best)
            output.append(vals)
            index.append(algorithm)
        else:
            print("Error")
            exit(1)
        return ks, output, index
