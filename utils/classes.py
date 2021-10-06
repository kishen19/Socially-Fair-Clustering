import numpy as np

class Point:
    def __init__(self,coordinates,group,weight=1,cluster=None):
        self.cx = np.asarray(coordinates)
        self.group = group
        self.weight = weight
        self.cluster = cluster

class Center:
    def __init__(self,coordinates,cluster,index=None):
        self.cx= np.asarray(coordinates)
        self.cluster = cluster
        self.index = index # when centers are points from the data

    def distance(self,point):
        return np.linalg.norm(self.cx-point.cx)


class Subspace:
    def __init__(self,basis,cluster=None):
        self.basis = np.asarray(basis)
        self.cluster = cluster

    # ||point^T * basis||
    def distance(self,point):
        # print(point.cx.shape)
        # print(self.basis.shape)
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
    def __init__(self, dataset, name, dt_string, data, dataGC, groups, algos):
        self.dataset = dataset
        self.name = name
        self.dt_string = dt_string
        self.data = data # Original Data
        self.dataGC = dataGC # Groupwise Data
        self.distmatrix = []
        self.groups = groups # Original Groups

        # Parameters
        self.n,self.d = len(data),len(data[0].cx) # no. of points and dimension
        self.ell = len(groups)
        self.isPCA = False
        
        self.iters = 0
        self.coresets = {}
        self.result = {algo:{} for algo in algos}
        self.dataP = {}
        self.distmatrixP = {}
        self.PCA_d = {}

    def add_PCA_data(self,data,k):
        self.isPCA = True
        self.dataP[k] = data
        self.PCA_d[k] = len(data[0].cx)
    
    def add_distmatrix(self,distmatrix,k):
        self.distmatrixP[k] = distmatrix

    def add_coreset(self,k,J,coreset,ctime):
        if k not in self.coresets:
            self.coresets[k] = {}
        if J not in self.coresets[k]:
            self.coresets[k][J] = []
        self.coresets[k][J].append({"data":coreset,"time":ctime})

    def get_params(self,k):
        if self.isPCA:
            return self.n, self.PCA_d[k], self.ell
        else:
            return self.n, self.d, self.ell

    def get_data(self,k):
        if self.isPCA:
            return self.dataP[k]
        else:
            return self.data
    
    def get_distmatrix(self,k):
        if self.isPCA:
            return self.distmatrixP[k]
        else:
            return self.distmatrix
    
    def get_groups_centered(self):
        return self.dataGC

    def get_centers(self,algorithm,k,J,coreset_num,init_num):
        return self.result[algorithm][k][J][coreset_num][init_num]['centers']
        
    def add_new_result(self, algorithm, k, J, coreset_num, init_num, running_time, centers, iters):
        if k not in self.result[algorithm]:
            self.result[algorithm][k] = {}
        if J not in self.result[algorithm][k]:
            self.result[algorithm][k][J] = {}
        if coreset_num not in self.result[algorithm][k][J]:
            self.result[algorithm][k][J][coreset_num] = {}
        if init_num not in self.result[algorithm][k][J][coreset_num]:
            self.result[algorithm][k][J][coreset_num][init_num] = {}
        self.result[algorithm][k][J][coreset_num][init_num] = {
                                                            'running_time': running_time,
                                                            'centers': centers,
                                                            'num_iters':iters,
                                                            'cost':{},
                                                            'coreset_cost':{},
                                                            'PCA_cost':{},
                                                        }

    def add_new_cost(self,algorithm, k, J, coreset_num, init_num, costs, coreset_costs):
        for group in costs:
            self.result[algorithm][k][J][coreset_num][init_num]["cost"][group] = costs[group]
            self.result[algorithm][k][J][coreset_num][init_num]["coreset_cost"][group] = coreset_costs[group]
    
    def add_new_PCA_cost(self,algorithm, k, J, coreset_num, init_num, PCA_costs):
        for group in PCA_costs:
            self.result[algorithm][k][J][coreset_num][init_num]["PCA_cost"][group] = PCA_costs[group]

    def k_vs_val(self, algorithm, val, J=0):
        ks = []
        output = []
        index = []
        if val=="running_time":
            groups = sorted(self.groups.values())
            vals = []
            ks = [sorted(self.result[algorithm].keys())]
            for k in ks[0]:
                runtime = []
                for cor_num in self.result[algorithm][k][J]:
                    for init_num in self.result[algorithm][k][J][cor_num]:
                        runtime.append(self.result[algorithm][k][J][cor_num][init_num][val])
                vals.append(np.mean(runtime))
            output.append(vals)
            index.append(algorithm)
        elif val=="cost" or val=="coreset_cost"or val=="PCA_cost":
            groups = sorted(self.groups.values())
            ks = [sorted(self.result[algorithm].keys()) for group in groups]
            vals = [[] for group in groups]
            for k in ks[0]:
                cost = np.asarray([np.inf for i in range(self.ell)])
                for cor_num in self.result[algorithm][k][J]:
                    for init_num in self.result[algorithm][k][J][cor_num]:
                        cur_cost = [self.result[algorithm][k][J][cor_num][init_num][val][group] for group in groups]
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
                cost = np.asarray([np.inf for i in range(self.ell)])
                for cor_num in self.result[algorithm][k][J]:
                    for init_num in self.result[algorithm][k][J][cor_num]:
                        cur_cost = [self.result[algorithm][k][J][cor_num][init_num][val[:-6]][group] for group in groups]
                        if max(cost) > max(cur_cost):
                            cost = cur_cost
                vals.append(max(cost)/min(cost))
            output.append(vals)
            index.append(algorithm)
        
        elif val=="average cost" or val=="average coreset cost":
            groups = sorted(self.groups.values())
            ks = [sorted(self.result[algorithm].keys())]
            vals = []
            means = []
            errors = []
            for k in ks[0]:
                cost = []
                for cor_num in self.result[algorithm][k][J]:
                    for init_num in self.result[algorithm][k][J][cor_num]:
                        if algorithm == 'ALGO2':
                            if self.result[algorithm][k][J][cor_num][init_num]["num_iters"]==min(20,self.iters):
                                cost.append(max([self.result[algorithm][k][J][cor_num][init_num]['cost'][group] for group in groups]))
                        else:
                            cost.append(max([self.result[algorithm][k][J][cor_num][init_num]['cost'][group] for group in groups]))
                                
                
                m = np.mean(cost)
                std = np.std(cost)
                means.append(m)
                errors.append(std)
            vals = [means,errors]
            output.append(vals)
            index.append(algorithm)
        else:
            print("Error")
            exit(1)
        return ks, output, groups


    def J_vs_val(self,algorithm,val,k=1):
        Js = []
        output = []
        index = []
        if val=="running_time":
            groups = sorted(self.groups.values())
            vals = []
            Js = [sorted(self.result[algorithm][k].keys())]
            for J in Js[0]:
                runtime = []
                for cor_num in self.result[algorithm][k][J]:
                    for init_num in self.result[algorithm][k][J][cor_num]:
                        runtime.append(self.result[algorithm][k][J][cor_num][init_num][val])
                vals.append(np.mean(runtime))
            output.append(vals)
            index.append(algorithm)
        elif val=="cost" or val=="coreset_cost"or val=="PCA_cost":
            groups = sorted(self.groups.values())
            Js = [sorted(self.result[algorithm][k].keys()) for group in groups]
            vals = [[] for group in groups]
            for J in Js[0]:
                cost = np.asarray([np.inf for i in range(self.ell)])
                for cor_num in self.result[algorithm][k][J]:
                    for init_num in self.result[algorithm][k][J][cor_num]:
                        cur_cost = [self.result[algorithm][k][J][cor_num][init_num][val][group] for group in groups]
                        if max(cost) > max(cur_cost):
                            cost = cur_cost
                for i,group in enumerate(groups):
                    vals[i].append(cost[i])
            output = vals
            index = [algorithm+" ("+group+")" for group in groups]
            
        elif val=="cost_ratio" or val=="coreset_cost_ratio":
            groups = sorted(self.groups.values())
            Js = [sorted(self.result[algorithm][k].keys())]
            vals = []
            for J in Js[0]:
                cost = np.asarray([np.inf for i in range(self.ell)])
                for cor_num in self.result[algorithm][k][J]:
                    for init_num in self.result[algorithm][k][J][cor_num]:
                        cur_cost = [self.result[algorithm][k][J][cor_num][init_num][val[:-6]][group] for group in groups]
                        if max(cost) > max(cur_cost):
                            cost = cur_cost
                vals.append(max(cost)/min(cost))
            output.append(vals)
            index.append(algorithm)
        else:
            print("Error")
            exit(1)
        return Js, output, groups