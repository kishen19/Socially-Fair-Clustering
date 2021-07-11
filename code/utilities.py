from random import randint, sample, shuffle
import numpy as np
from code.classes import Point, Dataset
from matplotlib import pyplot as plt
from itertools import cycle, islice

def gen_rand_partition(n,k):
    return [randint(0,k-1) for i in range(n)]

def gen_rand_centers(n,k):
    init = sample(range(n),k)
    shuffle(init)
    return init

def compute_cost(data,centers,z):
    cost = 0
    n = data.shape[0]
    for i in range(n):
        best = np.inf
        for center in centers:
            best = min(best,center.distance(Point(data.iloc[i],"dummy"))**z)
        cost += best
    return cost/n

def Socially_Fair_Clustering_Cost(data,svar,groups,centers,z):
    costs = {}
    for group in groups:
        data_group = data[np.asarray(svar)==group]
        group_cost = compute_cost(data_group,centers,z)
        costs[group] = group_cost
    return costs

def plot(results, y):
    '''
    resutls: list of Dataset objects; list
    y: y-axis label; str
    '''
    plt.rcParams["figure.figsize"] = (15,7)
    fig, axs = plt.subplots(1, len(results))
    
    for i, dataset in enumerate(results):
        algorithms = dataset.result.keys()
        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(len(algorithms) + 1))))
        markers = np.array(list(islice(cycle(['s', '.', '^',
                                             '|', 'o', 'x',
                                             '>', '<', 'p']),
                                      int(len(algorithms) + 1))))
        
        for j, alg in enumerate(algorithms):
            k, val = dataset.k_vs_val(alg, y)
            axs[i].plot(k, val, color=colors[j], markersize=10, marker=markers[j], fillstyle='none', label=alg)
            axs[i].legend(loc='upper right')
            axs[i].set_xlabel('('+chr(i+97)+')\t\t\t$k$\t\t\t')
            axs[i].set_title(dataset.name+' dataset')
        if i==0:
            axs[i].set_ylabel(y)
        
    plt.savefig(dataset.name+'_fig_sociallyFairClustering_'+y+'.png')



# if __name__=='__main__':
#     credit_noPCA = Dataset("credit", n=2, m=10, ell=2)
#     x=100
#     y=90
#     credit_noPCA.add_new_result('ALG1', k=2, cost=x/2, coreset_cost=y/3, running_time=1, num_iters=10)
#     credit_noPCA.add_new_result('ALG1', k=3, cost=x/3, coreset_cost=y/4, running_time=1.1, num_iters=10)
#     credit_noPCA.add_new_result('ALG1', k=4, cost=x/4, coreset_cost=y/5, running_time=1.2, num_iters=10)
#     credit_noPCA.add_new_result('ALG1', k=5, cost=x/5, coreset_cost=y/8, running_time=1.5, num_iters=10)
#     x=110
#     y=100
#     credit_noPCA.add_new_result('ALG2', k=2, cost=x/2, coreset_cost=y/3, running_time=1.2, num_iters=10)
#     credit_noPCA.add_new_result('ALG2', k=3, cost=x/3, coreset_cost=y/4, running_time=1.5, num_iters=10)
#     credit_noPCA.add_new_result('ALG2', k=4, cost=x/4, coreset_cost=y/5, running_time=1.8, num_iters=10)
#     x=150
#     y=135
#     credit_noPCA.add_new_result('ALG2', k=2, cost=x/2, coreset_cost=y/3, running_time=2, num_iters=10)
#     credit_noPCA.add_new_result('ALG2', k=3, cost=x/3, coreset_cost=y/4, running_time=2.5, num_iters=10)
#     credit_noPCA.add_new_result('ALG2', k=4, cost=x/4, coreset_cost=y/5, running_time=3.4, num_iters=10)

#     adult_noPCA = Dataset("adult", n=2, m=10, ell=2)
#     x=200
#     y=150
#     adult_noPCA.add_new_result('ALG1', k=2, cost=x/2, coreset_cost=y/3, running_time=1, num_iters=10)
#     adult_noPCA.add_new_result('ALG1', k=3, cost=x/3, coreset_cost=y/4, running_time=1.1, num_iters=10)
#     adult_noPCA.add_new_result('ALG1', k=4, cost=x/4, coreset_cost=y/5, running_time=1.2, num_iters=10)
#     x=210
#     y=100
#     adult_noPCA.add_new_result('ALG2', k=2, cost=x/2, coreset_cost=y/3, running_time=1.2, num_iters=10)
#     adult_noPCA.add_new_result('ALG2', k=3, cost=x/3, coreset_cost=y/4, running_time=1.5, num_iters=10)
#     adult_noPCA.add_new_result('ALG2', k=4, cost=x/4, coreset_cost=y/5, running_time=1.8, num_iters=10)
#     x=250
#     y=135
#     adult_noPCA.add_new_result('ALG2', k=2, cost=x/2, coreset_cost=y/3, running_time=2, num_iters=10)
#     adult_noPCA.add_new_result('ALG2', k=3, cost=x/3, coreset_cost=y/4, running_time=2.5, num_iters=10)
#     adult_noPCA.add_new_result('ALG2', k=4, cost=x/4, coreset_cost=y/5, running_time=3.4, num_iters=10)

#     plot([credit_noPCA, adult_noPCA], 'cost')
#     plot([credit_noPCA, adult_noPCA], 'running_time')
