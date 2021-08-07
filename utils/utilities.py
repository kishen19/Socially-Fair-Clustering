from random import randint, sample, shuffle
import numpy as np
from matplotlib import pyplot as plt
from itertools import cycle, islice

from utils import cluster_assign

def gen_rand_partition(n,k):
    return [randint(0,k-1) for i in range(n)]

def gen_rand_centers(n,k):
    init = sample(range(n),k)
    shuffle(init)
    return init

def distance(a,b):
    return np.linalg.norm(np.asarray(a)-np.asarray(b))

def compute_cost(data,centers,z):
    n = len(data)
    assign = cluster_assign.cluster_assign(np.asarray([x.cx for x in data]),np.asarray([c.cx for c in centers]))
    cost = 0
    tot = 0
    for i in range(n):
        cost += data[i].weight*(distance(centers[assign[i]].cx, data[i].cx)**z)
        tot += data[i].weight
    return cost/tot

def Socially_Fair_Clustering_Cost(data,groups,centers,z):
    costs = {}
    for group in groups:
        group_cost = compute_cost([x for x in data if x.group == group],centers,z)
        costs[groups[group]] = group_cost
    return costs

def plot(results, y):
    '''
    results: list of Dataset objects; list
    y: y-axis label; str
    '''
    plt.rcParams["figure.figsize"] = (7,7)
    fig, axs = plt.subplots(1, len(results))
    if len(results)==1:
        axs = [axs]
    # colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
    #                                         '#f781bf', '#a65628', '#984ea3',
    #                                         '#999999', '#e41a1c', '#dede00']),
    #                                 int(1000 + 1))))
    colors = np.array(list(islice(cycle(['royalblue', 'tab:red', 'black',
                                            'green', 'yellow']),
                                    int(1000 + 1))))                            
    markers = np.array(list(islice(cycle(['s', '^', '.',
                                            '|', 'o', 'x',
                                            '>', '<', 'p']),
                                    int(3 + 1))))
    
    linestyles = np.array(list(islice(cycle(['dotted', 'dashed', 'solid', 'dashdot']),
                                    int(3 + 1))))
        
    for i, dataset in enumerate(results):
        algorithms = dataset.result.keys()
        for j, algo in enumerate(algorithms):
            k_vals, output, groups = dataset.k_vs_val(algo, y)
            if y == 'cost' or y == 'coreset_cost':
                for group, name in enumerate(groups):
                    axs[i].plot(k_vals[group], output[group], color=colors[j], markersize=10, marker=markers[group], fillstyle='none', linestyle=linestyles[j], linewidth=1.5, label=algo+" ("+name+")")
            else:
                axs[i].plot(k_vals[0], output[0], color=colors[j], markersize=10, marker=markers[j], fillstyle='none', linestyle=linestyles[j],linewidth=1.5,  label=algo)
            axs[i].set_xlabel('$k$')
            axs[i].set_title(dataset.name+' dataset')
            if i==0:
                axs[i].set_ylabel(y)
            axs[i].legend(loc='upper right')
                
    plt.savefig("./plots/"+dataset.name+'_fig_sociallyFairClustering_'+y+'.png')