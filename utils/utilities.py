from random import randint, sample, shuffle
from os import makedirs
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

def assign_subspace(data,centers):
    assign = [-1 for i in range(len(data))]
    for i,x in enumerate(data):
        best = np.inf
        for j,center in enumerate(centers):
            dist = center.distance(x)
            if best < dist:
                assign[i] = j
                best = dist
    return np.asarray(assign)

def compute_cost(data,centers,J,z):
    n = len(data)
    if J==0:
        assign = cluster_assign.cluster_assign(np.asarray([x.cx for x in data]),np.asarray([c.cx for c in centers]))
    else:
        assign = assign_subspace(data,centers)
    cost = 0
    tot = 0
    for i in range(n):
        cost += data[i].weight*(centers[assign[i]].distance(data[i])**z)
        tot += data[i].weight
    return cost/tot

def Socially_Fair_Clustering_Cost(data,groups,centers,J,z):
    costs = {}
    for group in groups:
        group_cost = compute_cost([x for x in data if x.group == group],centers,J,z)
        costs[groups[group]] = group_cost
    return costs

def plot(results, y, param="k"):
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
                                    int(5 + 1))))
    
    markersize = np.array(list(islice(cycle([12,10,14]),
                                    int(5 + 1))))
    linestyles = np.array(list(islice(cycle(['dotted', 'solid', 'dashed', 'dashdot']),
                                    int(5 + 1))))
        
    for i, dataset in enumerate(results):
        algorithms = dataset.result.keys()
        for j, algo in enumerate(algorithms):
            param_vals, output, groups = dataset.k_vs_val(algo, y) if param=="k" else dataset.J_vs_val(algo, y)
            if y == 'cost' or y == 'coreset_cost':
                for group, name in enumerate(groups):
                    axs[i].plot(param_vals[group], output[group], color=colors[j], markersize=markersize[j], marker=markers[group], fillstyle='none', linestyle=linestyles[j], linewidth=1.5, label=algo+" ("+name+")")
            else:
                axs[i].plot(param_vals[0], output[0], color=colors[j], markersize=markersize[j], marker=markers[j], fillstyle='none', linestyle=linestyles[j],linewidth=1.5,  label=algo)
            axs[i].set_xlabel('$'+param+'$')
            axs[i].set_title(dataset.name+' dataset')
            if i==0:
                axs[i].set_ylabel(y)
            axs[i].legend(loc='upper right')
    makedirs("./plots/"+  dataset.dataset + "/" + dataset.dt_string,exist_ok=True)
    plt.savefig("./plots/"+  dataset.dataset + "/" + dataset.dt_string + "/" + dataset.name +'_'+param+"_vs_"+y+'.png')