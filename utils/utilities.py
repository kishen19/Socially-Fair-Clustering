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
        
    plt.savefig("./plots/"+dataset.name+'_fig_sociallyFairClustering_'+y+'.png')

def plot_ratios(results, y):
    '''
    results: list of Dataset objects; list
    y: y-axis label; str
    '''
    plt.rcParams["figure.figsize"] = (7,7)
    fig, axs = plt.subplots(1, len(results))
    if len(results)==1:
        axs = [axs]
    for i, dataset in enumerate(results):
        algorithms = [[algo for algo in dataset.result if algo[:4]=="ALGO"][0]]+[[algo for algo in dataset.result if algo[:5]=="Lloyd"][0]]
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
            axs[i].plot(k, val, color=colors[j], markersize=10, marker=markers[j], fillstyle='none', label=alg[:5])
            axs[i].set_ylim(ymin=1,ymax=max(val)+1)
            axs[i].legend(loc='upper right')
            axs[i].set_xlabel('('+chr(i+97)+')\t\t\t$k$\t\t\t')
            axs[i].set_title(dataset.name+' dataset')
        if i==0:
            axs[i].set_ylabel(y)
        
    plt.savefig("./plots/"+dataset.name+'_fig_sociallyFairClustering_'+y+'.png')