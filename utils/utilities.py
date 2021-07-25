from random import randint, sample, shuffle
import numpy as np
from utils import cluster_assign
from matplotlib import pyplot as plt
from itertools import cycle, islice

def gen_rand_partition(n,k):
    return [randint(0,k-1) for i in range(n)]

def gen_rand_centers(n,k):
    init = sample(range(n),k)
    shuffle(init)
    return init

def normalize_data(data):
    flags = [False]*data.shape[1]
    for i in range(data.shape[1]):
        data.iloc[:,i] = data.iloc[:,i] - data.iloc[:,i].mean()
        if data.iloc[:,i].std()!=0:
            data.iloc[:,i] = data.iloc[:,i]/data.iloc[:,i].std()
            flags[i] = True
    return data.loc[:,flags]

def compute_cost(data,centers,z):
    n = data.shape[0]
    assign = cluster_assign.cluster_assign(np.asarray(data),np.ones(n),np.asarray([c.cx for c in centers]))
    cost = 0
    for i in range(n):
        cost += (centers[assign[i]].distance(data.iloc[i])**z)
    return cost/n

def Socially_Fair_Clustering_Cost(data,svar,groups,centers,z):
    costs = {}
    for group in groups:
        data_group = data[np.asarray(svar)==group]
        group_cost = compute_cost(data_group,centers,z)
        costs[group] = group_cost
    return costs

def compute_wcost(data,centers,z):
    n = len(data)
    assign = cluster_assign.cluster_assign(np.asarray([x.cx for x in data]),np.ones(n),np.asarray([c.cx for c in centers]))
    cost = 0
    for i in range(n):
        cost += (centers[assign[i]].distance(data[i].cx)**z)
    return cost/sum([p.weight for p in data])

def wSocially_Fair_Clustering_Cost(data,groups,centers,z):
    costs = {}
    for group in groups:
        data_group = [x for x in data if x.group==group]
        group_cost = compute_wcost(data_group,centers,z)
        costs[group] = group_cost
    return costs

def plot(results, y):
    '''
    results: list of Dataset objects; list
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
        
    plt.savefig("./plots/"+dataset.name+'_fig_sociallyFairClustering_'+y+'.png')

def plot_ratios(results, y):
    '''
    results: list of Dataset objects; list
    y: y-axis label; str
    '''
    plt.rcParams["figure.figsize"] = (15,7)
    fig, axs = plt.subplots(1, len(results))
    
    for i, dataset in enumerate(results):
        algorithms = list(set([algo[:5].strip() for algo in dataset.result.keys()]))
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