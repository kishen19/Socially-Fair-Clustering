from sklearn.cluster import KMeans
import time
import numpy as np

from code.utilities import Socially_Fair_Clustering_Cost
from code.classes import Center

def lloyd(dataset,data,svar,groups,k,z,num_iters,init_centers,q):
    ell = len(groups)
    n = data.shape[0]
    num_inits = len(init_centers)
    runtimes = [0]*num_inits
    costs = [{group:0 for group in groups} for i in range(num_inits)]
    X = np.asarray(data)
    for init in range(num_inits):
        incenters = np.asarray([x.cx for x in init_centers[init]])
        _st = time.time()
        kmeans = KMeans(n_clusters=k,init=incenters,n_init=1,max_iter=100).fit(X)
        _ed = time.time()
        centersi, runtimei = [Center(x,i) for i,x in enumerate(kmeans.cluster_centers_)], _ed-_st
        costsi = Socially_Fair_Clustering_Cost(data,svar,groups,centersi,z)
        runtimes[init] = runtimei
        for group in groups:
            costs[init][group] = costsi[group]
    
    for init in range(num_inits):
        for group in groups:
            q.put([dataset,"Lloyd"+" ("+ groups[group] + ")",k,init,num_iters,costs[init][group],runtimes[init],0,0])