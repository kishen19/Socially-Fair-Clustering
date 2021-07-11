from sklearn.cluster import KMeans
import time
import numpy as np

from code.utilities import Socially_Fair_Clustering_Cost
from code.classes import Center

def lloyd(data,svar,groups,k,z,num_iters,init_centers):
    ell = len(groups)
    n = data.shape[0]
    num_trials = len(init_centers)
    runtime = 0
    costs = {group:0 for group in groups}
    X = np.asarray(data)
    for trial in range(num_trials):
        incenters = np.asarray([x.cx for x in init_centers[trial]])
        _st = time.time()
        kmeans = KMeans(n_clusters=k,init=incenters,n_init=1,max_iter=100).fit(X)
        _ed = time.time()
        centersi, runtimei = [Center(x,i) for i,x in enumerate(kmeans.cluster_centers_)], _ed-_st
        costsi = Socially_Fair_Clustering_Cost(data,svar,groups,centersi,z)
        runtime += runtimei
        for group in groups:
            costs[group] += costsi[group]
    for group in groups:
        costs[group] /= num_trials
    return costs, runtime/num_trials