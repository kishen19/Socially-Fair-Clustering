from sklearn.cluster import KMeans
import time
import numpy as np
from tqdm import tqdm
import pickle

from utils.classes import Center

def lloyd(dataset,name,k,z,num_iters):
    f = open("./results/"+dataset+"/" + name+"_k="+str(k) + "_picklefile","rb")
    results = pickle.load(f)
    f.close()
    n,d,ell = results.get_params()
    algos = [algo for algo in results.result if algo[:5]=="Lloyd"]
    X = np.asarray(results.get_data())
    for init in results.result[algos[0]][k][0]:
        if results.result[algos[0]][k][0][init]["num_iters"] < num_iters:
            centers = np.asarray([x.cx for x in results.result[algos[0]][k][0][init]["centers"]])
            _st = time.time()
            kmeans = KMeans(n_clusters=k,init=centers,n_init=1,max_iter=num_iters-results.result[algos[0]][k][0][init]["num_iters"]).fit(X)
            _ed = time.time()
            new_centers, runtime = [Center(x,i) for i,x in enumerate(kmeans.cluster_centers_)], _ed-_st
            for algo in algos:
                results.result[algo][k][0][init]["running_time"] += runtime
                results.result[algo][k][0][init]["centers"] = new_centers
                results.result[algo][k][0][init]["num_iters"] = num_iters
    f = open("./results/"+dataset+"/" + name+"_k="+str(k) + "_picklefile","wb")
    pickle.dump(results,f)
    f.close()