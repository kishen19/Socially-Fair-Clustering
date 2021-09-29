import kmedoids
import numpy as np
from tqdm import tqdm


def get_dist_matrix(data):
    n = len(data)
    distmatrix = np.zeros([n,n])
    for i in tqdm(range(n-1)):
        for j in range(i+1,n):
            dist = np.linalg.norm(data[i].cx-data[j].cx)
            distmatrix[i,j] = dist
            distmatrix[j,i] = dist
    return distmatrix

def solve_kmedian_clustering(data,distmatrix,k,d,ell,z,centers):
    # centers is a mask array
    out = kmedoids.fasterpam(distmatrix,np.asarray([c.index for c in centers]),max_iter=1)
    print(out.medoids,out.loss)
    return out.medoids, out.loss