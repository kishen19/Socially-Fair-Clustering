import pickle
import numpy as np
from tqdm import tqdm
import time

from utils.classes import Dataset
from utils.utilities import gen_rand_centers
from utils.preprocess import get_data, dataNgen, dataPgen
from code.solve import solve_clustering
from coresets import coresets


def init_dataset(algos, dataset, attr, name, coreset_sizes, num_inits, k, isPCA=False):
    init_centers = []
    flag = "P_k="+str(k) if isPCA else "N"
    
    dataN,svarN,groupsN = get_data(dataset,attr,"N") # Read original data
    data,svar,groups = get_data(dataset,attr,flag) # Read required data
    n = len(data)
    ell = len(groups)

    print("k="+str(k)+": Generating Initial Centers")
    for init_num in range(num_inits):
        mask = gen_rand_centers(n,k)
        centers = np.asarray([data[mask[i],:] for i in range(k)])
        init_centers.append(centers)
    print("k="+str(k)+": Done: Generating Initial Centers")

    resultsk = Dataset(name+"_k="+str(k),dataN,svarN,groupsN,algos)
    if isPCA:
        resultsk.add_PCA_data(data)
    
    print("k="+str(k)+": Generating Coresets")
    for coreset_size in coreset_sizes:
        coreset = []
        weights = []
        coreset_svar = []
        rem = coreset_size
        _coreset_time = 0
        for ind,group in enumerate(groups):
            data_group = data[svar==group]
            coreset_gen = coresets.KMeansCoreset(data_group,n_clusters=k,method="BLK17")
            coreset_group_size = int(data_group.shape[0]*coreset_size/n) if ind<ell-1 else rem
            rem-=coreset_group_size
            _st = time.time()
            coreset_group, weights_group = coreset_gen.generate_coreset(coreset_group_size)
            _ed = time.time()
            _coreset_time += (_ed - _st)
            coreset += list(coreset_group)
            weights += list(weights_group)
            coreset_svar += [group]*coreset_group_size
        resultsk.add_coreset(k,np.asarray(coreset),np.asarray(weights),np.asarray(coreset_svar),_coreset_time)
    print("k="+str(k)+": Done: Generating Coresets")

    for algo in algos:
        for cor_num in range(len(coreset_sizes)):
            for init_num in range(num_inits):
                resultsk.add_new_result(algo,k,cor_num,init_num,0,init_centers[init_num],0)

    f = open("./results/"+dataset+"/" + name+"_k="+str(k) + "_picklefile","wb")
    pickle.dump(resultsk,f)
    f.close()

def main():
    # Parameters:
    dataset = "credit" # "adult"
    attr = "EDUCATION" # "RACE" or "SEX"
    k_vals = range(4,17,2)
    algos = ['Lloyd','Fair-Lloyd','ALGO']
    num_inits = 10
    num_iters = 10
    coreset_sizes = [1000,2000,3000,4000,5000]
    z = 2
    isPCA = False
    
    # Preprocessing datasets
    dataNgen(dataset)
    if isPCA:
        for k in k_vals:
            dataPgen(dataset,k)

    namesuf="_wPCA" if isPCA else "_woPCA"
    name = dataset+"_"+attr+namesuf

    # Initialization
    for k in k_vals:
        init_dataset(algos, dataset, attr, name, coreset_sizes, num_inits, k, isPCA)

    # Run
    for iter in tqdm(range(1,num_iters+1)):
        solve_clustering(dataset,name,k_vals,z,iter)

if __name__=='__main__':
    main()