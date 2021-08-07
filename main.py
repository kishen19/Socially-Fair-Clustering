import pickle
import numpy as np
from tqdm import tqdm
import time

from utils.classes import Point, Center, Dataset
from utils.utilities import gen_rand_centers
from utils.preprocess import get_data, dataNgen, dataPgen
from code.solve import solve_clustering
from coresets import coresets

from sklearn.cluster import KMeans
from sklearn.utils.extmath import row_norms
from sklearn.utils import check_random_state

def init_dataset(algos, dataset, attr, name, coreset_sizes, num_inits, k, isPCA=False):
    init_centers = []
    flag = "P_k="+str(k) if isPCA else "N"
    
    dataN,groupsN = get_data(dataset,attr,"N") # Read original data
    data,groups = get_data(dataset,attr,flag) # Read required data
    n = len(data)
    ell = len(groups)

    print("k="+str(k)+": Generating Initial Centers")
    # for init_num in range(num_inits):
    #     mask = gen_rand_centers(n,k)
    #     centers = [Center(data[mask[i]].cx,i) for i in range(k)]
    #     init_centers.append(centers)
    X = np.asarray([x.cx for x in data])
    init = 'k-means++'
    x_squared_norms = row_norms(X, squared=True)
    kmeans = KMeans(k)
    random_state = check_random_state(0)
    for init_num in range(num_inits):
        centers = kmeans._init_centroids(X, init = init, x_squared_norms=x_squared_norms, random_state=random_state)
        centers = [Center(centers[i],i) for i in range(k)]
        init_centers.append(centers)
    print("k="+str(k)+": Done: Generating Initial Centers")

    resultsk = Dataset(name+"_k="+str(k),dataN,groupsN,algos)
    if isPCA:
        resultsk.add_PCA_data(data)
    
    # For ALGO
    if "ALGO" in algos:
        print("k="+str(k)+": Generating Coresets")
        for coreset_size in coreset_sizes:
            coreset = []
            rem = coreset_size
            _coreset_time = 0
            for ind,group in enumerate(groups):
                data_group = [x.cx for x in data if x.group == group]
                coreset_gen = coresets.KMeansCoreset(data_group,n_clusters=k,method="BLK17")
                coreset_group_size = int(len(data_group)*coreset_size/n) if ind<ell-1 else rem
                rem-=coreset_group_size
                _st = time.time()
                coreset_group, weights_group = coreset_gen.generate_coreset(coreset_group_size)
                _ed = time.time()
                _coreset_time += (_ed - _st)
                coreset += [Point(coreset_group[i],group,weights_group[i]) for i in range(coreset_group_size)]
            resultsk.add_coreset(k,np.asarray(coreset),_coreset_time)
        print("k="+str(k)+": Done: Generating Coresets")

        for cor_num in range(len(coreset_sizes)):
            for init_num in range(num_inits):
                resultsk.add_new_result("ALGO",k,cor_num,init_num,0,init_centers[init_num],0)
    # For other algorithms
    for algo in algos:
        if algo != 'ALGO':
            for init_num in range(num_inits):
                resultsk.add_new_result(algo,k,0,init_num,0,init_centers[init_num],0)

    f = open("./results/"+dataset+"/" + name+"_k="+str(k) + "_picklefile","wb")
    pickle.dump(resultsk,f)
    f.close()

def main():
    # Parameters:
    dataset = "credit" # "adult"
    attr = "EDUCATION" # "RACE" or "SEX"
    k_vals = range(4,17,2)
    algos = ['Lloyd','Fair-Lloyd','ALGO3']#,'ALGO']
    num_inits = 20
    num_iters = 20
    coreset_sizes = [1000,1000,1000,2000,2000,2000,3000,3000,3000]
    z = 2
    isPCA = False
    # ALGO2 related parameters
    n_samples = 5
    sample_size = 3000
    
    # Preprocessing datasets
    dataNgen(dataset)
    if isPCA:
        for k in k_vals:
            dataPgen(dataset,k)

    namesuf="_wPCA" if isPCA else "_woPCA"
    name = dataset+"_"+attr+namesuf

    # # Initialization
    # for k in k_vals:
    #     init_dataset(algos, dataset, attr, name, coreset_sizes, num_inits, k, isPCA)

    # Run
    # for iter in tqdm(range(1,num_iters+1)):
    for iter in tqdm(range(100,200)):
        solve_clustering(dataset,name,k_vals,z,iter,n_samples,sample_size)
    
if __name__=='__main__':
    main()