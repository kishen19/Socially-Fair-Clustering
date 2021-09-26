from os import makedirs
import numpy as np
from tqdm import tqdm
import time,datetime,yaml,pickle,json

# Imports relevant to Kmeans++ initialization
from sklearn.cluster import KMeans
from sklearn.utils.extmath import row_norms
from sklearn.utils import check_random_state

from utils.classes import Point, Center, Dataset
from utils.utilities import gen_rand_centers, gen_rand_partition
from utils.preprocess import get_data, dataPgen, dataNgen, dataNCgen
from code.solve import solve
from coresets import coresets

def main():
    # Importing Parameters:
    path = open("./config.yaml", 'r')
    params = yaml.load(path)
    
    # Reading Parameters:
    RUN_NEW = params["RUN_NEW"]
    DT_STRING = params["DT_STRING"]
    DATASET = params["DATASET"]
    ATTR = params["ATTR"]
    Z = params["Z"]
    K_VALS = params["K_VALS"]
    J_VALS = params["J_VALS"]
    NUM_INITS = params["NUM_INITS"]
    NUM_ITERS = params["NUM_ITERS"]
    ALGOS = params["ALGOS"]
    CORESET_SIZES = params["CORESET_SIZES"]
    ISPCA = params["ISPCA"]
    ISKMEANSINIT = params["ISKMEANSINIT"]
    ALGO2_N_SAMPLES = params["ALGO2_N_SAMPLES"]
    ALGO2_SAMPLE_SIZE = params["ALGO2_SAMPLE_SIZE"]
    
    if J_VALS[0] == 0:
        GEN_CORESET = coresets.KMeansCoreset
    else:
        GEN_CORESET = coresets.ProjectiveClusteringCoreset

    # Creating Directories:
    now = datetime.datetime.now()
    dt_string = ATTR+"_"+now.strftime("%d-%m-%Y_%H-%M-%S") if RUN_NEW is True else DT_STRING
    makedirs("./results/" + DATASET + "/" + dt_string, exist_ok=True)

    # Saving Params Info for reference
    f = open("./results/"+DATASET+"/"+dt_string+"/"+ "Params_Info.txt","w")
    f.write(json.dumps(params,indent=4))
    f.close()

    # Dataset Preprocessing
    if DATASET == 'LFW' and 0 not in J_VALS:
        dataNCgen(DATASET)
    else:
        dataNgen(DATASET)
    if ISPCA:
        for k in K_VALS:
            dataPgen(DATASET,k)

    NAMESUF = "_wPCA" if ISPCA else "_woPCA"
    NAME = ATTR + NAMESUF

    # Data and Dataset objects initialization
    if DATASET == 'LFW' and 0 not in J_VALS:
        dataN, dataGC, groups = get_data(DATASET,ATTR,"NC") # Get original data
    else:
        dataN, dataGC, groups = get_data(DATASET,ATTR,"N") # Get original data
    if RUN_NEW is True:
        results = Dataset(DATASET, NAME, dt_string, dataN, dataGC, groups, ALGOS)
        for k in K_VALS:
            if DATASET == 'LFW' and J > 0:
                flag = "P_k="+str(k) if ISPCA else "NC"
            else:
                flag = "P_k="+str(k) if ISPCA else "N"
            data,dataGC,groups = get_data(DATASET,ATTR,flag) # Get required data
            n = len(data)
            ell = len(groups)

            # this is tentative, discuss about initializing with partitions
            init_centers_partitions = []
            print("k="+str(k)+": Generating Initializations")
            if 0 in J_VALS:
                if ISKMEANSINIT:
                    X = np.asarray([x.cx for x in data])
                    x_squared_norms = row_norms(X, squared=True)
                    kmeans = KMeans(k)
                    random_state = check_random_state(0)
                    for init_num in range(NUM_INITS):
                        centers = kmeans._init_centroids(X, init = 'k-means++', x_squared_norms=x_squared_norms, random_state=random_state)
                        centers = [Center(centers[i],i) for i in range(k)]
                        init_centers_partitions.append(centers)
                else:
                    for init_num in range(NUM_INITS):
                        mask = gen_rand_centers(n,k)
                        centers = [Center(data[mask[i]].cx,i) for i in range(k)]
                        init_centers_partitions.append(centers)
            else:
                for init_num in range(NUM_INITS):
                    mask = gen_rand_partition(n,k)
                    init_centers_partitions.append(mask)
            print("k="+str(k)+": Done: Generating Initial Centers")

            if ISPCA:
                results.add_PCA_data(data,k)
            
            # For ALGO
            if "ALGO" in ALGOS:
                print("k="+str(k)+": Generating Coresets")
                for coreset_size in CORESET_SIZES:
                    coreset = []
                    rem = coreset_size
                    _coreset_time = 0
                    for ind,group in enumerate(groups):
                        data_group = [x.cx for x in data if x.group == group]
                        coreset_gen = GEN_CORESET(data_group,n_clusters=k)
                        coreset_group_size = int(len(data_group)*coreset_size/n) if ind<ell-1 else rem
                        rem-=coreset_group_size
                        _st = time.time()
                        coreset_group, weights_group = coreset_gen.generate_coreset(coreset_group_size)
                        _ed = time.time()
                        _coreset_time += (_ed - _st)
                        coreset += [Point(coreset_group[i],group,weights_group[i]) for i in range(coreset_group_size)]
                    results.add_coreset(k,coreset,_coreset_time)
                print("k="+str(k)+": Done: Generating Coresets")

            for J in J_VALS:
                if "ALGO" in ALGOS:
                    for cor_num in range(len(CORESET_SIZES)):
                        for init_num in range(NUM_INITS):
                            results.add_new_result("ALGO",k,J,cor_num,init_num,0,init_centers_partitions[init_num],0)
                
                # For other algorithms
                for algo in ALGOS:
                    if algo != 'ALGO':
                        for init_num in range(NUM_INITS):
                            results.add_new_result(algo,k,J,0,init_num,0,init_centers_partitions[init_num],0)
    else:
        f = open("./results/" + DATASET + "/" + dt_string + "/" + NAME + "_iters=0","rb")
        results = pickle.load(f)
        f.close()    

    f = open("./results/" + DATASET + "/" + dt_string + "/" + NAME + "_iters=0","wb")
    pickle.dump(results,f)
    f.close()

    # Start Code Execution
    for iter in tqdm(range(1,NUM_ITERS+1)):
        print(iter)
        solve(iter, DATASET, dt_string, NAME, K_VALS, Z, J_VALS, ALGO2_N_SAMPLES, ALGO2_SAMPLE_SIZE)
    
    print()
    print("Paste the following in config.yaml file")
    print(dt_string)
    
if __name__=='__main__':
    main()