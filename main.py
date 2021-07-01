from code.preprocess import credit_preprocess
from code.classes import Point,Center,Subspace,Affine
from code.algo import ALGO
from coresets import coresets
import pandas as pd
import numpy as np


def solve(dataset,k,z,num_iters):
    if dataset=="credit":
        data, sensitive, groups = credit_preprocess()
    else:
        pass
    
    ell = len(groups)
    # Step 1: Compute Coreset
    coreset = []
    for group in groups:
        data_group = data[np.asarray(sensitive)==group]
        coreset_gen = coresets.KMeansCoreset(data_group)
        coreset_size = data_group.shape[0]//5
        print(coreset_size)
        coreset_group, weights = coreset_gen.generate_coreset(coreset_size)
        coreset_group = pd.DataFrame(coreset_group,columns=data.columns)
        coreset += [Point(coreset_group.iloc[i],group,weights[i]) for i in range(coreset_size)]
        val1 = 0
        val2 = 0
        for i in range(data_group.shape[0]):
            val1 += np.linalg.norm(data_group.iloc[i])**2

        for i in range(coreset_size):
            val2 += weights[i]*np.linalg.norm(coreset_group.iloc[i])**2

        print(np.sqrt(val1),np.sqrt(val2))
    # Step 2: Fair-Lloyd's Algorithm
    solver = ALGO(coreset,k,ell,z)
    solver.run(num_iters)
    print(np.sqrt(np.asarray(solver.costs)))
    for group in groups:
        data_group = data[np.asarray(sensitive)==group]
        val1 = 0
        for i in range(data_group.shape[0]):
            best = 10**9+7
            for j in range(k):
                best = min(best,np.linalg.norm(data_group.iloc[i]-solver.centers[j].cx)**2)
            val1+=best
        print(np.sqrt(val1))


if __name__=='__main__':
    dataset = "credit"
    k = 2
    z = 2
    num_iters = 5
    solve(dataset,k,z,num_iters)
