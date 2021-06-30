from code.preprocess import credit_preprocess
from code.classes import Point,Center,Subspace
from code.algo import ALGO
from coresets import coresets
import pandas as pd
import numpy as np


def solve(dataset,k,z,num_iters):
    if dataset=="credit":
        data, sensitive, groups = credit_preprocess()
    else:
        pass
    
    # Step 1: Compute Coreset
    coreset = []
    for group in groups:
        coreset_gen = coresets.KMeansCoreset(data[np.asarray(sensitive)==group])
        coreset_size = 50 # TODO
        coreset_group, weights = coreset_gen.generate_coreset(coreset_size)
        coreset_group = pd.DataFrame(coreset_group,columns=data.columns)
        coreset += [Point(coreset_group.iloc[i],group,weights[i]) for i in range(coreset_size)]
    
    # Step 2: Fair-Lloyd's Algorithm
    solver = ALGO(coreset,k,z)
    solver.run()


if __name__=='__main__':
    data = "credit"
    k = 10
    z = 2
    num_iters = 100
    solve(data)
