import pandas as pd
import numpy as np

from code.classes import Point,Center,Subspace,Affine
from code.algo import ALGO
from coresets import coresets
from code.utilities import gen_rand_partition


def solve(data,svar,groups,k,z,num_iters):
    ell = len(groups)
    # Step 1: Compute Coreset
    coreset = []
    for group in groups:
        data_group = data[np.asarray(svar)==group]
        coreset_gen = coresets.KMeansCoreset(data_group)
        coreset_size = data_group.shape[0]//10
        print(coreset_size)
        coreset_group, weights = coreset_gen.generate_coreset(coreset_size)
        coreset_group = pd.DataFrame(coreset_group,columns=data.columns)
        coreset += [Point(coreset_group.iloc[i],group,weights[i]) for i in range(coreset_size)]

    # Step 2: Fair-Lloyd's Algorithm
    n = len(coreset)
    solver = ALGO(coreset,k,ell,z)
    solver.run(num_iters,gen_rand_partition(n,k))
    return solver.centers, solver.cost