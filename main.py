from code.solve import solve
from code.preprocess import credit_preprocess
from code.utilities import Socially_Fair_Clustering_Cost


def main():
    dataset = "credit"
    k = 10
    z = 2
    num_iters = 10

    if dataset=="credit":
        data, svar, groups = credit_preprocess()
    else:
        pass

    centers, coreset_cost = solve(data,svar,groups,k,z,num_iters)
    cost = Socially_Fair_Clustering_Cost(data,svar,groups,centers,z)
    print("Coreset Cost:",coreset_cost)
    print("Final Cost:",cost)


if __name__=='__main__':
    main()