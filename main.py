from code.solve import solve_clustering, solve_projective_linear
from code.preprocess import credit_preprocess,LFW_preprocess
from code.utilities import Socially_Fair_Clustering_Cost


def main():
    dataset = "credit"
    k = 1
    z = 2
    J = 10
    num_iters = 1

    if dataset=="credit":
        data, svar, groups = credit_preprocess()
    elif dataset=="LFW":
        data, svar, groups = LFW_preprocess()

    # centers, coreset_cost = solve_clustering(data,svar,groups,k,z,num_iters)
    centers, coreset_cost = solve_projective_linear(data,svar,groups,k,J,z,num_iters)
    cost = Socially_Fair_Clustering_Cost(data,svar,groups,centers,z)
    print("Coreset Cost:",coreset_cost)
    print("Final Cost:",cost)

if __name__=='__main__':
    main()