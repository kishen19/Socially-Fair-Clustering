import pylab
from cvxopt import solvers, matrix, spmatrix, mul, div
import numpy as np

import time
import warnings
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice


def clustering(k, d, ell, P, wts):

    # ~~~ parameters ~~~
    # k:    the number of clusters; python float
    # d:    dimension; python float
    # ell:  number of groups; python int
    # P:    partition of the points; numpy 2D array; P[j][i] gives P_ji in the form of another numpy array
    #       of dimension |P_ji| x d

    # ~~~ convex program ~~~
    #     minimize    (0.c_1 + ... + 0.c_k) + t
    #     subject to  \sum_{i \in [k]} \sum_{x \in P_{ji}} \norm{x - c_i}^2 <= t, forall j \in [\ell].
    #                       (non-linear constraints)
    # k+1 Variables:  c_1, ..., c_k and one scalar t.
    #
    # c_1,...,c_k:  k vectors in \R^d; centers.
    # t: one scalar; socially fair clustering cost.
    
    # ~~~ in matrix form to run cvxopt solver ~~~
    # the coefficients are 0 for all the coordinates of all the centers,
    # and 1 for t.
    c = matrix(k*d*[0.0] + [1.0]) 
    # in this case, n = k*d+1, i.e., we treat x as a vector of size k*d+1.

    def F(x=None, z=None):
        if x is None:  
            return ell, matrix(0.0, (k*d+1, 1)) # consider changing this x0
        # note that anything is in the domain of f.

        # converting matrix x of shape k*d+1 to numpy array x_np of shape (k,d) ignoring the last element of x
        x_np = np.array(x[:-1])
        x_np = x_np.reshape((k,d))
        # clustering cost minus t: \sum_{i \in [k]} \sum_{p \in P_{ji}} \norm{p - c_i}^2 - t.
        # using p for points to avoid confusion with x, the variables.
        # here f is ell dimensional
        f = matrix(0.0, (ell, 1))
         
        for j in range(ell):
            f[j] = -x[-1] # minus t
            for i in range(k):
                for p in range(len(P[j][i])):
                    f[j] += wts[j][i][p]*np.linalg.norm(P[j][i][p] - x_np[i])**2 ## change here for k,z clustering with z > 2


        # computing gradients w.r.t. the centers
        Df = matrix(0.0, (ell,k*d+1))
        for j in range(ell):
            for i in range(k):
                sum_p_ji = sum([wts[j][i][p]*P[j][i][p] for p in range(len(P[j][i]))]) # sum of the points belonging to jth group in the ith cluster
                sum_w_ji = sum([wts[j][i][p] for p in range(len(P[j][i]))])
                Df[j, i*d:(i+1)*d] = 2*(sum_w_ji*x_np[i] - sum_p_ji) ## change here for k,z clustering with z > 2
        Df[:,-1] = -1.0 # gradient w.r.t. the variable t.
        if z is None: return f, Df

        
        # H = z_0*Dsr_0 + z_1*Dsr_1 + ... + z_{ell-1}*Dsr_{ell-1}
        H = matrix(0.0, (k*d+1, k*d+1))
        
        for j in range(ell):
            Dsr = matrix(0.0, (k*d+1, k*d+1))
            for i in range(k):
                sum_w_ji = sum([wts[j][i][p] for p in range(len(P[j][i]))])
                Dsr[i*d:(i+1)*d, i*d:(i+1)*d] = 2*sum_w_ji*np.eye(d)
            Dsr[ -1,-1] = 0.0 # double derivate w.r.t. t is 0
            H += z[j]*Dsr
            
        return f, Df, H

    
    # solve and return c_1, ..., c_k, t
    
    sol = solvers.cpl(c, F)
    centers = np.array(sol['x'][:-1])
    val = sol['x'][-1]
    centers = centers.reshape((k,d))
    return  centers, val









'''
####################### TOY DATASETS ########################

np.random.seed(0)

# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 1000
ell = 2
k = 2
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5],
                             random_state=random_state)

# # ============
# # Set up cluster parameters
# # ============
plt.figure(figsize=(16, 13))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.95, wspace=.05,
                    hspace=.01)

plot_num = 1

default_base = {'quantile': .3,
                'eps': .3,
                'damping': .9,
                'preference': -200,
                'n_neighbors': 10,
                'n_clusters': 3,
                'min_samples': 20,
                'xi': 0.05,
                'min_cluster_size': 0.1}

datasets = [
    (noisy_circles, {'damping': .77, 'preference': -240,
                     'quantile': .2, 'n_clusters': 2,
                     'min_samples': 20, 'xi': 0.25}),
    (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
    (varied, {'eps': .18, 'n_neighbors': 2,
              'min_samples': 5, 'xi': 0.035, 'min_cluster_size': .2}),
    (aniso, {'eps': .15, 'n_neighbors': 2,
             'min_samples': 20, 'xi': 0.1, 'min_cluster_size': .2}),
    ('gaussians', {'mean1': 2.0, 'mean2': -2.0})
             ]

for i_dataset, (dataset, algo_params) in enumerate(datasets):
    # update parameters with dataset-specific values
    params = default_base.copy()
    params.update(algo_params)

    g = []
    q = []
    if 'gaussians' in dataset:
        X = []
        for i in range(int(n_samples/3)):
            X.append(np.random.multivariate_normal(mean=[20.0,0.0],cov=np.eye(2)))
            q.append(0)
            g.append(0)
        for i in range(int(n_samples/3)):
            X.append(np.random.multivariate_normal(mean=[-20.0,0.0],cov=np.eye(2)))
            q.append(1)
            g.append(1)
        for i in range(int(n_samples/3)):
            X.append(np.random.multivariate_normal(mean=[0.0,0.0],cov=np.eye(2)))
            q.append(2)
            g.append(2)
        X = np.asarray(X)
        ell = 3
        k=3



    else:
        X, y = dataset
        # creating another variable g that represents the group membership. Groups are 0 or 1.
        g = np.random.randint(ell, size=len(y))
        # g = np.array([0,1,1,0])

        # creating an initial random clusters.
        q = np.random.randint(k, size=len(y))
        # q = np.array([1,1,0,0])
    
    # X = np.array([[2,1], [1,1], [-3,0],[-2,1]])
    d = X.shape[-1]

    # normalize dataset for easier parameter selection
    # X = StandardScaler().fit_transform(X)


    

    # partitioning the data based on groups and clusters
    partition = []
    for j in range(ell):
        partition.append([])
    for j in range(ell):
        for i in range(k):
            partition[j].append([])
    par = np.asarray(partition)
    print(par.shape)

    for p in range(len(X)):
        partition[g[p]][q[p]].append(X[p])
    
    # for j in range(ell):
    #     for i in range(k):
    #         print(len(partition[j][i]))

    cvxopt_centers, opt = clustering(k, d, ell, partition)






#     # estimate bandwidth for mean shift
#     bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])

#     # connectivity matrix for structured Ward
#     connectivity = kneighbors_graph(
#         X, n_neighbors=params['n_neighbors'], include_self=False)
#     # make connectivity symmetric
#     connectivity = 0.5 * (connectivity + connectivity.T)

#     # ============
#     # Create cluster objects
#     # ============
#     # ============
#     ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    # two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
#     ward = cluster.AgglomerativeClustering(
#         n_clusters=params['n_clusters'], linkage='ward',
#         connectivity=connectivity)
    spectral = cluster.SpectralClustering(
        n_clusters=params['n_clusters'], eigen_solver='arpack',
        affinity="nearest_neighbors")
#     dbscan = cluster.DBSCAN(eps=params['eps'])
#     optics = cluster.OPTICS(min_samples=params['min_samples'],
#                             xi=params['xi'],
#                             min_cluster_size=params['min_cluster_size'])
#     affinity_propagation = cluster.AffinityPropagation(
#         damping=params['damping'], preference=params['preference'])
#     average_linkage = cluster.AgglomerativeClustering(
#         linkage="average", affinity="cityblock",
#         n_clusters=params['n_clusters'], connectivity=connectivity)
#     birch = cluster.Birch(n_clusters=params['n_clusters'])
#     gmm = mixture.GaussianMixture(
#         n_components=params['n_clusters'], covariance_type='full')

    clustering_algorithms = (
        ('Spectral\nClustering', spectral),
        ('Convex\nProgram', cvxopt_centers)
    )
    # clustering_algorithms = (
    #     ('MiniBatch\nKMeans', two_means),
    #     ('Affinity\nPropagation', affinity_propagation),
    #     ('MeanShift', ms),
    #     ('Spectral\nClustering', spectral),
    #     ('Ward', ward),
    #     ('Agglomerative\nClustering', average_linkage),
    #     ('DBSCAN', dbscan),
    #     ('OPTICS', optics),
    #     ('BIRCH', birch),
    #     ('Gaussian\nMixture', gmm)
    # )

    for name, algorithm in clustering_algorithms:
        t0 = time.time()

        # catch warnings related to kneighbors_graph
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the " +
                "connectivity matrix is [0-9]{1,2}" +
                " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning)
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding" +
                " may not work as expected.",
                category=UserWarning)
            if 'Spectral' in name:
                algorithm.fit(X)

            

                t1 = time.time()
                if hasattr(algorithm, 'labels_'):
                    y_pred = algorithm.labels_.astype(int)
                else:
                    y_pred = algorithm.predict(X)
            else:
                t1 = time.time()
                y_pred = q

        plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name, size=18)

        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_pred) + 1))))
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])
        
        if 'Convex' in name:
            plt.scatter(cvxopt_centers[:,0], cvxopt_centers[:,1], s=40, color='black')

        # plt.xlim(-25, 25)
        # plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
        plot_num += 1

plt.savefig(
'figcluster.png'
)


'''