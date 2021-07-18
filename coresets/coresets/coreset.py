from __future__ import print_function, absolute_import, division
import abc
import numpy as np
from sklearn.utils import check_array, check_random_state
from collections import Counter

from utils import cluster_assign


class Coreset(object):
    """
    Abstract class for coresets.

    Parameters
    ----------
    X : ndarray, shape (n_points, n_dims)
        The data set to generate coreset from.
    w : ndarray, shape (n_points), optional
        The weights of the data points. This allows generating coresets from a
        weighted data set, for example generating coreset of a coreset. If None,
        the data is treated as unweighted and w will be replaced by all ones array.
    random_state : int, RandomState instance or None, optional (default=None)
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, X, n_clusters, w=None, random_state=None,method="FL11"):
        X = check_array(X, accept_sparse="csr", order='C',dtype=[np.float64, np.float32])
        self.X = X
        self.w = w if w is not None else np.ones(X.shape[0])
        self.n_clusters = n_clusters
        self.n_samples = X.shape[0]
        self.random_state = check_random_state(random_state)
        self.method = method
        self.centers = []
        self.eps = 0.1
        self.calc_sampling_distribution()

    @abc.abstractmethod
    def calc_sampling_distribution(self):
        """
        Calculates the coreset importance sampling distribution.
        """
        pass

    def generate_coreset(self, size):
        """
        Generates a coreset of the data set.

        Parameters
        ----------
        size : int
            The size of the coreset to generate.

        """
        if self.method == "FL11":
            assign = cluster_assign.cluster_assign(self.X, self.w, self.centers)
            cnt = Counter(assign)
            while True:
                ind = np.random.choice(self.n_samples, size=size-self.n_clusters, p=self.p, replace=False)
                weights = 1. / ((size-self.n_clusters) * self.p[ind])
                wts = np.asarray([(1+10*self.eps)*cnt[i] for i in range(self.n_clusters)])
                for i in range(size-self.n_clusters):
                    wts[assign[ind[i]]] -= weights[i]
                if np.all(wts>=0):
                    print("Success")
                    return np.append(self.X[ind],np.asarray(self.centers),axis=0), np.append(weights,wts)
                else:
                    print("Fail")
        else:
            ind = np.random.choice(self.n_samples, size=size, p=self.p, replace=False)
            return self.X[ind], 1. / (size * self.p[ind])
