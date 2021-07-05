import scipy
import numpy as np

from coresets.coresets import Coreset
from coresets.coresets import sensitivity


class ProjectiveClusteringCoreset(Coreset):
    """
    Class for generating Subspace Approximation coreset based on the sensitivity framework
    with importance sampling [1].

    Parameters
    ----------
        X : ndarray, shape (n_points, n_dims)
            The data set to generate coreset from.
        w : ndarray, shape (n_points), optional
            The weights of the data points. This allows generating coresets from a
            weighted data set, for example generating coreset of a coreset. If None,
            the data is treated as unweighted and w will be replaced by all ones array.
        n_clusters : int
            Number of clusters used for the initialization step.
        random_state : int, RandomState instance or None, optional (default=None)

    References
    ----------
        [1] 
    """

    def __init__(self, X, w=None, n_clusters=10, J=10, random_state=None):
        self.n_clusters = n_clusters
        self.J = J
        super(ProjectiveClusteringCoreset, self).__init__(X, w, random_state)

    def calc_sampling_distribution(self):
        sens = []
        for p in self.X:
            sens.append(self.calc_sens(self.X,p,self.J))
        sens = np.asarray(sens)
        self.p = sens/np.sum(sens)

    def sorted_eig(self,A):
        eig_vals, eig_vecs =scipy.linalg.eigh(A)  	
        eig_vals_sorted = np.sort(eig_vals)[::-1]
        eig_vecs = eig_vecs.T
        eig_vecs_sorted = eig_vecs[eig_vals.argsort()][::-1]
        return eig_vals_sorted,eig_vecs_sorted

    def get_unitary_matrix(self,n,m):
        a = np.random.random(size=(n, m))
        q, _ = np.linalg.qr(a)
        return q
        
    def get_gamma(self,A_tag,l,d):
        vals, _ = self.sorted_eig(A_tag)
        sum_up = 0;sum_down = 0
        for i in range (l) : 
            sum_up += vals[d-i-1]
            sum_down += vals[i]
        return sum_up/sum_down

    def calc_sens(self,A,p,j,eps=0.1):	
        d = A.shape[1]
        l = d-j
        A_tag = np.dot(A.T , A)
        p = np.reshape(p, (p.shape[0], 1)).T
        p_tag = np.dot(p.T,p)
        x = self.get_unitary_matrix(d, l)
        gamma = self.get_gamma(A_tag,l,d)
        stop_rule = (gamma*eps)/(1-gamma)
        s_l = -np.inf
        s_old = 0
        step = 0
        while step < 1000:
            s_new = np.trace(np.dot(np.dot(x.T,p_tag), x))/np.trace(np.dot(np.dot(x.T, A_tag), x))
            s_l = max(s_l,s_new)
            G = p_tag - s_new*A_tag
            _ , ev = self.sorted_eig(G)
            x = ev[:l].T
            if s_new - stop_rule < s_old:
                return s_l
            s_old = s_new
            step += 1
        return s_l