from cvxopt import solvers, matrix
import cvxpy as cp
import numpy as np

# %env JAX_ENABLE_X64=1
# %env JAX_PLATFORM_NAME=cpu

# import autograd.numpy as jnp
# from autograd import grad

# import jax
# import jax.numpy as jnp
# from jax import grad
# from jax.ops import index, index_add, index_update

def kzclustering(data, k, d, ell, q, centers):

    # ~~~ parameters ~~~
    # k:    the number of clusters; python float
    # d:    dimension; python float
    # ell:  number of groups; python int
    # data: List of Point Objects
    # Let P_{ji} be points of group j in cluster i

    # ~~~ convex program ~~~
    #     minimize    (0.c_1 + ... + 0.c_k) + t
    #     subject to  (\sum_{i \in [k]} \sum_{x \in P_{ji}} wt(x)*\norm{x - c_i}^2)/(\sum_{x \in P_{j}} wt(x)) <= t, forall j \in [\ell].
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
    wts = np.zeros([ell,k])
    group_costs = np.zeros(ell)
    for p in data:
        if centers is not None:
            group_costs[p.group] += p.weight*(np.linalg.norm(p.cx-centers[p.cluster].cx)**q)
        else:
            group_costs[p.group] += p.weight*(np.linalg.norm(p.cx)**q)
        wts[p.group,p.cluster] += p.weight
    
    for j in range(ell):
        group_costs[j]/=np.sum(wts[j])

    val = max(group_costs)
    def F(x=None, z=None):
        if x is None:  
            x0 = matrix(0.0, (k*d+1, 1))
            if centers is not None:
                for i in range(k):
                    x0[i*d:(i+1)*d] = centers[i].cx
            x0[-1] = val
            return ell, x0  # consider changing this x0
        # note that anything is in the domain of f.

        # converting matrix x of shape k*d+1 to numpy array x_np of shape (k,d) ignoring the last element of x
        x_np = np.array(x[:-1])
        x_np = x_np.reshape((k,d))
        # clustering cost minus t: (\sum_{i \in [k]} \sum_{p \in P_{ji}} wt(p)*\norm{p - c_i}^2)/(\sum_{p \in P_j} wt(p)) - t.
        # using p for points to avoid confusion with x, the variables.
        # here f is ell dimensional
        f = matrix(0.0, (ell, 1))
        wts = [[0 for i in range(k)] for j in range(ell)]
        for p in data:
            f[int(p.group)] += p.weight*np.linalg.norm(x_np[p.cluster] - p.cx)**q
            wts[p.group][p.cluster] += p.weight

        
        for j in range(ell):
            f[j] /= sum(wts[j])
            f[j] -= x[-1]

        # computing gradients w.r.t. the centers
        Df = matrix(0.0, (ell,k*d+1))

        for p in data:
            S_p = np.linalg.norm(x_np[p.cluster] - p.cx)**2
            if S_p != 0:
                Df[int(p.group),p.cluster*d:(p.cluster+1)*d] += p.weight*q*(S_p**(q/2-1))*(x_np[p.cluster] - p.cx)
        
        for j in range(ell):
            Df[j,:] /= sum(wts[j])

        Df[:,-1] = -1.0 # gradient w.r.t. the variable t.

        if z is None: return f, Df

        # H = z_0*Dsr_0 + z_1*Dsr_1 + ... + z_{ell-1}*Dsr_{ell-1}
        
        H_groups = [matrix(0.0, (k*d+1, k*d+1)) for i in range(ell)]
        for p in data:
            S_p = np.linalg.norm(x_np[p.cluster] - p.cx)**2
            if S_p!=0:
                H_groups[p.group][p.cluster*d:(p.cluster+1)*d,p.cluster*d:(p.cluster+1)*d] += q*(p.weight/sum(wts[p.group]))*(S_p**(q/2-1))*np.eye(d)
            if q != 2 and S_p!=0:
                H_groups[p.group][p.cluster*d:(p.cluster+1)*d,p.cluster*d:(p.cluster+1)*d] += q*(q-2)*(p.weight/sum(wts[p.group]))*(S_p**(q/2-2))*np.matmul(np.transpose(np.asarray([x_np[p.cluster] - p.cx])),np.asarray([x_np[p.cluster] - p.cx]))

        H = matrix(0.0, (k*d+1, k*d+1))
        for j in range(ell):
            H_groups[j][-1,-1] = 0.0
            H += z[j]*H_groups[j]
        return f, Df, H

    
    # solve and return c_1, ..., c_k, t
    solvers.options['show_progress'] = False
    # solvers.options['maxiters'] = 7
    sol = solvers.cpl(c, F)
    centers = np.array(sol['x'][:-1])
    val = sol['x'][-1]
    centers = centers.reshape((k,d))
    return  centers, val


def kzclustering_means(data, k, d, ell, q, centers):

    # ~~~ parameters ~~~
    # k:    the number of clusters; python float
    # d:    dimension; python float
    # ell:  number of groups; python int
    # data: List of Point Objects
    # Let P_{ji} be points of group j in cluster i

    # ~~~ convex program ~~~
    #     minimize    (0.c_1 + ... + 0.c_k) + t
    #     subject to  (\sum_{i \in [k]} \sum_{x \in P_{ji}} wt(x)*\norm{x - c_i}^2)/(\sum_{x \in P_{j}} wt(x)) <= t, forall j \in [\ell].
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
    wts = np.zeros((ell,k)) # since we take the whole data, this gives number of points.
    means = np.zeros((ell, k, d))
    alphas = np.zeros((ell,k))
    alpha_group = np.zeros(ell)
    group_costs = np.zeros(ell)
    flag = np.zeros((ell,k))
    for p in data:
        means[p.group][p.cluster] += p.cx
        wts[p.group][p.cluster] += 1
    for j in range(ell):
        for i in range(k):
            if wts[j][i] == 0:
                flag[j][i] = 1
            else:
                means[j][i] /= wts[j][i]
        
    for p in data:
        alphas[p.group][p.cluster] += np.linalg.norm(p.cx-means[p.group][p.cluster])**2
        
    for j in range(ell):
        if flag[j][i] == 0:
            alpha_group[j] = np.sum(alphas[j])/np.sum(wts[j])
        else:
            alpha_group[j] = 0

    for j in range(ell):
        if centers is not None:
            for i in range(k):
                if not flag[j][i]:
                    group_costs[j] += (np.linalg.norm(means[j][i]-centers[i].cx)**q)*wts[j][i]
        else:
            for i in range(k):
                if not flag[j][i]:
                    group_costs[j] += (np.linalg.norm(means[j][i])**q)*wts[j][i]
        group_costs[j] /= np.sum(wts[j])
        group_costs[j] += alpha_group[j]
    

    val = max(group_costs)
    def F(x=None, z=None):
        if x is None:  
            x0 = matrix(0.0, (k*d+1, 1))
            if centers is not None:
                for i in range(k):
                    x0[i*d:(i+1)*d] = centers[i].cx
            x0[-1] = val
            return ell, x0  # consider changing this x0
        # note that anything is in the domain of f.

        # converting matrix x of shape k*d+1 to numpy array x_np of shape (k,d) ignoring the last element of x
        x_np = np.array(x[:-1])
        x_np = x_np.reshape((k,d))
        # clustering cost minus t: (\sum_{i \in [k]} \sum_{p \in P_{ji}} wt(p)*\norm{p - c_i}^2)/(\sum_{p \in P_j} wt(p)) - t.
        # using p for points to avoid confusion with x, the variables.
        # here f is ell dimensional
        f = matrix(0.0, (ell, 1))
        for j in range(ell):
            for i in range(k):
                if not flag[j][i]:
                    f[j] += (np.linalg.norm(means[j][i] - x_np[i])**q)*wts[j][i] 
            f[j] /= np.sum(wts[j])
            f[j] += alpha_group[j]
            f[j] -= x[-1]
            
        # computing gradients w.r.t. the centers
        Df = matrix(0.0, (ell,k*d+1))

        for j in range(ell):
            for i in range(k):
                if not flag[j][i]:
                    Df[j,i*d:(i+1)*d] += 2*(wts[j][i]/np.sum(wts[j]))*(x_np[i]- means[j][i])
        
        
        Df[:,-1] = -1.0 # gradient w.r.t. the variable t.

        if z is None: return f, Df

        # H = z_0*Dsr_0 + z_1*Dsr_1 + ... + z_{ell-1}*Dsr_{ell-1}
        
        H_groups = [matrix(0.0, (k*d+1, k*d+1)) for i in range(ell)]
        for j in range(ell):
            for i in range(k):
                if not flag[j][i]:
                    H_groups[j][i*d:(i+1)*d,i*d:(i+1)*d] += 2*(wts[j][i]/sum(wts[j]))*np.eye(d)
            
        H = matrix(0.0, (k*d+1, k*d+1))
        for j in range(ell):
            H_groups[j][-1,-1] = 0.0
            H += z[j]*H_groups[j]
        return f, Df, H

    
    # solve and return c_1, ..., c_k, t
    solvers.options['show_progress'] = False
    sol = solvers.cpl(c, F)
    centers = np.array(sol['x'][:-1])
    val = sol['x'][-1]
    centers = centers.reshape((k,d))
    return  centers, val
    
# def kzclustering_SGD(data, k, d, ell, q):

#     # ~~~ parameters ~~~
#     # k:    the number of clusters; python float
#     # d:    dimension; python float
#     # ell:  number of groups; python int
#     # data: List of Point Objects
#     # Let P_{ji} be points of group j in cluster i
#     wts = [[0 for i in range(k)] for j in range(ell)]
#     for p in data:
#         wts[p.group][p.cluster] += p.weight
#     # print("in grad descent")
#     def func(x):
#         f = np.zeros(ell)
#         for p in data:
#             f[int(p.group)] += p.weight*np.linalg.norm(x[p.cluster] - p.cx)**q
            
#         Df = np.zeros((ell,k,d))

#         for p in data:
#             S_p = np.linalg.norm(x[p.cluster] - p.cx)**2
#             Df[int(p.group),p.cluster] += p.weight*q*(S_p**(q/2-1))*(x[p.cluster] - p.cx)
        
#         for j in range(ell):
#             f[j] /= sum(wts[j])
#             Df[j] /= sum(wts[j])
        
#         max_j = np.argmax(f)
#         return f[max_j], Df[max_j]
#         # f = [0.0]*ell
#         # # f = jnp.zeros(ell)
#         # for p in data:
#         #     f[int(p.group)] += p.weight*(jnp.linalg.norm(x[p.cluster] - p.cx))**2
#         # for j in range(ell):
#         #     f[j] = f[j]/sum(wts[j])
#         # return max(f)
    
#     centers = np.zeros((k,d))
#     learning_rate = 0.01
    
#     for i in range(1000):
#         val, grads = func(centers)
#         centers -= learning_rate * grads
        

#         if i % 10 == 0:
#             learning_rate -= 0.00005
#             # print(val, learning_rate)
        

#     return centers, val



# def kzclustering_SGD(data, k, d, ell, q):

#     # ~~~ parameters ~~~
#     # k:    the number of clusters; python float
#     # d:    dimension; python float
#     # ell:  number of groups; python int
#     # data: List of Point Objects
#     # Let P_{ji} be points of group j in cluster i
#     wts = [[0 for i in range(k)] for j in range(ell)]
#     for p in data:
#         wts[p.group][p.cluster] += p.weight
#     # print("in grad descent")
#     def func(x):
#         f = jnp.zeros(ell)
#         for p in data:
#             f = index_add(f, index[p.group], p.weight*(jnp.linalg.norm(x[p.cluster] - p.cx))**2)
#         for j in range(ell):
#             f = index_update(f, index[j], f[j]/sum(wts[j]))
#         return jnp.amax(f)
        
#         # f = [0.0]*ell
#         # # f = jnp.zeros(ell)
#         # for p in data:
#         #     f[int(p.group)] += p.weight*(jnp.linalg.norm(x[p.cluster] - p.cx))**2
#         # for j in range(ell):
#         #     f[j] = f[j]/sum(wts[j])
#         # return max(f)
    
#     centers = jnp.zeros((k,d))
#     learning_rate = 0.001
#     grad_f = jax.jit(grad(func))
#     print('before')
#     grad_f(centers)
#     print('after')
#     for i in range(100):
#         centers -= learning_rate * grad_f(centers)
#         learning_rate*0.5

#         if i % 10 == 0:
#             print(func(centers))
        

#     return centers, func(centers)

def linearprojclustering(data, k, J, d, ell, q):
    wts = [[0 for i in range(k)] for j in range(ell)]
    for p in data:
        wts[p.group][p.cluster] += p.weight

    X = []
    for i in range(k):
        X.append(cp.Variable((d,d), symmetric=True))
    t = cp.Variable()

    # The operator >> denotes matrix inequality.
    constraints = [X[i] >> 0 for i in range(k)]
    constraints +=  [np.eye(d) >> X[i] for i in range(k)]
    constraints += [cp.trace(X[i]) >= d-J for i in range(k)]
    
    obj = [0 for j in range(ell)]
    
    for p in data:
        cx = np.array([p.cx])
        obj[p.group] += p.weight*np.power( (cx@X[p.cluster])@np.transpose(cx), q*0.5) 
    
    for j in range(ell):
        obj[j] /= sum(wts[j])
                            
    constraints += [obj[j] <= t for j in range(ell)]
    

    # constraints += [np.sum( [wt_a/normalisationfactor*np.power( (np.transpose(a)@X[0])@a, p*0.5) for a in points[:2]] + [np.power( (np.transpose(a)@X[1])@a, p*0.5) for a in points[4:6]]) <= t]
    # constraints += [np.sum( [np.power( (np.transpose(a)@X[0])@a, p*0.5) for a in points[2:4]] + [np.power( (np.transpose(a)@X[1])@a, p*0.5) for a in points[6:8]]) <= t]

    prob = cp.Problem(cp.Minimize(t), constraints)
    prob.solve(solver='MOSEK')

    # Print result.
    # print("The optimal value is", prob.value)
    
    return [X[i].value for i in range(len(X))], prob.value

# def kzclustering(data, k, d, ell, q):
#     wts = [[0 for i in range(k)] for j in range(ell)]
#     for p in data:
#         wts[p.group][p.cluster] += p.weight

#     X = []
#     for i in range(k):
#         X.append(cp.Variable((1,d)))
#     t = cp.Variable()
#     obj = [0 for j in range(ell)]
    
#     for p in data:
#         cx = np.array([p.cx])
#         obj[int(p.group)] += p.weight*np.power( cp.atoms.norm(cx-X[int(p.cluster)]) ,q)    

    
#     for j in range(ell):
#         obj[j] /= sum(wts[j])
                            
#     constraints = [obj[j] <= t for j in range(ell)]
#     prob = cp.Problem(cp.Minimize(t), constraints)
#     prob.solve(verbose=True)
#     centers = [X[i].value[0] for i in range(len(X))]
#     cost = prob.value


#     return centers, cost
