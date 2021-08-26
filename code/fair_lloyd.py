import numpy as np

def line_search(deltaA, deltaB, alphaA, alphaB, l, num_iters):
    gamma = 0.5
    gammal = 1
    gammah = 0
    T = num_iters
    for i in range(T):
        x = ((1 - gamma) * alphaB * l)/(gamma * alphaA + (1 - gamma) * alphaB)
        x = np.nan_to_num(x)
        f = deltaA + np.sum(alphaA * np.power(x,2))
        g = deltaB + np.sum(alphaB * np.power(l-x,2))
        
        cost = max(f, g)
        
        if abs(f - g) < 1e-10:
            break
        
        if f > g:
            gammah = gamma
            gamma = (gamma + gammal)/2
        elif f < g:
            gammal = gamma
            gamma = (gamma + gammah)/2
    
    return cost, x

def mw(deltas,alphas,mus,num_iters):
    ell,k = alphas.shape
    gammas = np.asarray([1/ell for i in range(ell)])
    T = num_iters
    for t in range(T):
        gammasxalphas = (gammas*alphas.T).T
        coeff = gammasxalphas/np.sum(gammasxalphas,axis=0)
        C = np.asarray([np.sum([coeff[j,i]*mus[j,i] for j in range(ell)],axis=0) for i in range(k)])
        fs = deltas +  np.asarray([np.sum(np.power(np.linalg.norm(C-mus[j],axis=1),2)) for j in range(ell)])
        F = max(fs)
        ds = F-fs
        gammas = gammas*(1-ds/(max(ds)*np.sqrt(t+1+1)))
        gammas/=np.sum(gammas)
    return F,C

def solve_kmeans_clustering(data,k,d,ell,z,method="line_search",num_iters=64):
    mus = np.zeros([ell,k,d])
    l = np.zeros(k)
    alphas = np.zeros([ell,k])
    group_size = np.zeros(ell)
    deltas = np.zeros(ell)
    dataC = [[[] for j in range(ell)] for i in range(k)]
    for x in data:
        dataC[x.cluster][x.group].append(x.cx)
        group_size[x.group] += 1
    for i in range(k):
        for j in range(ell):
            dataC[i][j] = np.asarray(dataC[i][j])
    for i in range(k):
        for j in range(ell):
            alphas[j,i] = dataC[i][j].shape[0]/group_size[j]
        if np.sum(alphas[:,i])!=0:
            w = np.nonzero(alphas[:,i])[0][0]
            mubase = np.mean(dataC[i][int(w)],axis=0)
            for j in range(ell):
                if alphas[j,i]==0:
                    mus[j,i,:] = mubase
                else:
                    mus[j,i,:] = np.mean(dataC[i][j],axis=0)
            if method=="line_search":
                l[i] = np.linalg.norm(mus[0,i,:] - mus[1,i,:])

        for j in range(ell):
            if dataC[i][j].shape[0] > 0:
                deltas[j] += np.sum(np.power(np.linalg.norm(dataC[i][j] - mus[j,i,:],axis=1),2))

    deltas /= np.asarray([sum([dataC[i][j].shape[0] for i in range(k)]) for j in range(ell)])
    if method=="line_search":
        cost, x = line_search(deltas[0], deltas[1], alphas[0], alphas[1], l, num_iters)
        centers = np.zeros([k,d])
        for i in range(k):
            if l[i] == 0:
                centers[i, :] = mus[0,i,:]
            else:
                centers[i, :] = ((l[i]-x[i])*mus[0,i,:] + x[i]*mus[1,i,:])/l[i]
    elif method=="mw":
        cost,centers = mw(deltas,alphas,mus,num_iters)
    return centers, cost