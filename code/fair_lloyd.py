import numpy as np

def line_search(deltaA, deltaB, alphaA, alphaB, l):
    gamma = 0.5
    gammal = 1
    gammah = 0
    T = 64
    for i in range(T):
        x = ((1 - gamma) * alphaB * l)/(gamma * alphaA + (1 - gamma) * alphaB)
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

def mw():
    return 0,[]

def solve_kmeans_clustering(data,svar,clustering,k,d,ell,z,method="line_search"):
    mus = np.zeros([ell,k,d])
    l = np.zeros(k)
    alphas = np.zeros([ell,k])
    deltas = np.zeros(ell)
    for i in range(k):
        datai = [data[np.logical_and(svar==j,clustering==i)] for j in range(ell)]
        for j in range(ell):
            alphas[j,i] = datai[j].shape[0]
        if method=="line_search":
            if alphas[0,i] + alphas[1,i] != 0:
                if alphas[0,i] == 0 or alphas[1,i] == 0:
                    mus[0,i,:] = np.mean(np.concat((datai[0],datai[1]),axis=0), axis=0)
                    mus[1,i,:] = mus[0,i,:]
                else:
                    mus[0,i,:] = np.mean(datai[0], axis=0)
                    mus[1,i,:] = np.mean(datai[1], axis=0)
                l[i] = np.linalg.norm(mus[0,i,:] - mus[1,i,:])
                deltas[0] += np.sum(np.power(np.linalg.norm(datai[0] - mus[0,i,:],axis=1),2))
                deltas[1] += np.sum(np.power(np.linalg.norm(datai[1] - mus[1,i,:],axis=1),2))
        
    deltas /= np.asarray([data[svar==j].shape[0] for j in range(ell)])
    if method=="line_search":
        cost, x = line_search(deltas[0], deltas[1], alphas[0], alphas[1], l)
        centers = np.zeros([k,d])
        for i in range(k):
            if l[i] == 0:
                centers[i, :] = mus[0,i,:]
            else:
                centers[i, :] = ((l[i]-x[i])*mus[0,i,:] + x[i]*mus[1,i,:])/l[i]
    else:
        cost,x = mw()
        centers = np.zeros([k,d])
    return centers, cost