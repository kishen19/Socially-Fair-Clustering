import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from utils.classes import Point

def normalize_data(data):
    flags = [False]*data.shape[1]
    for i in range(data.shape[1]):
        data.iloc[:,i] = data.iloc[:,i] - data.iloc[:,i].mean()
        if data.iloc[:,i].std()!=0:
            data.iloc[:,i] = data.iloc[:,i]/data.iloc[:,i].std()
            flags[i] = True
    return data.loc[:,flags]

def credit_preprocess(sens,attr):
    if attr == "EDUCATION":
        svar = np.asarray([0 if (sens[i]==1 or sens[i]==2) else 1 for i in range(len(sens))])
        groups = {0:"Lower Education",1:"Higher Education"}
    elif attr == "AGE": # Pending
        svar = []
        groups = {}
    elif attr == "SEX":
        svar = sens-1
        groups = {0:"Male",1:"Female"}
    elif attr == "MARRIAGE":
        svar = np.asarray([1 if sens[i]==2 else 0 for i in range(len(sens))])
        groups = {0:"Not Married",1:"Married"}
    return svar,groups

def adult_preprocess(sens,attr):
    if attr == "SEX":
        svar = np.asarray([0 if sens[i]=="Female" else 1 for i in range(len(sens))])
        groups = {0:"Female",1:"Male"}
    elif attr == "RACE":
        vals = list(set(list(sens)))
        svar = np.asarray([vals.index(sens[i]) for i in range(len(sens))])
        groups = {vals.index(val):val for val in vals}
    return svar,groups

def LFW_preprocess(sens,attr): # Pending
    return [],{}
    dataAll = pd.read_csv("./data/LFW.csv",index_col = 0)
    sensitive = dataAll["sex"]
    data = dataAll.iloc[:,4:]
    dataN = normalize_data(data)
    return dataN, sensitive, {0:"Female",1:"Male"}

def get_data(dataset, attr, flag):
    data = pd.read_csv("./data/" + dataset + "/" + dataset + flag + ".csv",index_col=0)
    sens = pd.read_csv("./data/" + dataset + "/" + dataset + "_sensattr.csv",index_col=0)
    data = np.asarray(data)
    sens = np.asarray(sens.loc[:,attr])
    if dataset=="credit":
        svar, groups = credit_preprocess(sens,attr)
    elif dataset=="adult":
        svar, groups = adult_preprocess(sens,attr)
    elif dataset=="LFW":
        svar, groups = LFW_preprocess(sens,attr)
    data = [Point(data[i],int(svar[i])) for i in range(data.shape[0])]
    return data,groups

def dataNgen(dataset):
    data = pd.read_csv("./data/" + dataset + "/" + dataset + ".csv",index_col = 0)
    dataN = normalize_data(data)
    dataN.to_csv("./data/" + dataset + "/" + dataset + "N.csv")

def dataPgen(dataset,k):
    dataN = pd.read_csv("./data/" + dataset + "/" + dataset + "N.csv",index_col = 0)
    pca = PCA(n_components=k)
    dataP = pd.DataFrame(pca.fit_transform(dataN))
    dataP.to_csv("./data/" + dataset + "/" + dataset + "P_k=" + str(k) + ".csv")