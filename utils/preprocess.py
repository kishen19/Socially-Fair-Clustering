import pandas as pd
from utils.utilities import normalize_data
from sklearn.decomposition import PCA

def credit_preprocess(sens,attr):
    if attr == "EDUCATION":
        svar = pd.DataFrame([0 if (sens.iloc[i]==1 or sens.iloc[i]==2) else 1 for i in range(len(sens))])
        groups = {0:"Lower Education",1:"Higher Education"}
    elif attr == "AGE": # Pending
        svar = []
        groups = {}
    elif attr == "SEX":
        svar = sens-1
        groups = {0:"Male",1:"Female"}
    elif attr == "MARRIAGE":
        svar = pd.DataFrame([1 if sens.iloc[i]==2 else 0 for i in range(len(sens))])
        groups = {0:"Not Married",1:"Married"}
    return svar,groups

def adult_preprocess(sens,attr):
    if attr == "SEX":
        svar = pd.DataFrame([0 if sens.iloc[i]=="Female" else 1 for i in range(len(sens))])
        groups = {0:"Female",1:"Male"}
    elif attr == "RACE":
        vals = list(set(list(sens)))
        svar = pd.DataFrame([vals.index(sens.iloc[i]) for i in range(len(sens))])
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
    sens = sens.loc[:,attr]
    if dataset=="credit":
        svar, groups = credit_preprocess(sens,attr)
    elif dataset=="adult":
        svar, groups = adult_preprocess(sens,attr)
    elif dataset=="LFW":
        svar, groups = LFW_preprocess(sens,attr)
    return data,svar,groups

def dataNgen(dataset):
    data = pd.read_csv("./data/" + dataset + "/" + dataset + ".csv",index_col = 0)
    dataN = normalize_data(data)
    dataN.to_csv("./data/" + dataset + "/" + dataset + "N.csv")

def dataPgen(dataset,k):
    dataN = pd.read_csv("./data/" + dataset + "/" + dataset + "N.csv",index_col = 0)
    pca = PCA(n_components=k)
    dataP = pd.DataFrame(pca.fit_transform(dataN))
    dataP.to_csv("./data/" + dataset + "/" + dataset + "P_k=" + str(k) + ".csv")