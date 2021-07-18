import pandas as pd
from sklearn.decomposition import PCA
from utils.utilities import normalize_data

def dataNgen(dataset):
    data = pd.read_csv("./data/" + dataset + "/" + dataset + ".csv",index_col = 0)
    sensitive = data.iloc[:,0]
    if dataset=="credit":
        sensitiveN = pd.DataFrame([0 if (sensitive.iloc[i]==1 or sensitive.iloc[i]==2) else 1 for i in range(len(sensitive))],columns=[data.columns[0]])
    elif dataset=="adult":
        sensitiveN = sensitive-1
        data = data.iloc[:,1:]
    dataN = normalize_data(data)
    DataN = pd.concat([sensitiveN,dataN],axis=1)
    DataN.to_csv("./data/" + dataset + "/" + dataset + "N.csv")

def dataPgen(dataset,k):
    dataN = pd.read_csv("./data/" + dataset + "/" + dataset + "N.csv",index_col = 0)
    svar = dataN.iloc[:,0]
    dataN = dataN.iloc[:,1:]
    pca = PCA(n_components=k)
    dataP = pd.DataFrame(pca.fit_transform(dataN))
    DataP = pd.concat([svar,dataP],axis=1)
    DataP.to_csv("./data/" + dataset + "/" + dataset + "P_k=" + str(k) + ".csv")