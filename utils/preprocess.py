import pandas as pd
from utils.utilities import normalize_data

def credit_preprocess():
    data = pd.read_csv("./data/credit.csv",index_col = 0)
    sensitive = data.iloc[:,0]
    sensitiveN = pd.DataFrame([0 if (sensitive.iloc[i]==1 or sensitive.iloc[i]==2) else 1 for i in range(len(sensitive))],columns=[data.columns[0]])
    data = data.iloc[:,1:]
    dataN = normalize_data(data)
    return dataN,sensitiveN,{0:"Lower Education",1:"Higher Education"}

def LFW_preprocess():
    dataAll = pd.read_csv("./data/LFW.csv",index_col = 0)
    sensitive = dataAll["sex"]
    data = dataAll.iloc[:,4:]
    dataN = normalize_data(data)
    return dataN, sensitive, {0:"Female",1:"Male"}