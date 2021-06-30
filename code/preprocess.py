import pandas as pd

def credit_preprocess():
    data = pd.read_csv("./data/credit.csv",index_col = 0)
    sensitive = data.iloc[:,0]
    sensitiveN = pd.DataFrame([1 if (sensitive.iloc[i]==1 or sensitive.iloc[i]==2) else 2 for i in range(len(sensitive))],columns=[data.columns[0]])
    data = data.iloc[:,1:]
    flags = [False]*data.shape[1]
    for i in range(data.shape[1]):
        data.iloc[:,i] = data.iloc[:,i] - data.iloc[:,i].mean()
        if data.iloc[:,i].std()!=0:
            data.iloc[:,i] = data.iloc[:,i]/data.iloc[:,i].std()
            flags[i] = True
    data = data.loc[:,flags]
    return data,sensitiveN,{1:"Lower Education",2:"Higher Education"}