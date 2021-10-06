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

def normalize_and_crop_data(data):
    
    # columns are 'asian', 'black', 'white', 'sex', 'X0', ... , 'X1763'
    # cropping data 
    columns = list(data.columns.values)
    cropped_data = pd.DataFrame()
    darray = data.to_numpy()
    crop_length = 11
    sens = darray[:,:4]
    darray = darray[:,4:].reshape(darray.shape[0],42,42)
    darray = darray[:,crop_length:-crop_length, crop_length:-crop_length].reshape(darray.shape[0],-1)
    darray = np.concatenate((sens,darray),axis=1)
    data = pd.DataFrame(darray, columns = columns[:darray.shape[1]])
    flags = [False]*data.shape[1]

    for i in range(data.shape[1]):
        # columns centering
        data.iloc[:,i] = data.iloc[:,i] - data.iloc[:,i].mean()
        if data.iloc[:,i].std()!=0:
            data.iloc[:,i] = data.iloc[:,i]/data.iloc[:,i].std()
            flags[i] = True
    
    # group_centered = []
    # data_lowEd = data[,:]
    # lowEd_copy = data_lowEd

    # % date for high educated population
    # data_highEd = data(find(~normalized),:)
    # highEd_copy = data_highEd

    # mean_lowEd = mean(lowEd_copy,1)
    # mean_highEd = mean(highEd_copy, 1)

    # % centering data for high- and low-educated
    # for i=1:n
    # lowEd_copy(i,:) = lowEd_copy(i,:) - mean_lowEd
    # end

    # for i=1:n
    # highEd_copy(i,:) = highEd_copy(i,:) - mean_highEd
    # end


    return data.loc[:,flags]

def credit_preprocess(sens,attr):
    if attr == "EDUCATION":
        sens = np.asarray(sens.loc[:,attr])
        svar = np.asarray([0 if (sens[i]==1 or sens[i]==2) else 1 for i in range(len(sens))])
        groups = {0:"Lower Education",1:"Higher Education"}
    elif attr == "AGE": # Pending
        sens = np.asarray(sens.loc[:,attr])
        svar = []
        groups = {}
    elif attr == "GENDER":
        sens = np.asarray(sens.loc[:,"SEX"])
        svar = sens-1
        groups = {0:"Male",1:"Female"}
    elif attr == "MARRIAGE":
        sens = np.asarray(sens.loc[:,attr])
        svar = np.asarray([1 if sens[i]==2 else 0 for i in range(len(sens))])
        groups = {0:"Not Married",1:"Married"}
    return svar,groups

def adult_preprocess(sens,attr):
    sens = np.asarray(sens.loc[:,attr])
    if attr == "GENDER":
        svar = np.asarray([0 if sens[i].strip()=="Female" else 1 for i in range(len(sens))])
        groups = {0:"Female",1:"Male"}
    elif attr == "RACE":
        vals = list(set(list(sens)))
        svar = np.asarray([vals.index(sens[i]) for i in range(len(sens))])
        groups = {vals.index(val):val for val in vals}
    return svar,groups

def german_preprocess(sens,attr):
    sens = np.asarray(sens.loc[:,attr])
    svar = np.asarray([0 if sens[i]<25 else 1 for i in range(len(sens))])
    groups = {0:"age<25",1:"age>=25"}
    return svar,groups

def LFW_preprocess(sens,attr):
    if attr == "GENDER":
        sens = np.asarray(sens.loc[:,"sex"])
        svar = sens #np.asarray([0 if sens[i].strip()=="Female" else 1 for i in range(len(sens))])
        groups = {0:"Female",1:"Male"}
    elif attr == "RACE":
        vals = sens.columns[:-1]
        sens = np.asarray(sens.loc[:,vals])
        svar = np.asarray([0 if sens[i,0]==1 else (1 if sens[i,1]==1 else 2) for i in range(sens.shape[0])])
        groups = {i:vals[i] for i in range(len(vals))}
    return svar,groups

def german_preprocess(sens,attr):
    if attr == "AGE":
        sens = np.asarray(sens.loc[:,"AGE"])
        svar = np.asarray([0 if sens[i]<=25 else 1 for i in range(len(sens))])
        groups = {0:"> 25 yrs",1:"<= 25 yrs"}
    return svar,groups

def bank_preprocess(sens,attr):
    if attr == "AGE":
        sens = np.asarray(sens.loc[:,"age"])
        svar = np.asarray([0 if sens[i] > 40 else 1 for i in range(len(sens))])
        groups = {0:"> 40 yrs",1:"<= 40 yrs"}
    elif attr == "EDUCATION":
        sens = np.asarray(sens.loc[:,"education"])
        svar = np.asarray([1 if (sens[i][:4]=="prof" or sens[i][:4]=="univ") else 0 for i in range(len(sens))])
        groups = {0:"Lower Education",1:"Higher Education"}
    return svar,groups

def skillcraft_preprocess(sens,attr):
    if attr == "AGE":
        sens = np.asarray(sens.loc[:,"Age"])
        svar = np.asarray([0 if sens[i] >= 21 else 1 for i in range(len(sens))])
        groups = {0:">= 21 yrs",1:"< 21 yrs"}
    return svar,groups

def statlog_preprocess(sens,attr):
    if attr == "AGE":
        sens = np.asarray(sens.loc[:,"Age"])
        svar = np.asarray([0 if sens[i] > 50 else 1 for i in range(len(sens))])
        groups = {0:"> 50 yrs",1:"<= 50 yrs"}
    elif attr == "GENDER":
        sens = np.asarray(sens.loc[:,"Gender"])
        svar = sens
        groups = {0:"Female",1:"Male"}
    return svar,groups

def get_data(dataset, attr, flag):
    x = pd.read_csv("./data/" + dataset + "/" + dataset + flag + ".csv",index_col=0)
    sens = pd.read_csv("./data/" + dataset + "/" + dataset + "_sensattr.csv",index_col=0)
    x = np.asarray(x)
    if dataset=="credit":
        svar, groups = credit_preprocess(sens,attr)
    elif dataset=="adult":
        svar, groups = adult_preprocess(sens,attr)
    elif dataset=="LFW":
        svar, groups = LFW_preprocess(sens,attr)
    elif dataset=="german":
        svar, groups = german_preprocess(sens,attr)
    elif dataset=="bank":
        svar, groups = bank_preprocess(sens,attr)
    elif dataset=="skillcraft":
        svar, groups = skillcraft_preprocess(sens,attr)
    elif dataset=="statlog":
        svar, groups = statlog_preprocess(sens,attr)
    y = [[] for i in groups]
    data = []
    for i in range(x.shape[0]):
        data.append(Point(x[i],int(svar[i])))
        y[int(svar[i])].append(x[i])
    
    dataGC = [] 
    # print(dataGC[0][:,1].mean())
    for j in range(len(y)):
        y[j] = np.asarray(y[j])
        y[j] = y[j] - np.mean(y[j],axis=0)
        dataGC.append([Point(y[j][i],j) for i in range(y[j].shape[0])])
    return data,dataGC,groups

def dataNgen(dataset):
    data = pd.read_csv("./data/" + dataset + "/" + dataset + ".csv",index_col = 0)
    dataN = normalize_data(data)
    dataN.to_csv("./data/" + dataset + "/" + dataset + "N.csv")

def dataNCgen(dataset):
    data = pd.read_csv("./data/" + dataset + "/" + dataset + ".csv",index_col = 0)
    dataNC = normalize_and_crop_data(data)
    dataNC.to_csv("./data/" + dataset + "/" + dataset + "NC.csv")


def dataPgen(dataset,k):
    dataN = pd.read_csv("./data/" + dataset + "/" + dataset + "N.csv",index_col = 0)
    pca = PCA(n_components=k)
    dataP = pd.DataFrame(pca.fit_transform(dataN))
    dataP.to_csv("./data/" + dataset + "/" + dataset + "P_k=" + str(k) + ".csv")