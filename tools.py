#### 1. loading packages

import numpy as np
import pandas as pd
import os
import sys
import random
from sklearn import preprocessing
import warnings
if os.name == 'posix':
    from sklearn.exceptions import ConvergenceWarning
#from sklearn.exceptions import FutureWarning

# 1.2) need to change PATH here
if os.name == 'posix':
    project_path = os.getcwd()
else:
    project_path = 'C:/Users/Victor/Documents/programmes/Github/blemais'
    sys.path.append(project_path)
    sys.path.append(project_path+'/regressions')



#### 2. usedull functions


def loadData(filename):
    PDmaize = pd.read_table(os.path.join(project_path, "data", filename))
    ind2name = list(PDmaize)
    name2ind = {i:j for j,i in enumerate(ind2name)}
    return np.array(PDmaize), ind2name, name2ind
    
def addColumn(arr, ind2name, name2ind, name, column):
    column=np.transpose(np.array([column]))
    ind2name.append(name)
    name2ind[name] = len(ind2name)-1
    arr = np.concatenate((arr, column), axis=1)
    return arr, ind2name, name2ind


def addDE(maize, ind2name, name2ind, RUM=10, name=False):
    if type(name) != type(""):
        name = str(RUM)    
    maize, ind2name, name2ind = addColumn(maize, ind2name, name2ind,  "RU" + name + "_1", np.array([RUM for i in range(len(maize))]))
    
    for i in range(2,10):
        colRU = "RU" + name + "_" + str(i)
        colRU1 = "RU" + name + "_" + str(i - 1)
        colPR = "PR_" + str(i)
        colETP = "ETP_" + str(i)
        colETR = "ETR" + name + str(i)
        colDE = "DE" + name + "_" + str(i)
    
        DE = (maize[:,name2ind[colPR]] - maize[:,name2ind[colETP]]) < 0
    
        maize, ind2name, name2ind = addColumn(maize, ind2name, name2ind, colRU, np.minimum(maize[:,name2ind[colRU1]] + maize[:,name2ind[colPR]] - maize[:,name2ind[colETP]],RUM)*(1 - DE) + DE*np.maximum(maize[:,name2ind[colRU1]]*np.exp((maize[:,name2ind[colPR]] - maize[:,name2ind[colETP]])/RUM),0))
        maize, ind2name, name2ind = addColumn(maize, ind2name, name2ind, colETR, DE*(maize[:,name2ind[colRU1]] + maize[:,name2ind[colPR]] - maize[:,name2ind[colRU]]) + (1-DE)*(maize[:,name2ind[colETP]]))
        maize, ind2name, name2ind = addColumn(maize, ind2name, name2ind, colDE, maize[:,name2ind[colETP]] - maize[:,name2ind[colETR]])
        
    return maize, ind2name, name2ind


def addTm(maize, ind2name, name2ind):
    for i in range(1,10):
        maize, ind2name, name2ind = addColumn(maize, ind2name, name2ind, "Tm_" + str(i), (maize[:,name2ind["Tn_" + str(i)]] + maize[:,name2ind["Tx_" + str(i)]])/2 )        
    return maize, ind2name, name2ind

def addGDD(maize, ind2name, name2ind, baseGDD=5, GDDname=False):
    if type(GDDname) != type(""):
        GDDname = "GDD" + str(baseGDD)
    for i in range(1,10):
        maize, ind2name, name2ind = addColumn(maize, ind2name, name2ind, GDDname + "_" + str(i), np.maximum(maize[:,name2ind["Tx_" + str(i)]]-5,5))        
    maize, ind2name, name2ind = addColumn(maize, ind2name, name2ind, GDDname + "_49", maize[:,name2ind[GDDname + "_4"]]+maize[:,name2ind[GDDname + "_5"]]+maize[:,name2ind[GDDname + "_6"]]+maize[:,name2ind[GDDname + "_7"]]+maize[:,name2ind[GDDname + "_8"]]+maize[:,name2ind[GDDname + "_9"]])
    return maize, ind2name, name2ind

def addVarAn(maize, ind2name, name2ind):
    maize, ind2name, name2ind = addColumn(maize, ind2name, name2ind, "Tm_4_9", (maize[:,name2ind["Tm_4"]] + maize[:,name2ind["Tm_5"]] + maize[:,name2ind["Tm_6"]]+maize[:,name2ind["Tm_7"]] + maize[:,name2ind["Tm_8"]] + maize[:,name2ind["Tm_9"]])/6)
    maize, ind2name, name2ind = addColumn(maize, ind2name, name2ind, "PR_4_9", (maize[:,name2ind["PR_4"]] + maize[:,name2ind["PR_5"]] + maize[:,name2ind["PR_6"]]+maize[:,name2ind["PR_7"]] + maize[:,name2ind["PR_8"]] + maize[:,name2ind["PR_9"]]))
    maize, ind2name, name2ind = addColumn(maize, ind2name, name2ind, "RV_4_9", (maize[:,name2ind["Tm_4"]] + maize[:,name2ind["RV_5"]] + maize[:,name2ind["RV_6"]]+maize[:,name2ind["RV_7"]] + maize[:,name2ind["RV_8"]] + maize[:,name2ind["RV_9"]]))
    maize, ind2name, name2ind = addColumn(maize, ind2name, name2ind, "DE_4_9", (maize[:,name2ind["DE10_4"]] + maize[:,name2ind["DE10_5"]] + maize[:,name2ind["DE10_6"]]+maize[:,name2ind["DE10_7"]] + maize[:,name2ind["DE10_8"]] + maize[:,name2ind["DE10_9"]]))
    return maize, ind2name, name2ind



def delVar(x, xind2name, xname2ind, name):
    if isinstance(name,list):
        for n in name:
            x, xind2name, xname2ind = delVar(x, xind2name, xname2ind, n)
            # x = x[:,np.array(list(set(range(x.shape[1]))-set([xname2ind[n]])),dtype=np.int)]
            # xind2name.remove(n)
            # del xname2ind[n]
    else:
        x = x[:,np.array(list(set(range(x.shape[1]))-set([xname2ind[name]])),dtype=np.int)]
        xind2name.remove(name)
        xname2ind = {j:i for i,j in enumerate(xind2name)}
        # del xname2ind[name]
    return x, xind2name, xname2ind


def selVar(x, xind2name, xname2ind, name):
    x = x[:,np.array([xname2ind[n] for n in name],dtype=np.int)]
    xind2name = name
    xname2ind = {n : xname2ind[n] for n in name}
    return x, xind2name, xname2ind
    
    
def splitTestYear(x, y, year, nb_year=4, seed=0, n=0):
    random.seed(seed)
    sel_year=np.array(list(set(year)))
    random.shuffle(sel_year)
    ind_test = np.array(np.array(range(n*nb_year,n*nb_year+nb_year))%(len(sel_year)),dtype=np.int)
    year_test = sel_year[ind_test]
    year_train = np.array(list(set(sel_year) - set(year_test)),dtype=np.int)
    x_test = x[np.asarray([(i in year_test) for i in year]),:]
    y_test = y[np.asarray([(i in year_test) for i in year])]
    x_train = x[np.asarray([(i in year_train) for i in year]),:]
    y_train = y[np.asarray([(i in year_train) for i in year])]
    return x_train, x_test, y_train, y_test


def splitTestYear2(x, y, year, nb_year=4, seed=0, n=0):
    random.seed(seed)
    sel_year=np.array(list(set(year)))
    random.shuffle(sel_year)
    ind_test = np.array(np.array(range(n*nb_year,n*nb_year+nb_year))%(len(sel_year)),dtype=np.int)
    year_test = sel_year[ind_test]
    year_train = np.array(list(set(sel_year) - set(year_test)),dtype=np.int)
    x_test = x[np.asarray([(i in year_test) for i in year]),:]
    y_test = y[np.asarray([(i in year_test) for i in year])]
    x_train = x[np.asarray([(i in year_train) for i in year]),:]
    y_train = y[np.asarray([(i in year_train) for i in year])]
    return x_train, x_test, y_train, y_test, year_train, year_test



#def aggregateData(x, y, year, var):


class split_func_for_reg:
    def __init__(self, year):
        self.year = year
        # self.x = x

    def __call__(self, x, y, test_size=0.1, random_state=0):
        # assert x.tolist() == self.x.tolist()
        if isinstance(test_size, float):
            # x_train, x_test, y_train, y_test = splitTestYear(x, y, self.year, nb_year=int(test_size*len(set(self.year))), seed=random_state, n=0)
            # return splitTestYear(x, y, self.year, nb_year=int(test_size*len(set(self.year))), seed=random_state, n=0)
            return splitTestYear(x, y, self.year, nb_year=int(test_size*len(set(self.year))), seed=int(random_state*test_size), n=random_state)
        else:
            # x_train, x_test, y_train, y_test = splitTestYear(x, y, self.year, nb_year=test_size, seed=random_state, n=0)
            # return splitTestYear(x, y, self.year, nb_year=test_size, seed=random_state, n=0)
            return splitTestYear(x, y, self.year, nb_year=test_size, seed=int(random_state*test_size), n=random_state - int(random_state*test_size)*int(1/test_size))
        print(len(list(set(x_train[:,0]))))
        print(len(list(set(x_test[:,0]))))
        return x_train, x_test, y_train, y_test


class split_func_for_reg_2:
    def __init__(self, year, year_year):
        self.year = year
        self.year_year = year_year
        # self.x = x

    def __call__(self, x, y, x_year, y_year, test_size=0.1, random_state=0):
        # assert x.tolist() == self.x.tolist()
        if isinstance(test_size, float):
            # x_train, x_test, y_train, y_test = splitTestYear(x, y, self.year, nb_year=int(test_size*len(set(self.year))), seed=random_state, n=0)
            # return splitTestYear(x, y, self.year, nb_year=int(test_size*len(set(self.year))), seed=random_state, n=0)
            x_train_, x_test, y_train_, y_test, year_train, year_test = splitTestYear2(x, y, self.year, nb_year=int(test_size*len(set(self.year))), seed=int(random_state*test_size), n=random_state)
        else:
            # x_train, x_test, y_train, y_test = splitTestYear(x, y, self.year, nb_year=test_size, seed=random_state, n=0)
            # return splitTestYear(x, y, self.year, nb_year=test_size, seed=random_state, n=0)
            x_train_, x_test, y_train_, y_test, year_train, year_test = splitTestYear2(x, y, self.year, nb_year=test_size, seed=int(random_state*test_size), n=random_state - int(random_state*test_size)*int(1/test_size))
        
        x_test_ = x_year[np.asarray([(i in year_test) for i in self.year_year.astype(int)]),:]
        y_test_ = y_year[np.asarray([(i in year_test) for i in self.year_year.astype(int)])]
        x_train = x_year[np.asarray([(i in year_train) for i in self.year_year.astype(int)]),:]
        y_train = y_year[np.asarray([(i in year_train) for i in self.year_year.astype(int)])]
    
        print(len(list(set(x_train[:,0]))))
        print(len(list(set(x_test[:,0]))))
        return x_train, x_test, y_train, y_test

