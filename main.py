### Balbou et Victor

#V: Je fais un peu nawak, on reparlera organisation de code etc...

# exec(open('C:/Users/Victor/Documents/programmes/Github/blemais/main.py').read())


# chargement packages
import numpy as np
import pandas as pd
import os
import sys
import random

from regressions.regressions import *
from multi_armed_bandit.multi_armed_bandit import *
from sklearn import preprocessing
import warnings
#from sklearn.exceptions import FutureWarning
if os.name == 'posix':
    from sklearn.exceptions import ConvergenceWarning

if os.name == 'posix':
    project_path = os.getcwd()
else:
    project_path = 'C:/Users/Victor/Documents/programmes/Github/blemais'
    sys.path.append(project_path)
    sys.path.append(project_path+'/regressions')

    
# we load data
def loadData(filename):
    PDmaize = pd.read_table(os.path.join(project_path, "data", filename))
    ind2name = list(PDmaize)
    name2ind = {i:j for j,i in enumerate(ind2name)}
    return np.array(PDmaize), ind2name, name2ind

maize, ind2name, name2ind = loadData("TrainingDataSet_Maize.txt")


def addColumn(arr, ind2name, name2ind, name, column):
    column=np.transpose(np.array([column]))
    ind2name.append(name)
    name2ind[name] = len(ind2name)-1
    arr = np.concatenate((arr, column), axis=1)
    return arr, ind2name, name2ind




# maize[name2ind["Tx_1"]]

# creation de nouvelles variables

###  Deficit hydrique

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
    return maize, ind2name, name2ind
    

maize, ind2name, name2ind = addDE(maize, ind2name, name2ind)
maize, ind2name, name2ind = addTm(maize, ind2name, name2ind)
maize, ind2name, name2ind = addGDD(maize, ind2name, name2ind)
maize, ind2name, name2ind = addVarAn(maize, ind2name, name2ind)





if os.name == 'posix':
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)



np.mean(maize[:,1])

y = preprocessing.scale(maize[:, 1])



def delVar(x, xind2name, xname2ind, name):
    x = x[:,set(range(len(x)))-set(xname2ind[name])]
    xind2name = xind2name[set(range(len(x)))-set(xname2ind[name])]
    del xname2ind[name]
    return x, xind2name, xname2ind

x = preprocessing.scale(maize[:, 2:])
xind2name = ind2name[2:]
xname2ind = name2ind
for i in set(range(len(ind2name)))-set(range(2,len(ind2name))):
    del xname2ind[ind2name[i]]


year = maize[:, 0]

def splitTestYear (x, y, year, nb_year=4, seed=0, n=0):
    random.seed(seed)    
    sel_year=np.array(list(set(year)))
    random.shuffle(sel_year)
    ind_test =     np.array(np.array(range(n*nb_year,n*nb_year+nb_year))%(len(sel_year)),dtype=np.int)
    year_test = sel_year[ind_test]
    year_train = np.array(list(set(sel_year) - set(year_test)),dtype=np.int)
    x_test = x[np.asarray([(i in year_test) for i in year]),:]
    y_test = y[np.asarray([(i in year_test) for i in year])]
    x_train = x[np.asarray([(i in year_train) for i in year]),:]
    y_train = y[np.asarray([(i in year_train) for i in year])]
    return x_train, y_train, x_test, y_test



# err = run_all_regressions(x, y, regs=0, verbose=True, show=False, x_test=0.1, final_verbose=range(5))
#err = run_all_regressions(x, y, regs="regressions/reg_lists/features.py", verbose=True, show=False, x_test=0.1, final_verbose=range(15))
sel = Uniform_MAB(1, 37)
err = run_all_regressions(x, y, regs=0, verbose=True, show=False, x_test=0.1, final_verbose=range(15),selection_algo=sel, seed=3)
# err = run_all_regressions(x, y, regs=[SVR()], verbose=True, show=False, x_test=0.1,selection_algo=sel)


# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# pca.fit(x)
# import copy
# a = copy.deepcopy(pca.components_)
# ia = np.argsort(a[1,:])
# [(round(a[1,i],3), ind2name[i]) for i in ia]
# plt.plot(a[:,ia].transpose())

x,xind2name,xname2ind = delVar(x, xind2name, xname2ind, "NUMD")

x,xind2name,xname2ind = delVar(x, xind2name, xname2ind, "IRR")


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(2)#, interaction_only=True)
poly.fit(x)
xx = poly.transform(x)

xx = np.concatenate([x, x*x], axis=1)
