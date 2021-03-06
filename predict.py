

#### 1. loading packages

# 1.1) import non homemade modules
#import numpy as np
#import pandas as pd
import os
import sys
import random
from sklearn import preprocessing
from copy import copy
#import warnings
#if os.name == 'posix':
#    from sklearn.exceptions import ConvergenceWarning
#from sklearn.exceptions import FutureWarning

# 1.2) need to change PATH here

project_path = 'C:/Users/Victor/Documents/programmes/Github/blemais'
sys.path.append(project_path)
sys.path.append(os.path.join(project_path, "regressions"))
project_path_regressions = os.path.join(project_path, "regressions")

# 1.3) import our homemade modules

from tools import *
#from regressions.regressions import *
#from multi_armed_bandit.multi_armed_bandit import *



#### 2. Other boring parameters for code execution

if os.name == 'posix':
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)



#### 3. Processing train data

maize_train, ind2name_train, name2ind_train = loadData("TrainingDataSet_Maize.txt")


# 3.1) creating variables in train dataset

maize_train, ind2name_train, name2ind_train = addDE(maize_train, ind2name_train, name2ind_train)
maize_train, ind2name_train, name2ind_train = addTm(maize_train, ind2name_train, name2ind_train)
maize_train, ind2name_train, name2ind_train = addGDD(maize_train, ind2name_train, name2ind_train)
maize_train, ind2name_train, name2ind_train = addVarAn(maize_train, ind2name_train, name2ind_train)


maize_train = np.concatenate([maize_train, maize_train*maize_train], axis=1)
ind2name_train = ind2name_train+[ n+"_sqrd" for n in ind2name_train]
name2ind_train = {j:i for i,j in enumerate(ind2name_train)}

maize_train, ind2name_train, name2ind_train = delVar(maize_train, ind2name_train, name2ind_train, "NUMD_sqrd")
maize_train, ind2name_train, name2ind_train = delVar(maize_train, ind2name_train, name2ind_train, "yield_anomaly_sqrd")
maize_train, ind2name_train, name2ind_train = delVar(maize_train, ind2name_train, name2ind_train, "year_harvest_sqrd")
maize_train, ind2name_train, name2ind_train = delVar(maize_train, ind2name_train, name2ind_train, "IRR_sqrd")


year_train = maize_train[:, name2ind_train["year_harvest"]]
dep_train = maize_train[:, name2ind_train["NUMD"]]

maize_train, ind2name_train, name2ind_train = delVar(maize_train, ind2name_train, name2ind_train, "NUMD")
maize_train, ind2name_train, name2ind_train = delVar(maize_train, ind2name_train, name2ind_train, "year_harvest")

y_train = maize_train[:, name2ind_train["yield_anomaly"]]

x_train = copy(maize_train)
xind2name_train = copy(ind2name_train)
xname2ind_train = copy(name2ind_train)

# 3.2) Selecting variable
sel = ['ETP_1','ETP_2','ETP_3','ETP_4','ETP_5','ETP_6','ETP_7','ETP_8','ETP_9',
'PR_1','PR_2','PR_3','PR_4','PR_5','PR_6','PR_7','PR_8','PR_9',
'RV_1','RV_2','RV_3','RV_4','RV_5','RV_6','RV_7','RV_8','RV_9',
'SeqPR_1','SeqPR_2','SeqPR_3','SeqPR_4','SeqPR_5','SeqPR_6','SeqPR_7','SeqPR_8','SeqPR_9',
'Tn_1','Tn_2','Tn_3','Tn_4','Tn_5','Tn_6','Tn_7','Tn_8','Tn_9',
'Tx_1','Tx_2','Tx_3','Tx_4','Tx_5','Tx_6','Tx_7','Tx_8','Tx_9',
'IRR']

x_train,xind2name_train,xname2ind_train = selVar(x_train, xind2name_train, xname2ind_train, sel)

# 3.3) Scaling train data

scaler_x = preprocessing.StandardScaler().fit(x_train)
scaler_y = preprocessing.StandardScaler().fit(y_train[:,np.newaxis])


x_train_s = copy(x_train)
x_train_s = scaler_x.transform(x_train_s)
y_train_s = copy(y_train)
y_train_s = scaler_y.transform(y_train_s[:,np.newaxis])[:,0]


#### 4. Fitting the model

reg = Regression_With_Custom_Kernel(KernelRidge(), Tanimoto())

reg.fit(x_train_s,y_train_s)


#### 5. Processing test data

maize_test, ind2name_test, name2ind_test = loadData("TestDataSet_Maize_blind.txt")


# 3.1) creating variables in train dataset


maize_test, ind2name_test, name2ind_test = addDE(maize_test, ind2name_test, name2ind_test)
maize_test, ind2name_test, name2ind_test = addTm(maize_test, ind2name_test, name2ind_test)
maize_test, ind2name_test, name2ind_test = addGDD(maize_test, ind2name_test, name2ind_test)
maize_test, ind2name_test, name2ind_test = addVarAn(maize_test, ind2name_test, name2ind_test)


maize_test = np.concatenate([maize_test, maize_test*maize_test], axis=1)
ind2name_test = ind2name_test+[ n+"_sqrd" for n in ind2name_test]
name2ind_test = {j:i for i,j in enumerate(ind2name_test)}

maize_test, ind2name_test, name2ind_test = delVar(maize_test, ind2name_test, name2ind_test, "NUMD_sqrd")
maize_test, ind2name_test, name2ind_test = delVar(maize_test, ind2name_test, name2ind_test, "year_harvest_sqrd")
maize_test, ind2name_test, name2ind_test = delVar(maize_test, ind2name_test, name2ind_test, "IRR_sqrd")


year_test = maize_test[:, name2ind_test["year_harvest"]]
dep_test = maize_test[:, name2ind_test["NUMD"]]

maize_test, ind2name_test, name2ind_test = delVar(maize_test, ind2name_test, name2ind_test, "NUMD")
maize_test, ind2name_test, name2ind_test = delVar(maize_test, ind2name_test, name2ind_test, "year_harvest")

x_test = copy(maize_test)
xind2name_test = copy(ind2name_test)
xname2ind_test = copy(name2ind_test)

# 3.2) Selecting variable

x_test,xind2name_test,xname2ind_test = selVar(x_test, xind2name_test, xname2ind_test, sel)

# 3.3) Scaling test data
x_test_s = copy(x_test)
x_test_s = scaler_x.transform(x_test_s)

#### 5. Predict model on test data

#check names
xind2name_test == xind2name_train

y_test_s = reg.predict(x_test_s)
y_test = copy(y_test_s)
y_test = scaler_y.inverse_transform(y_test[:,np.newaxis])[:,0]

#### 6. Export data

export1 = pd.DataFrame(y_test[np.newaxis,:])
export1.to_csv(os.path.join(project_path, "data", "mais_prediction_MoinardPierre1.txt"),sep='\t',index=False,header=False)

y_test[:,np.newaxis].shape
x_test.shape

maize_export = np.concatenate((year_test[:,np.newaxis],dep_test[:,np.newaxis],x_test,y_test[:,np.newaxis]),axis=1)
ind2name_export = ["year_harvest","NUMD"] + xind2name_test + ["predicted_yield_anomaly"]
name2ind_export = {j:i for i,j in enumerate(ind2name_export)}

export2 = pd.DataFrame(maize_export,columns = ind2name_export)
export2.to_csv(os.path.join(project_path, "data", "mais_prediction_MoinardPierre2.txt"),sep='\t',index=False,header=True)
