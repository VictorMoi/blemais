import sys
import os

#### 1. loading packages

# 1.1) import non homemade modules
#import numpy as np
#import pandas as pd

import random
from sklearn import preprocessing
from copy import copy

# 1.2) need to change PATH here

project_path = 'C:/Users/Victor/Documents/programmes/blemais/envoie'
sys.path.append(project_path)
sys.path.append(os.path.join(project_path, "kernels"))


# 1.3) import our homemade modules

from tools import *
from includes import *


#### 2. Other boring parameters for code execution

if os.name == 'posix':
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)



#### 3. Processing train data

maize_train, ind2name_train, name2ind_train = loadData("TrainingDataSet_Maize_VMAP.txt")


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
'Tm_1','Tm_2','Tm_3','Tm_4','Tm_5','Tm_6','Tm_7','Tm_8','Tm_9',
'ETR101','ETR102','ETR103','ETR104','ETR105','ETR106','ETR107','ETR108','ETR109',
'RU10_1','RU10_2','RU10_3','RU10_4','RU10_5','RU10_6','RU10_7','RU10_8','RU10_9',
'DE10_1','DE10_2','DE10_3','DE10_4','DE10_5','DE10_6','DE10_7','DE10_8','DE10_9',
'GDD5_1','GDD5_2','GDD5_3','GDD5_4','GDD5_5','GDD5_6','GDD5_7','GDD5_8','GDD5_9',
'GDD5_4_9','DE_4_9','Tm_4_9','PR_4_9','RV_4_9',
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

#reg1 = Regression_With_Custom_Kernel(KernelRidge(alpha=2.5), Tanimoto())
#reg1.fit(x_train_s,y_train_s)

#reg2 = Regression_With_Custom_Kernel(KernelRidge(alpha=0.25), Cauchy(sigma=30))
#reg2.fit(x_train_s,y_train_s)

#reg3 = Regression_With_Custom_Kernel(KernelRidge(alpha=0.15), Exponential(sigma=8.5))
#reg3.fit(x_train_s,y_train_s)

reg4 = Regression_With_Custom_Kernel(KernelRidge(alpha=0.7), RBF(gamma=0.0025))
reg4.fit(x_train_s,y_train_s)


#### 5. Processing test data

maize_test, ind2name_test, name2ind_test = loadData("TestDataSet_Maize_blind.txt")

# 5.1) creating variables in train dataset


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

# 5.2) Selecting variable

x_test,xind2name_test,xname2ind_test = selVar(x_test, xind2name_test, xname2ind_test, sel)

# 5.3) Scaling test data
x_test_s = copy(x_test)
x_test_s = scaler_x.transform(x_test_s)

#### 6. Predict model on test data

#check names
xind2name_test == xind2name_train

#y_test_s1 = reg1.predict(x_test_s)
#y_test_s2 = reg2.predict(x_test_s)
#y_test_s3 = reg3.predict(x_test_s)
y_test_s = reg4.predict(x_test_s)

#y_test_s = (y_test_s1 + y_test_s2 + y_test_s3 + y_test_s4)/4

y_test = copy(y_test_s)
y_test = scaler_y.inverse_transform(y_test[:,np.newaxis])[:,0]

#### 7. Export data

export1 = pd.DataFrame(y_test[np.newaxis,:])
export1.to_csv(os.path.join(project_path, "data", "mais_prediction_MoinardPierre1.txt"),sep='\t',index=False,header=False)

y_test[:,np.newaxis].shape
x_test.shape

maize_export = np.concatenate((year_test[:,np.newaxis],dep_test[:,np.newaxis],x_test,y_test[:,np.newaxis]),axis=1)
ind2name_export = ["year_harvest","NUMD"] + xind2name_test + ["predicted_yield_anomaly"]
name2ind_export = {j:i for i,j in enumerate(ind2name_export)}

export2 = pd.DataFrame(maize_export,columns = ind2name_export)
export2.to_csv(os.path.join(project_path, "data", "mais_prediction_MoinardPierre2.txt"),sep='\t',index=False,header=True)
