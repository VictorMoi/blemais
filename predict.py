

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
if os.name == 'posix':
    project_path = os.getcwd()
    # project_path_regressions = os.path.join(project_path, "regressions")
else:
    project_path = 'C:/Users/Victor/Documents/programmes/Github/blemais'
    sys.path.append(project_path)
    sys.path.append(os.path.join(project_path, "regressions"))
    project_path_regressions = os.path.join(project_path, "regressions")

# 1.3) import our homemade modules
from tools import *
from regressions.regressions import *
from multi_armed_bandit.multi_armed_bandit import *



#### 2. Other boring parameters for code execution

if os.name == 'posix':
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    
#### 3. Processing data

# 3.1) loading data

maize, ind2name, name2ind = loadData("TrainingDataSet_Maize.txt")


maize_test, ind2name_test, name2ind_test = loadData("TestDataSet_Maize_blind.txt")


# 3.2) creating variables

maize, ind2name, name2ind = addDE(maize, ind2name, name2ind)
maize, ind2name, name2ind = addTm(maize, ind2name, name2ind)
maize, ind2name, name2ind = addGDD(maize, ind2name, name2ind)
maize, ind2name, name2ind = addVarAn(maize, ind2name, name2ind)


maize_squared = copy(maize)
ind2name_squared = copy(ind2name)
name2ind_squared = copy(name2ind)

maize_squared = np.concatenate([maize_squared, maize_squared*maize_squared], axis=1)
maize_squaredind2name = ind2name+[ n+"_sqrd" for n in ind2name]
maize_squaredname2ind = {j:i for i,j in enumerate(maize_squaredind2name)}

maize_squared, maize_squaredind2name, maize_squaredname2ind = delVar(maize_squared, maize_squaredind2name, maize_squaredname2ind, "NUMD_sqrd")
maize_squared, maize_squaredind2name, maize_squaredname2ind = delVar(maize_squared, maize_squaredind2name, maize_squaredname2ind, "yield_anomaly_sqrd")
maize_squared, maize_squaredind2name, maize_squaredname2ind = delVar(maize_squared, maize_squaredind2name, maize_squaredname2ind, "year_harvest_sqrd")
maize_squared, maize_squaredind2name, maize_squaredname2ind = delVar(maize_squared, maize_squaredind2name, maize_squaredname2ind, "IRR_sqrd")

# 3.3) creating other datasets from mai one (maize)

maize_scaled = preprocessing.scale(maize)
maize_squared = preprocessing.scale(maize_squared)
ind2name_scaled = copy(ind2name)
name2ind_scaled = copy(name2ind)

y = maize_scaled[:, name2ind_scaled["yield_anomaly"]]
y_mean = mean(maize_scaled[:, name2ind_scaled["yield_anomaly"]])
y_sd = np.var()


year = maize[:, name2ind["year_harvest"]]
dep = maize[:, name2ind["NUMD"]]

x = copy(maize_scaled)
xind2name = copy(ind2name_scaled)
xname2ind = copy(name2ind_scaled)


# 3.4) Selecting variable


#### 4. Fitting the model

reg = Regression_With_Custom_Kernel(KernelRidge(), Tanimoto()))

reg.fit()