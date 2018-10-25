### Alban et Totor

# exec(open('C:/Users/Victor/Documents/programmes/Github/blemais/main.py').read())


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


x = copy(maize_scaled)
xind2name = copy(ind2name_scaled)
xname2ind = copy(name2ind_scaled)


# mapping = {i[8] + i[7]:i[0] for i in x[:,:]}
# mapping_y = {i[8] + i[7]:i[0] for i in x[:,:]}
# yr = x[:,0]

#x,xind2name,xname2ind = delVar(x, xind2name, xname2ind, ["year_harvest", "yield_anomaly"])

x,xind2name,xname2ind = delVar(x, xind2name, xname2ind, "year_harvest")
x,xind2name,xname2ind = delVar(x, xind2name, xname2ind, "yield_anomaly")

#x,xind2name,xname2ind = delVar(x, xind2name, xname2ind, "IRR")

x_squared = copy(maize_squared)
x_squaredind2name = copy(maize_squaredind2name)
x_squaredname2ind = copy(maize_squaredname2ind)
x_squared,x_squaredind2name,x_squaredname2ind = delVar(x_squared, x_squaredind2name, x_squaredname2ind, "year_harvest")
x_squared,x_squaredind2name,x_squaredname2ind = delVar(x_squared, x_squaredind2name, x_squaredname2ind, "yield_anomaly")


# aa = [mapping[i[5]+i[6]] for i in x[:,:]]
# 1/0
x_reduced = copy(maize_scaled)
x_reducedind2name = copy(ind2name_scaled)
x_reducedname2ind = copy(name2ind_scaled)
x_reduced,x_reducedind2name,x_reducedname2ind = delVar(x_reduced, x_reducedind2name, x_reducedname2ind, ["year_harvest","yield_anomaly"])
sel1 = ['ETP_5','ETP_6','ETP_7','ETP_8','ETP_9','PR_4','PR_5','SeqPR_8','SeqPR_9','Tm_5','Tm_6','Tm_7','Tm_8','Tm_9']
x_reduced,x_reducedind2name,x_reducedname2ind = selVar(x_reduced, x_reducedind2name, x_reducedname2ind, sel1)

year = maize[:, name2ind["year_harvest"]]

#### 4) Runing regressions

#err = run_all_regressions(x, y, regs=0, verbose=True, show=False, x_test=0.1, final_verbose=range(5))

# err = run_all_regressions(x, y, regs="regressions/reg_lists/features.py", verbose=True, show=False, x_test=0.1, final_verbose=range(15))
# err = run_all_regressions(x, y, regs="regressions/reg_lists/five_best.py", verbose=True, show=False, x_test=0.1, final_verbose=range(15))

# sel = Uniform_MAB(1, 370)
# err = run_all_regressions(x, y, regs=0, verbose=True, show=False, x_test=0.1, final_verbose=range(15),selection_algo=sel)
# err = run_all_regressions(x, y, regs=0, verbose=True, show=False, x_test=0.1, final_verbose=range(5))
#err = run_all_regressions(x, y, regs="regressions/reg_lists/features.py", verbose=True, show=False, x_test=0.1, final_verbose=range(15))
sel = Uniform_MAB(1, 12*5)
#err = run_all_regressions(x, y, regs="regressions/reg_lists/five_best.py", verbose=True, show=False, x_test=0.1, final_verbose=range(15), selection_algo=sel, seed=3, split_func=split_func_for_reg(year))
# err = run_all_regressions(x, y, regs=[SVR()], verbose=True, show=False, x_test=0.1,selection_algo=sel)



# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# pca.fit(x)
# import copy
# a = copy.deepcopy(pca.components_)
# ia = np.argsort(a[1,:])
# [(round(a[1,i],3), ind2name[i]) for i in ia]
# plt.plot(a[:,ia].transpose())


#err = run_all_regressions(x, y, regs=0, verbose=True, show=False, x_test=0.1, final_verbose=range(15), selection_algo=sel, seed=5, split_func=split_func_for_reg(year))

#err = run_all_regressions(x, y, regs="regressions/reg_lists/five_best.py", verbose=True, show=False, x_test=0.1, final_verbose=range(15), selection_algo=sel, seed=5, split_func=split_func_for_reg(year))

#err = run_all_regressions(x, y, regs="regressions/reg_lists/five_best.py", verbose=True, show=False, x_test=0.1, final_verbose=False, selection_algo=sel, seed=0, split_func=split_func_for_reg(year))


err = run_all_regressions(x_squared, y, regs="regressions/reg_lists/five_best.py", verbose=True, show=False, x_test=0.1, final_verbose=True, selection_algo=sel, seed=0, split_func=split_func_for_reg(year))


y_pred = err[0]['reg'][1].predict(x)

export = np.column_stack((x,y,y_pred))
xind2name+["yield_anomaly_real","yield_anomaly_SVR"]
df = pd.DataFrame(export,columns = xind2name+["yield_anomaly_real","yield_anomaly_SVR"])
df.to_csv(project_path+"/data/predict.csv")


#err = run_all_regressions(x_squared, y, regs="regressions/reg_lists/five_best.py", verbose=True, show=False, x_test=0.1, final_verbose=False, selection_algo=sel, seed=5, split_func=split_func_for_reg(year))


#err = run_all_regressions(x, y, regs="C:/Users/Victor/Documents/programmes/Github/blemais/regressions/reg_lists/five_best.py", verbose=True, show=False, x_test=0.1, final_verbose=range(15))

# from sklearn.preprocessing import PolynomialFeatures
# poly = PolynomialFeatures(2)#, interaction_only=True)
# poly.fit(x)
# xx = poly.transform(x)

# xx = np.concatenate([x, x*x], axis=1)

#x=np.array([[0,1,2,3,4,5,6],[7,8,9,10,11,12,13]])

