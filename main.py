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


year = maize[:, name2ind["year_harvest"]]
dep = maize[:, name2ind["NUMD"]]

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


x_lobell = copy(x_squared)
x_lobellind2name = copy(x_squaredind2name)
x_lobellname2ind = copy(x_squaredname2ind)
sel_lobell = ['Tm_4_9','PR_4_9','Tm_4_9_sqrd','PR_4_9_sqrd']
x_lobell,x_lobellind2name,x_lobellname2ind = selVar(x_lobell2, x_lobellind2name, x_lobell2name2ind, sel_lobell)

x_lobell2 = copy(x_squared)
x_lobell2ind2name = copy(x_squaredind2name)
x_lobell2name2ind = copy(x_squaredname2ind)
sel_lobell2 = ['Tm_4_9','PR_4_9','Tm_4_9_sqrd','PR_4_9_sqrd','DE_4_9','DE_4_9_sqrd']
x_lobell2,x_lobell2ind2name,x_lobell2name2ind = selVar(x_lobell2, x_lobellind2name, x_lobell2name2ind, sel_lobell2)

x_an = copy(x_squared)
x_anind2name = copy(x_squaredind2name)
x_anname2ind = copy(x_squaredname2ind)
sel_an = ['Tm_4_9','PR_4_9','Tm_4_9_sqrd','PR_4_9_sqrd','DE_4_9','DE_4_9_sqrd','RV_4_9','RV_4_9_sqrd']
x_an,x_anind2name,x_anname2ind = selVar(x_an, x_anind2name, x_anname2ind, sel_an)



x_year = copy(x)
xind2name_year = copy(xind2name)
xname2ind_year = copy(xname2ind)
year[:,np.newaxis].shape
mat_year = np.repeat(year[:,np.newaxis],np.unique(year).shape[0],axis=1)
mat_year.shape
mat_year = mat_year == np.unique(year[:,np.newaxis])
mat_year = mat_year.T
#mat_year.shape
#np.sum(mat_year,axis = 1)[:,np.newaxis].shape
#np.sum(mat_year,axis = 1)
mat_year = mat_year.astype(float)/(np.sum(mat_year,axis = 1)[:,np.newaxis])
x_year = mat_year.dot(x_year)

#x_dep.shape
#x.shape
#x_dep

y_year = copy(y)
y_year = mat_year.dot(y_year)

year_year = copy(year)
year_year = mat_year.dot(year_year)
year_year = np.asarray([int(round(i)) for i in year_year])


#### 4) Runing regressions

#err = run_all_regressions(x, y, regs=0, verbose=True, show=False, x_test=0.1, final_verbose=range(5))

# err = run_all_regressions(x, y, regs="regressions/reg_lists/features.py", verbose=True, show=False, x_test=0.1, final_verbose=range(15))
# err = run_all_regressions(x, y, regs="regressions/reg_lists/five_best.py", verbose=True, show=False, x_test=0.1, final_verbose=range(15))

# sel = Uniform_MAB(1, 370)
# err = run_all_regressions(x, y, regs=0, verbose=True, show=False, x_test=0.1, final_verbose=range(15),selection_algo=sel)
# err = run_all_regressions(x, y, regs=0, verbose=True, show=False, x_test=0.1, final_verbose=range(5))
#err = run_all_regressions(x, y, regs="regressions/reg_lists/features.py", verbose=True, show=False, x_test=0.1, final_verbose=range(15))
sel = Uniform_MAB(1, 1)#12*5)
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


# s = split_func_for_reg(year)
# x_train, x_test, y_train, y_test = s(x, y)

# x = x_train
# y = y_train

import sklearn
# d = sklearn.metrics.pairwise.euclidean_distances(x_train, x_test)

# means = []
# for i,v in enumerate(x_test):
#     ind = np.argsort(d[:,i])
#     means += [  np.mean(y_train[ind[1:4]])  ]

# np.mean((y_test - means)**2)


# a = sorted(range(d.shape[0]), key=lambda x:np.sum(d[x,:]))
# x1 = x[a,:]
# y1 = y[a]
# d1 = d[a,:][:,a]
# year1 = year[a]


# err = run_all_regressions(x, y, regs="regressions/reg_lists/five_best.py", verbose=True, show=False, xx_test=0.1, final_verbose=False, selection_algo=sel, seed=0, save_all_fit_regs=True, split_func=split_func_for_reg(year))

# reg = err[0]['reg'][1]

# c = 0+reg.coef_
# c = c/np.linalg.norm(c)
# xx = np.dot(x, np.diag(c))

# x = xx


#x = preprocessing.scale(np.concatenate([x, x*x], axis=1))

s = split_func_for_reg_2(year)


sel = Uniform_MAB(1, 3)
#err = run_all_regressions(x, y, regs="regressions/reg_lists/one_of_each.py", verbose=True, show=False, x_test=0.1, final_verbose=True, selection_algo=sel, seed=0, save_all_fit_regs=True, split_func=split_func_for_reg(year))
err = run_all_regressions(x, y, regs="regressions/reg_lists/one_of_each.py", verbose=True, show=False, x_test=0.1, final_verbose=True, selection_algo=sel, seed=0, save_all_fit_regs=True, split_func=split_func_for_reg(year))
1/0


d = sklearn.metrics.pairwise.euclidean_distances(x, x)
a = sorted(range(d.shape[0]), key=lambda x:np.sum(d[x,:]))
x = x[a,:]
y = y[a]
year = year[a]


a = np.argsort(year)
x = x[a,:]
y = y[a]

d = sklearn.metrics.pairwise.euclidean_distances(x, x)
plt.imshow(d)
plt.show()
1/0
colors = 'bkrgmc'

for i,c in enumerate(colors):
    years = list(set(year))
    indexes = (year == years[i])
    iindexes = (year != years[i])

    xx = x[indexes]
    xxi = x[iindexes]    
    yy = y[indexes]
    yyi = y[iindexes]
    
    d = sklearn.metrics.pairwise.euclidean_distances(xx, xxi)
    
    y1 = np.repeat(yy[:,np.newaxis], yy.shape[0], axis=1)
    y2 = np.repeat(yy[np.newaxis,:], yy.shape[0], axis=0)
    yy = y1 - y2
    
    indexes = np.concatenate([np.ones(20000), np.zeros(3394**2)])[:yy.shape[0]**2]
    np.random.shuffle(indexes)
    indexes = indexes == 1
    
    p_x = yy.flatten()[indexes]
    p_y = d.flatten()[indexes]
    
    plt.scatter(p_x, p_y, s=1, c=c)
plt.show()


1/0

# err = run_all_regressions(x, y, regs="regressions/reg_lists/five_best.py", verbose=True, show=False, x_test=0.1, final_verbose=False, selection_algo=sel, seed=0, save_all_fit_regs=True, split_func=split_func_for_reg(year))

# reg = err[0]['reg'][1]

# c = 0+reg.coef_
# c = c/np.linalg.norm(c)
# xx = np.dot(x, np.diag(c))

# x = xx


sel = Uniform_MAB(1, 3)
err = run_all_regressions(x, y, regs="regressions/reg_lists/five_best.py", verbose=True, show=False, x_test=0.1, final_verbose=True, selection_algo=sel, seed=0, save_all_fit_regs=True, split_func=split_func_for_reg(year))


1/0

xx_test = np.dot(x_test, np.diag(c))
dd = sklearn.metrics.pairwise.euclidean_distances(xx, xx)

means = []
for i,v in enumerate(xx_test):
    ind = np.argsort(dd[:,i])
    means += [  np.mean(y_train[ind[1:4]])  ]

np.mean((y_test - means)**2)




1/0
dd = sklearn.metrics.pairwise.euclidean_distances(xx, xx)
aa = sorted(range(dd.shape[0]), key=lambda x:np.sum(dd[x,:]))
dd1 = dd[aa,:][:,aa]
yy1 = y[aa]
y1 = np.repeat(yy1[:,np.newaxis], 3394, axis=1)
y2 = np.repeat(yy1[np.newaxis,:], 3394, axis=0)
yy2 = y1 - y2

points = [(i,j) for i,j in zip(yy2.flatten(),dd1.flatten())]

points_x = yy2.flatten()
points_y = dd1.flatten()




for b in np.unique(year):
    print(b)
    x2 = x[year==b,:]
    y2 = y[year==b]
    err = run_all_regressions(x2, y2, regs="regressions/reg_lists/five_best.py", verbose=True, show=False, x_test=0.1, final_verbose=False, selection_algo=sel, seed=None, save_all_fit_regs=True)


# err = run_all_regressions(x, y, regs="regressions/reg_lists/five_best.py", verbose=True, show=False, x_test=0.1, final_verbose=range(4), selection_algo=sel, seed=None, split_func=split_func_for_reg(year), save_all_fit_regs=True)


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


err = run_all_regressions(x_year, y_year, regs="regressions/reg_lists/one_of_each.py", verbose=True, show=False, x_test=0.1, final_verbose=False, selection_algo=sel, seed=5, split_func=split_func_for_reg(year_year))




