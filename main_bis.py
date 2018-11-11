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
x,xind2name,xname2ind = delVar(x, xind2name, xname2ind, "NUMD")



#x,xind2name,xname2ind = delVar(x, xind2name, xname2ind, "IRR")

x_squared = copy(maize_squared)
x_squaredind2name = copy(maize_squaredind2name)
x_squaredname2ind = copy(maize_squaredname2ind)
x_squared,x_squaredind2name,x_squaredname2ind = delVar(x_squared, x_squaredind2name, x_squaredname2ind, "year_harvest")
x_squared,x_squaredind2name,x_squaredname2ind = delVar(x_squared, x_squaredind2name, x_squaredname2ind, "yield_anomaly")
x_squared,x_squaredind2name,x_squaredname2ind = delVar(x_squared, x_squaredind2name, x_squaredname2ind, "NUMD")

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
x_lobell,x_lobellind2name,x_lobellname2ind = selVar(x_lobell, x_lobellind2name, x_lobellname2ind, sel_lobell)

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


x_origin = copy(x_squared)
x_originind2name = copy(x_squaredind2name)
x_originname2ind = copy(x_squaredname2ind)
sel_origin = ['ETP_1','ETP_2','ETP_3','ETP_4','ETP_5','ETP_6','ETP_7','ETP_8','ETP_9',
'PR_1','PR_2','PR_3','PR_4','PR_5','PR_6','PR_7','PR_8','PR_9','RV_1','RV_2','RV_3','RV_4','RV_5','RV_6','RV_7','RV_8','RV_9',
'SeqPR_1','SeqPR_2','SeqPR_3','SeqPR_4','SeqPR_5','SeqPR_6','SeqPR_7','SeqPR_8','SeqPR_9',
'Tn_1','Tn_2','Tn_3','Tn_4','Tn_5','Tn_6','Tn_7','Tn_8','Tn_9',
'Tx_1','Tx_2','Tx_3','Tx_4','Tx_5','Tx_6','Tx_7','Tx_8','Tx_9',
'IRR']
x_origin,x_originind2name,x_originname2ind = selVar(x_origin, x_originind2name, x_originname2ind, sel_origin)

x_originsqrd = copy(x_squared)
x_originsqrdind2name = copy(x_squaredind2name)
x_originsqrdname2ind = copy(x_squaredname2ind)
sel_originsqrd = sel_origin + [
'ETP_1_sqrd','ETP_2_sqrd','ETP_3_sqrd','ETP_4_sqrd','ETP_5_sqrd','ETP_6_sqrd','ETP_7_sqrd','ETP_8_sqrd','ETP_9_sqrd',
'PR_1_sqrd','PR_2_sqrd','PR_3_sqrd','PR_4_sqrd','PR_5_sqrd','PR_6_sqrd','PR_7_sqrd','PR_8_sqrd','PR_9_sqrd',
'RV_1_sqrd','RV_2_sqrd','RV_3_sqrd','RV_4_sqrd','RV_5_sqrd','RV_6_sqrd','RV_7_sqrd','RV_8_sqrd','RV_9_sqrd',
'SeqPR_1_sqrd','SeqPR_2_sqrd','SeqPR_3_sqrd','SeqPR_4_sqrd','SeqPR_5_sqrd','SeqPR_6_sqrd','SeqPR_7_sqrd','SeqPR_8_sqrd','SeqPR_9_sqrd',
'Tn_1_sqrd','Tn_2_sqrd','Tn_3_sqrd','Tn_4_sqrd','Tn_5_sqrd','Tn_6_sqrd','Tn_7_sqrd','Tn_8_sqrd','Tn_9_sqrd',
'Tx_1_sqrd','Tx_2_sqrd','Tx_3_sqrd','Tx_4_sqrd','Tx_5_sqrd','Tx_6_sqrd','Tx_7_sqrd','Tx_8_sqrd','Tx_9_sqrd']
x_originsqrd,x_originsqrdind2name,x_originsqrdname2ind = selVar(x_originsqrd, x_originsqrdind2name, x_originsqrdname2ind, sel_originsqrd)




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


all_data = [(x,y,year), (x_reduced,y,year), (x_lobell,y,year), (x_lobell2,y,year), (x_an,y,year), (x_squared,y,year)]

















import numpy as np


from blocks.utils.base import *
from blocks.utils.custom import *
from blocks.utils.redirect import *
from blocks.dataset.matrix_dataset import *
from blocks.dataset.split_dataset import *
from blocks.regressor.regressor import *
from blocks.regressor.includes import *
from blocks.error_measure.prediction_error import *
from blocks.utils.decorators import *
from blocks.multi_armed_bandit.multi_armed_bandit import *

from processing.scale import *

import warnings
if os.name == 'posix':
    from sklearn.exceptions import ConvergenceWarning
    #from sklearn.exceptions import FutureWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)



def load(filepath):
    """
    Load data from a file
    filepath : (str) : the location of the file
    """
    if (filepath[-4:] != '.npy'):
        with open(filepath, 'r') as f:
            data = f.read()
    else:
        data = np.load(filepath)
        try:
            data = data.item()['']
        except ValueError:
            pass
    return data





class Repeat(Custom_Input_Block):
    def __init__(self, input_block, repeat_times):
        super(Repeat, self).__init__(input_block)
        self.repeat_times = repeat_times
        
    def custom_compute(self):
        output = []
        for x in xrange(self.repeat_times):
            output.append(self._input_block())
        return output


class Mean(Custom_Input_Block):
    def custom_compute(self):
        return np.mean(self._input_block(), axis=0)

class Select_Train(Custom_Input_Block):
    def custom_compute(self):
        return self._input_block()[0]

class Select_Test(Custom_Input_Block):
    def custom_compute(self):
        return self._input_block()[1]



from copy import deepcopy as cp
import time
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR








# # all_data = load("blocks/all_data.npy")
# # normal, reduced, lobell, lobell2, an, squared
# idat = 0

# lb = []

# d = Matrix_Dataset(all_data[idat][0], all_data[idat][1])
# y = Select_Train(Matrix_Dataset(all_data[idat][2], None))

# # s = Split_Dataset(d, test_size=0.1)
# # lb.append(s)

# #ss = Split_Dataset_N_Parts(d, 10, seed=0)
# #lb.append(s)

# s = Non_Uniform_Split_Dataset_N_Parts(d, y, 10, seed=0)
# lb.append(s)

# j = Join_Dataset_N_Parts(lb[-1], index=-1)
# lb.append(j)

# sc = Scaler_x(lb[-1])
# lb.append(sc)


# r = Regressor(lb[-1], regressor=SVR)
# lb.append(r)

# p = Prediction_Error_L2(lb[-1])
# lb.append(p)

# si = Run_Function_Before(lb[-1], lambda x=None:j.set_param(index=j.index+1), force=True)
# lb.append(si)



# rp = Repeat(lb[-1], 10)










# 1/0












# all_data = load("blocks/all_data.npy")
# normal, reduced, lobell, lobell2, an, squared
idat = 0

lb = []

# d = Matrix_Dataset(all_data[idat][0], all_data[idat][1])
# y = Select_Train(Matrix_Dataset(all_data[idat][2], None))
# lb.append(d)

yy = Select_Train(Matrix_Dataset(year, None))

d = []
#d += [Matrix_Dataset(x, y)]
d += [Matrix_Dataset(x, y)]
d += [Matrix_Dataset(x_lobell, y)]
d += [Matrix_Dataset(x_an, y)]
d += [Matrix_Dataset(x_origin, y)]
d += [Matrix_Dataset(x_originsqrd, y)]
d += [Matrix_Dataset(x_squared, y)]

sd = Select_Output(d, 0)
lb.append(sd)

# s = Split_Dataset(d, test_size=0.1)
# lb.append(s)

#ss = Split_Dataset_N_Parts(d, 10, seed=0)
#lb.append(s)

s = Non_Uniform_Split_Dataset_N_Parts(sd, y, 10, seed=0)
lb.append(s)

j = Join_Dataset_N_Parts(lb[-1], index=0)
lb.append(j)

sc = Scaler_x(lb[-1])
lb.append(sc)

r = []
# r += [Regressor(lb[-1], regressor=LinearRegression)]
# r += [Regressor(lb[-1], regressor=Ridge)]
# r += [Regressor(lb[-1], regressor=RandomForestRegressor)]
# r += [Regressor(lb[-1], regressor=SVR)]
# r += [Kernel_Regressor(lb[-1], regressor=SVR, kernel=RBF())]
# r += [Kernel_Regressor(lb[-1], regressor=SVR, kernel=Tanimoto())]
for reg in all_regressors:
    if reg().get_params().has_key("kernel"):
        for ker in all_kernels:
            if isinstance(ker, str):
                r.append(Kernel_Regressor(lb[-1], regressor=reg, kernel=ker))
            else:
                r.append(Kernel_Regressor(lb[-1], regressor=reg, kernel=ker()))
    else:
        r.append(Regressor(lb[-1], regressor=reg))
lb += r

so = Select_Output(r, 0)
lb.append(so)
t = Measure_Time(lb[-1])
lb.append(t)
# sn = Snapshot_Attributes(lb[-1], "time")
# lb.append(sn)
def verbose_func():
    print "{:<2} : train_err : {:<5}, test_err : {:<5}, time : {:<5} : {}".format(s.seed, round(p.output[0],3), round(p.output[1],3), round(t.time,3), r[so.index].name)

def error_func():
    print "Regression error : {}".format(r[so.index].name)
    return 0

def timeout_func():
    print "Regression timed out : {}".format(r[so.index].name)
    return 0

# v = Verbose_(Prediction_Error_L2(lb[-1]), func2)
# lb.append(v)
p = Prediction_Error_L2(lb[-1])
lb.append(p)
v = Verbose(lb[-1], verbose_func)
lb.append(v)
sl = Select_Test(lb[-1])
lb.append(sl)

to = Timeout(lb[-1], seconds=3, default_func=timeout_func)
lb.append(to)
e = Ignore_Exception(lb[-1], exception=[ValueError, TypeError], default_func=error_func)
lb.append(e)

# sl = Redirect_To_Test(lb[-1])
# lb.append(sl)

class Nbr_Calls:
    def __init__(self, s, so, sd, i, j):
        self.n = 0
        self.s = s
        self.so = so
        self.sd = sd
        self.i = i
        self.j = j
    def __call__(self):
        #self.s.set_param(seed=self.n)
        self.s.set_param(index=self.n)
        self.so.set_param(index=self.i)
        self.sd.set_param(index=self.j)
        self.n += 1

#f = Run_Function_Before(lb[-1], lambda x=None : setattr(s,"changed_here",True))
#f = Run_Function_Before(lb[-1], lambda x=None : s.set_seed(s.seed+1))
f = [Run_Function_Before(lb[-1], Nbr_Calls(j, so, sd, i, k), force=True) for i in xrange(len(r)) for k in xrange(len(d))]
lb += f
mab = Uniform_MAB(f, 1)
