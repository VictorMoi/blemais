import numpy as np
import matplotlib.pyplot as plt



split_func = split_func_for_reg(year)

x_tr, x_te, y_tr, y_te = split_func(x, y, test_size=0.1, random_state=0)
x_tr, x_te, y_tr, y_te = split_func(x, y, test_size=0.1, random_state=1)
x_tr, x_te, y_tr, y_te = split_func(x, y, test_size=0.1, random_state=2)
x_tr, x_te, y_tr, y_te = split_func(x, y, test_size=0.1, random_state=3)
x_tr, x_te, y_tr, y_te = split_func(x, y, test_size=0.1, random_state=4)
x_tr, x_te, y_tr, y_te = split_func(x, y, test_size=0.1, random_state=5)


x_te.shape

yp = err[0]['reg'][1].predict(x_tr)
plt.plot([-4,4], [-4,4], 'r')
plt.scatter(yp, y_tr, color='k', s=5)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()


yp = err[0]['reg'][1].predict(x_te)
plt.plot([-4,4], [-4,4], 'r')
plt.scatter(yp, y_te, color='k', s=5)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()



for i in range(x_te.shape[1]):
    plt.scatter(x_te[:,i], y_te, color='k', s=5)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()











split_func = split_func_for_reg(year)

x_tr, x_te, y_tr, y_te = split_func(x, y, test_size=0.1, random_state=0)

    


from copy import deepcopy as cp



sel = Uniform_MAB(1, 1)

reg = []

all_x_tr = []#np.zeros((0, x.shape[1]))
all_y_tr = []#np.zeros(0)
all_x_te = []#np.zeros((0, x.shape[1]))
all_y_te = []#np.zeros(0)
all_x_tr2 = np.zeros((0, x.shape[1]))
all_y_tr2 = np.zeros(0)
all_x_te2 = np.zeros((0, x.shape[1]))
all_y_te2 = np.zeros(0)


for i in range(10):
    x_tr, x_te, y_tr, y_te = split_func(x, y, test_size=0.1, random_state=i)
    err = run_all_regressions(x, y, regs="regressions/reg_lists/five_best.py", verbose=True, show=False, x_test=0.1, final_verbose=False, selection_algo=sel, seed=i, split_func=split_func_for_reg(year))
    reg.append(cp(err[0]['reg'][1]))
    all_x_tr.append(x_tr) # = np.concatenate([all_x_tr, x_tr])
    all_y_tr.append(y_tr) # = np.concatenate([all_y_tr, y_tr])
    all_x_te.append(x_te) # = np.concatenate([all_x_te, x_te])
    all_y_te.append(y_te) # = np.concatenate([all_y_te, y_te])
    all_x_tr2 = np.concatenate([all_x_tr2, x_tr])
    all_y_tr2 = np.concatenate([all_y_tr2, y_tr])
    all_x_te2 = np.concatenate([all_x_te2, x_te])
    all_y_te2 = np.concatenate([all_y_te2, y_te])


    
yp = np.concatenate([r.predict(xte) for r, xte in zip(reg,all_x_te)])

#yp = err[0]['reg'][1].predict(all_x_te)
plt.plot([-4,4], [-4,4], 'r')
plt.scatter(yp, all_y_te2, color='k', s=5)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()



#for i in range(10):
for i in range(all_x_te2.shape[1]):
    plt.scatter(all_x_te2[:,i], all_y_te2, color='k', s=5)
    reg = ("Linear Regression", LinearRegression())
    run_one_regression(all_x_te2[:,i:i+1], all_y_te2, reg, error_func=mean_squared_error, x_test=None, y_test=None, verbose=False, show=False, i="", seed=None, debug=False)
    plt.plot([np.min(all_x_te2[:,i]), np.max(all_x_te2[:,i])], [0,0], 'r')
    plt.plot([np.min(all_x_te2[:,i]), np.max(all_x_te2[:,i])], [reg[1].coef_[0]*np.min(all_x_te2[:,i]), reg[1].coef_[0]*np.max(all_x_te2[:,i])], 'b')
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.title(xind2name[i])
    plt.show()


