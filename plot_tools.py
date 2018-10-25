import numpy as np
import matplotlib.pyplot as plt



split_func = split_func_for_reg(year)


x_tr, x_te, y_tr, y_te = split_func(x, y, test_size=0.1, random_state=5)


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



from copy import deepcopy as cp



sel = Uniform_MAB(1, 1)

reg = []

for i in range(10):
    err = run_all_regressions(x, y, regs="regressions/reg_lists/five_best.py", verbose=True, show=False, x_test=0.1, final_verbose=False, selection_algo=sel, seed=i, split_func=split_func_for_reg(year))
    reg.append(cp(err[0]['reg'][1]))


x_tr, x_te, y_tr, y_te = split_func(x, y, test_size=0.1, random_state=5)
