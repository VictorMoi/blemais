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
plt.scatter(yp, y_tr, 'k')
plt.show()


yp = err[0]['reg'][1].predict(x_te)
plt.plot([-4,4], [-4,4], 'r')
plt.scatter(yp, y_te, '.k')
plt.show()
