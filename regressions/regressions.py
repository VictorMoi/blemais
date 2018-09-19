#from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os
import time
#from matplotlib.colors import ListedColormap

# from pykernels.pykernels.basic import *

# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import LassoLarsIC
from sklearn.linear_model import MultiTaskLasso
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.linear_model import Lars
from sklearn.linear_model import LarsCV
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ARDRegression
# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import HuberRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.pipeline import Pipeline
from sklearn.svm import SVR


def help():
    print """
Regressions, ie takes an array as input and outputs a float

Uses the package scikit-learn 

This link explains the possible regressions choices :
    http://scikit-learn.org/stable/modules/linear_model.html

Interesting as well :
    http://scikit-learn.org/stable/model_selection.html

Non exhaustive list of possible regressions :
    Ridge(alpha = .5)
    RidgeCV(alphas=[0.1, 1.0, 10.0])
    Lasso(alpha = 0.1)
    LassoCV(alphas=[0.1, 1.0, 10.0])
    LassoLars(alpha = 0.1)
    LassoLarsCV()
    LassoLarsIC()
    ElasticNet(alpha=1.0, l1_ratio=0.5)
    ElasticNetCV(alphas=[0.1, 1.0, 10.0], l1_ratio=[0.1, 0.5, 0.9])
    Lars()
    LarsCV()
    OrthogonalMatchingPursuit(n_nonzero_coefs = 1)
    BayesianRidge()
    ARDRegression()
    SGDRegressor()
    PassiveAggressiveRegressor()
    RANSACRegressor()
    TheilSenRegressor()
    HuberRegressor()
    KernelRidge()
    GaussianProcessRegressor()
    SVR()

Example of use :
    # Create data
    x_train = np.random.randn(1000,2)
    y_train = np.dot(x_train, [2,5]) + np.random.randn(1000)/10.
    x_test = np.random.randn(1000,2)
    y_test = np.dot(x_test, [2,5]) + np.random.randn(1000)/10.
    # Data more plottable here :
    x_train = np.random.randn(1000,1)
    y_train = x_train*3 + 2 + np.random.randn(1000,1)/10.
    x_test = np.random.randn(1000,1)
    y_test = x_test*3 + 2 + np.random.randn(1000,1)/10.
    
    # Chose regression algorithm
    reg = LinearRegression()

    # Fit the model
    reg.fit(x_train, y_train)

    # Make prediction
    pred_y_train = reg.predict(x_train)
    pred_y_test = reg.predict(x_test)
    
    # Compute and print mean square error
    ms_err_train = mean_squared_error(pred_y_train, y_train)
    ms_err_test = mean_squared_error(pred_y_test, y_test)
    print "Train mean square error : {}".format(ms_err_train)
    print "Test mean square error : {}".format(ms_err_test)

    # Show regression
    show_regression(reg, x_test, y_test)
"""


    
def show_regression(reg, x, y):
    """
    Print a 2D decision space
    """
    if (len(x.shape) > 1) and (x.shape[1] != 1):
        print "Warning : you cannot use the function show_regression if you are not in 1D"
    else:
        if (len(x.shape) == 1):
            x_min = np.min(x)
            x_max = np.max(x)
        else:
            x_min = np.min(x[:,0])
            x_max = np.max(x[:,0])
        x_span = x_max-x_min
        x_min = x_min - x_span/20.
        x_max = x_max + x_span/20.
        h = x_span/100
        xl = np.arange(x_min, x_max, h).reshape(-1, 1)
        yl = reg.predict(xl)
        plt.plot(x, y, '.k')
        plt.plot(xl, yl, 'r')
        plt.show()




def get_regressions(n=0):
    """
    Return a list of regressions, bigger or smaller depending of the value of n
    if n is negative, it returns only one regression
    if n is zero, it returns one regression of each type
    if n is strictly positive, it returns more classfiers (Not implemented yet)
    otherwise, it returns an empty list
    """
    try:
        if (type(n) == int):
            if (n < 0):
                regressions = [("Nearest Neighbors", KNeighborsRegression(3))]
            elif (n == 0):
                this_file_path = '/'.join(__file__.split('/')[:-1])
                with open(os.path.join(this_file_path, "reg_lists/one_of_each.py")) as f:
                    r = f.read()
                    exec(r)
            else:
                regressions = []
        elif (type(n) == str):
            if (n[-3:] == ".py"):
                with open(n) as f:
                    r = f.read()
                    exec(r)
            else:
                exec(r)
        else:
            regressions = []
    except:
        print "Error while loading a list of regressions, the error is likely to be in the argument n."
        raise
    return [(i[1], i[0]) for i in regressions]



def run_one_regression(x_train, y_train, reg, error_func=mean_squared_error, x_test=None, y_test=None, verbose=True, show=True, i="", _error_test=None):
    # We define reg and name etc
    reg, name = _get_reg_attributes(reg)
    # We run the regression
    try:
        reg.fit(x_train, y_train)
        if (x_test is None) or (y_test is None):
            error_train = error_func(reg.predict(x_train), y_train)
            error_test = _error_test
        else:
            error_train = error_func(reg.predict(x_train), y_train)
            error_test = error_func(reg.predict(x_test), y_test)
            x_train = np.concatenate([x_train, x_test], axis=0)
            y_train = np.concatenate([y_train, y_test], axis=0)
        if show:
            t = _repr_show(i, name, error_train, error_test)
            plt.title(t)
            show_regression(reg, x_train, y_train)
        if verbose:
            t = _repr_verbose(i, name, error_train, error_test)
            print t
    except ValueError:
        print "Kernel {} failed with the data provided".format(name)
        return (0, 0)
    except KeyboardInterrupt:
        raise
    except:
        print "Kernel {} failed".format(name)
        return (0, 0)
    return (error_train, error_test)



def _get_reg_attributes(reg):
    # Return reg and name for a dict or tuple regression
    name = ""
    if (type(reg) == tuple):
        rg = reg[0]
        if len(reg) > 1:
            name = reg[1]
    elif (type(reg) == dict):
        rg = reg["reg"]
        if "name" in reg.keys():
            name = reg["name"]
    else:
        rg = reg
    return rg, name
    


def _repr_show(i, name, error_train, error_test=None):
    # Representation for a plot title when we test a regression
    t = "{}\n".format(i)
    t += ("" if (name == "") else ("name : {}\n".format(name)))
    t += "error_train : {0:.3f}".format(error_train)
    t += ("" if (error_test is None) else "\nerror_test : {0:.3f}".format(error_test))
    return t



def _repr_verbose(i, name, error_train, error_test=None):
    # Representation of a verbose line when we test a regression
    t = "{}".format(i)
    t += " : error_train : {0:.3f}".format(error_train)
    t += ("" if (error_test is None) else " : error_test : {0:.3f}".format(error_test))
    t += ("" if (name == "") else ("   -   name : {}".format(name)))
    return t
            


def _verbose_show_proper(length, verbshow):
    # We properly define verbose (or show), ie it will be a list of bool
    if (type(verbshow) == bool):
        res = [verbshow for i in range(length)]
    elif ((type(verbshow) == list) or (type(verbshow) == tuple) or (type(verbshow) == np.ndarray)):
        if (len(verbshow) >= length) and (type(verbshow[0]) == bool):
            res = [i for i in verbshow[:length]]
        else:
            res = [False for i in range(length)]
            for i in verbshow:
                if (i < len(res)):
                    res[i] = True
    else:
        res = [False for i in range(length)]
    return res



def run_all_regressions(x_train, y_train, regs=0, error_func=mean_squared_error, verbose=True, show=False, x_test=None, y_test=None, selection_algo=None, final_verbose=range(10), final_show=False, sort_key=lambda x:x["error_test"]):
    """
    Try a lot of different regressions, and can show some of them
    """
    # We define regs
    if (type(regs) == int):
        regs = get_regressions(0)
    # We properly define show, ie it will be a list of bool
    show = _verbose_show_proper(len(regs), show)
    verbose = _verbose_show_proper(len(regs), verbose)
    # We properly define test_size
    if (type(x_test) == int):
        test_size = float(x_test)/x_train.shape[0]
    elif (type(x_test) == float):
        test_size=x_test
    else:
        test_size=None
    # We run all the regressions following selection_algo
    nbr_ex = 0
    start_time = time.time()
    if selection_algo is None:
        # In this section there are no particular reg selection
        # We separate the train test data if asked of
        if x_test is None:
            x_tr, x_te, y_tr, y_te = (x_train, x_test, y_train, y_test)
        else:
            x_tr, x_te, y_tr, y_te = train_test_split(x_train, y_train, test_size=test_size)
        # We try over all regressions
        errors = []
        for ic, sho, verb, reg in zip(range(len(show)), show, verbose, regs):
            nbr_ex += 1
            tr, te = run_one_regression(x_tr, y_tr, reg, error_func, x_te, y_te, verbose=verb, show=sho, i=ic)
            errors.append({"i":ic, "error_train":tr, "error_test":te, "reg":reg})
    else:
        # In this section we follow the class selection_algo to perform the regressions tests
        selection_algo.set(n_arms=len(regs))
        arm = selection_algo.next_arm()
        while (arm is not None):
            # We separate the train test data if asked of
            if x_test is None:
                x_tr, x_te, y_tr, y_te = (x_train, x_test, y_train, y_test)
            else:
                x_tr, x_te, y_tr, y_te = train_test_split(x_train, y_train, test_size=test_size)
            tr, te = run_one_regression(x_tr, y_tr, regs[arm], error_func, x_te, y_te, verbose[arm], show[arm], i=nbr_ex)
            selection_algo.update_reward(te, arm=arm, other_data=tr)
            arm = selection_algo.next_arm()
            nbr_ex += 1
        errors = []
        for ic, tr, te, reg in zip(range(len(regs)), selection_algo.other_data, selection_algo.mean_rewards, regs):
            errors.append({"i":ic, "error_train":np.mean(tr), "error_test":te, "reg":reg})
    # Now we have finished the tests of the regressions
    # We print the final results obtained
    final_show = _verbose_show_proper(nbr_ex, final_show)
    final_verbose = _verbose_show_proper(nbr_ex, final_verbose)
    if any(verbose) or any(final_verbose):
        print "\nFinished running {} examples in {} seconds\n".format(nbr_ex, time.time() - start_time)
    if any(final_verbose) or any(final_show):
        errors_sorted = sorted(errors, key=sort_key)
        for ic, ss, sho, verb in zip(range(len(final_show)), errors_sorted, final_show, final_verbose):
            run_one_regression(x_train, y_train, ss["reg"], error_func, verbose=verb, show=sho, i=ic, _error_test=ss["error_test"])
    return errors



def load_dataset(dataset="default"):
    if (dataset == "boston"):
        from sklearn.datasets import load_boston
        boston = load_boston()
        return preprocessing.scale(boston.data), preprocessing.scale(boston.target)
    x = np.random.randn(1000,1)
    y = x*3 + 2 + np.random.randn(1000,1)/10.
    return x, y
    # if (dataset == "moons"):
    #     from sklearn.datasets import make_moons
    #     return make_moons(noise=0.3, random_state=0)
    # elif (dataset == "circles"):
    #     from sklearn.datasets import make_circles
    #     return make_circles(noise=0.2, factor=0.5, random_state=1)
    

        
def run_examples(verbose=True, show=False, dataset="default"):
    """
    Run all type of regressions possible on an classical dataset
    """
    x, y = load_dataset(dataset)
    run_all_regressions(x, y, regs=0, verbose=verbose, show=show, x_test=0.1)



        
# def run_examples(x=None, y=None, show=True):
#     """
#     Run all type of regressions possible
#     """
#     if (x is None) or (y is None):
#         from sklearn.datasets import make_moons
#         x, y = make_moons(noise=0.3, random_state=0)
#     names = ["Nearest Neighbors", "Linear SVM", "Sigmoid SVM", "rbf SVM",
#              "RBF SVM", "Gaussian Process", "Decision Tree", "Random Forest",
#              "Neural Net", "AdaBoost", "Naive Bayes", "QDA"]
#     for i, reg in enumerate([KNeighborsClassifier(3),
#                              SVC(kernel="linear", C=0.025),
#                              SVC(kernel="sigmoid", C=0.025),
#                              SVC(kernel="rbf", C=0.025),
#                              SVC(gamma=2, C=1),
#                              GaussianProcessClassifier(1.0 * RBF(1.0)),
#                              DecisionTreeClassifier(max_depth=5),
#                              RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#                              MLPClassifier(alpha=1),
#                              AdaBoostClassifier(),
#                              GaussianNB(),
#                              QuadraticDiscriminantAnalysis()]):
#         reg.fit(x, y)
#         error = reg.error(x, y)
#         if show:
#             plt.title("name : {}\nerror : {}".format(names[i], error))
#             print_decision_space(reg, x, y)
#         else:
#             print "name : {}   -   error : {}".format(names[i], error)
