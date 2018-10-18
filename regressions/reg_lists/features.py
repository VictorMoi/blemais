reg = [

    ("Linear Regression", LinearRegression()), 
    ("Ridge", Ridge(alpha = .5)),
    ("RidgeCV", RidgeCV(alphas=[0.1, 1.0, 10.0])),
    ("Lasso", Lasso(alpha = 0.1)),
    ("LassoCV", LassoCV(alphas=[0.1, 1.0, 10.0])),
    ("LassoLars", LassoLars(alpha = 0.1)),
    ("LassoLarsCV", LassoLarsCV()),
    ("LassoLarsIC", LassoLarsIC()),
    ("ElasticNet", ElasticNet(alpha=1.0, l1_ratio=0.5)),
    ("ElasticNetCV", ElasticNetCV(alphas=[0.1, 1.0, 10.0], l1_ratio=[0.1, 0.5, 0.9])),
    ("Lars", Lars()),
    ("LarsCV", LarsCV()),
    ("Orthogonal Matching Pursuit", OrthogonalMatchingPursuit(n_nonzero_coefs = 1)),
    ("Bayesian Ridge", BayesianRidge()),
    #("ARD Regression", ARDRegression()),
    ("SGD Regression", SGDRegressor()),
    ("Passive Aggressive Regression", PassiveAggressiveRegressor()),
    ("RANSAC Regression", RANSACRegressor()),
    #("Theil Sen Regression", TheilSenRegressor()),
    ("Huber Regression", HuberRegressor()),
    ("Kernel Ridge", KernelRidge()),
    ("Gaussian Process Regression", GaussianProcessRegressor()),
    ("SVR", SVR()),
    ("Multi Task Lasso", MultiTaskLasso()),
    ("Multi Task Lasso CV", MultiTaskLassoCV()),
    ("Multi Task Elastic Net", MultiTaskElasticNet()),
    ("Multi Task Elastic Net CV", MultiTaskElasticNetCV()),
    ("Kernel Ridge RBF", Regression_With_Custom_Kernel(KernelRidge(), RBF())),
    ("Kernel Ridge Linear", Regression_With_Custom_Kernel(KernelRidge(), Linear())),
    ("Kernel Ridge Polynomial", Regression_With_Custom_Kernel(KernelRidge(), Polynomial())),
    ("Kernel Ridge Cossim", Regression_With_Custom_Kernel(KernelRidge(), Cossim())),
    ("Kernel Ridge Exponential", Regression_With_Custom_Kernel(KernelRidge(), Exponential())),
    ("Kernel Ridge Laplacian", Regression_With_Custom_Kernel(KernelRidge(), Laplacian())),
    ("Kernel Ridge RationalQuadratic", Regression_With_Custom_Kernel(KernelRidge(), RationalQuadratic())),
    ("Kernel Ridge InverseMultiquadratic", Regression_With_Custom_Kernel(KernelRidge(), InverseMultiquadratic())),
    ("Kernel Ridge Cauchy", Regression_With_Custom_Kernel(KernelRidge(), Cauchy())),
    ("Kernel Ridge TStudent", Regression_With_Custom_Kernel(KernelRidge(), TStudent())),
    #("Kernel Ridge ANOVA", Regression_With_Custom_Kernel(KernelRidge(), ANOVA())),
    #("Kernel Ridge Wavelet", Regression_With_Custom_Kernel(KernelRidge(), Wavelet())),
    #("Kernel Ridge Fourier", Regression_With_Custom_Kernel(KernelRidge(), Fourier())),
    ("Kernel Ridge Tanimoto", Regression_With_Custom_Kernel(KernelRidge(), Tanimoto())),
    ("Kernel Ridge Sorensen", Regression_With_Custom_Kernel(KernelRidge(), Sorensen())),
    #("Kernel Ridge AdditiveChi2", Regression_With_Custom_Kernel(KernelRidge(), AdditiveChi2())),
    #("Kernel Ridge Chi2", Regression_With_Custom_Kernel(KernelRidge(), Chi2())),
    #("Kernel Ridge Min", Regression_With_Custom_Kernel(KernelRidge(), Min())),
    #("Kernel Ridge GeneralizedHistogramIntersection", Regression_With_Custom_Kernel(KernelRidge(), GeneralizedHistogramIntersection())),
    #("Kernel Ridge MinMax", Regression_With_Custom_Kernel(KernelRidge(), MinMax())),
    #("Kernel Ridge Spline", Regression_With_Custom_Kernel(KernelRidge(), Spline())),
    #("Kernel Ridge Log", Regression_With_Custom_Kernel(KernelRidge(), Log())),
    #("Kernel Ridge Power", Regression_With_Custom_Kernel(KernelRidge(), Power()))

]

regressions = [] + reg

from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import SelectFdr
from sklearn.feature_selection import SelectFwe
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.pipeline import Pipeline

# regressions += [("RFE + " + i[0], Pipeline([('RFE', RFE(estimator=i[1])), i])) for i in regressions]


from sklearn.decomposition import PCA
#pca = PCA(n_components=20)



regressions += [("PCA 20 + " + i[0], Pipeline([('PCA 20', PCA(n_components=20)), i])) for i in reg]

# regressions += [("PCA 10 + " + i[0], Pipeline([('PCA 10', PCA(n_components=10)), i])) for i in reg]

# regressions += [("RFE + " + i[0], Pipeline([('RFE', RFE(estimator=i[1])), i])) for i in reg]

# regressions += [("RFECV + " + i[0], Pipeline([('RFECV', RFECV(estimator=i[1])), i])) for i in reg]

# regressions += [("Var Thresh + " + i[0], Pipeline([('Var Thresh', VarianceThreshold(threshold=0.1)), i])) for i in reg]

# regressions += [("Select + " + i[0], Pipeline([('Select', SelectFromModel(estimator=i[1])), i])) for i in reg]

# regressions += [("Select K best + " + i[0], Pipeline([('Select K best', SelectKBest(f_regression, k=2)), i])) for i in reg]

# regressions += [("Select Percentile + " + i[0], Pipeline([('Select Percentile', SelectPercentile(f_regression, percentile=30)), i])) for i in reg]

# regressions += [("Select fpr + " + i[0], Pipeline([('Select fpr', SelectFpr(f_regression, alpha=0.3)), i])) for i in reg]

# regressions += [("Select fdr + " + i[0], Pipeline([('Select fdr', SelectFdr(f_regression, alpha=0.3)), i])) for i in reg]

# regressions += [("Select fwe + " + i[0], Pipeline([('Select fwe', SelectFwe(f_regression, alpha=0.3)), i])) for i in reg]


