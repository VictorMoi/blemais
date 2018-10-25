reg = [

    ("SVR", SVR()),
    ("Kernel Ridge RBF", Regression_With_Custom_Kernel(KernelRidge(), RBF())),
    ("Kernel Ridge Cauchy", Regression_With_Custom_Kernel(KernelRidge(), Cauchy())),
    ("Kernel Ridge Tanimoto", Regression_With_Custom_Kernel(KernelRidge(), Tanimoto()))
]



regressions = [] + reg

from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import f_regression
#from sklearn.feature_selection import mutual_info_regression
#from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import SelectPercentile
#from sklearn.feature_selection import SelectFpr
#from sklearn.feature_selection import SelectFdr
#from sklearn.feature_selection import SelectFwe
#from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.pipeline import Pipeline

# regressions += [("RFE + " + i[0], Pipeline([('RFE', RFE(estimator=i[1])), i])) for i in regressions]


from sklearn.decomposition import PCA
#pca = PCA(n_components=20)



#regressions += [("PCA 20 + " + i[0], Pipeline([('PCA 20', PCA(n_components=20)), i])) for i in reg]

#regressions += [("PCA 10 + " + i[0], Pipeline([('PCA 10', PCA(n_components=10)), i])) for i in reg]
#
#regressions += [("RFE + " + i[0], Pipeline([('RFE', RFE(estimator=i[1])), i])) for i in reg]
#
#regressions += [("RFECV + " + i[0], Pipeline([('RFECV', RFECV(estimator=i[1])), i])) for i in reg]
#
#regressions += [("Var Thresh + " + i[0], Pipeline([('Var Thresh', VarianceThreshold(threshold=0.1)), i])) for i in reg]
#
#regressions += [("Select + " + i[0], Pipeline([('Select', SelectFromModel(estimator=i[1])), i])) for i in reg]
#
#regressions += [("Select K best + " + i[0], Pipeline([('Select K best', SelectKBest(f_regression, k=2)), i])) for i in reg]
#
#regressions += [("Select Percentile + " + i[0], Pipeline([('Select Percentile', SelectPercentile(f_regression, percentile=30)), i])) for i in reg]
#
#regressions += [("Select fpr + " + i[0], Pipeline([('Select fpr', SelectFpr(f_regression, alpha=0.3)), i])) for i in reg]
#
#regressions += [("Select fdr + " + i[0], Pipeline([('Select fdr', SelectFdr(f_regression, alpha=0.3)), i])) for i in reg]
#
#regressions += [("Select fwe + " + i[0], Pipeline([('Select fwe', SelectFwe(f_regression, alpha=0.3)), i])) for i in reg]
#

