# Here we code the predictions models


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

# from pykernels.pykernels.basic import RBF




X = np.array([[1,1], [0,0], [1,0], [0,1]])
y = np.array([1, 1, 0, 0])

#X = np.concatenate([np.random.randn(2,100), np.random.randn(2,100) + [[2],[2]]], axis = 1)
X = np.concatenate([np.random.randn(100,3), np.random.randn(100,3) + [2,2,2]], axis = 0)
y = np.concatenate([np.ones(100), np.zeros(100)], axis = 0)

X = np.concatenate([np.random.randn(100,2), np.random.randn(100,2) + [2,2]], axis = 0)
y = np.concatenate([np.ones(100), np.zeros(100)], axis = 0)


from sklearn.kernel_ridge import KernelRidge
clf = KernelRidge(kernel='linear')
clf = KernelRidge(kernel='rbf')

clf.fit(X,y)
#prediction = clf.predict(X)

score = clf.score(X, y)



# print 'Testing XOR'

# for clf, name in [(SVC(kernel=RBF(), C=1000), 'pykernel'), (SVC(kernel='rbf', C=1000), 'sklearn')]:
#     clf.fit(X, y)
#     print name
#     print clf
#     print 'Predictions:', clf.predict(X)
#     print 'Accuracy:', accuracy_score(clf.predict(X), y)


    
