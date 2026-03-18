# [1] train_test_split

import numpy as np
from sklearn.model_selection import train_test_split
X, y = np.arange(10).reshape((5, 2)), range(5)
print(X)
print('----------------------------------')
print(y)
print('----------------------------------')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=41)
print('----------------------------------')
print(X_train)
print('----------------------------------')
print(y_train)
print('----------------------------------')
print(X_test)
print('----------------------------------')
print(y_test)
print('----------------------------------')
# يمكن ان يتم عمل تقسيم داخلي فيها بحيث يكون عشوائي او غير عشوائي
train_test_split(y, shuffle=True)

# [2] KFold
import numpy as np
from sklearn.model_selection import KFold

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8],[7, 8],[7, 8],[7, 8],[7, 8]])
y = np.array([11, 22, 33, 44,22,11,22,44])

nn = 3

kf = KFold(n_splits=nn)
kf.get_n_splits(X)
KFold(n_splits=nn, random_state=None, shuffle=False)

for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    print('X_train \n' , X_train)
    print('X_test \n' , X_test)
    print('y_train \n' ,y_train)
    print('y_test \n' , y_test)


# [3] Repeated KFold
import numpy as np
from sklearn.model_selection import RepeatedKFold
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([11, 22, 33, 44])

rkf = RepeatedKFold(n_splits=4, n_repeats=4, random_state=44)
for train_index, test_index in rkf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    print('X_train \n' , X_train)
    print('X_test \n' , X_test)
    print('y_train \n' ,y_train)
    print('y_test \n' , y_test)
    print('*********************')

import numpy as np
from sklearn.model_selection import RepeatedKFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
random_state = 12883823
rkf = RepeatedKFold(n_splits=2, n_repeats=3, random_state=random_state)
for train, test in rkf.split(X):
    print("%s %s" % (train, test))

# [4] Stratified KFold

import numpy as np
from sklearn.model_selection import StratifiedKFold
X = np.array([[1, 2,1, 2], [3, 4,1, 2], [5, 6,1, 2], [7, 8,1, 2],[1, 2,1, 2], [3, 4,1, 2], [5, 6,1, 2], [7, 8,1, 2]])
y = np.array([2,2,2,1,1,2,1,1])
skf = StratifiedKFold(n_splits=3)
skf.get_n_splits(X, y)

print(skf)

for train_index, test_index in skf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print('X_train \n' , X_train)
    print('X_test \n' , X_test)
    print('y_train \n' ,y_train)
    print('y_test \n' , y_test)
    print('*********************')


# [5] Repeated Stratified KFold
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([0,0,1,1])
rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=2,random_state=36851234)
for train_index, test_index in rskf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print('X_train \n' , X_train)
    print('X_test \n' , X_test)
    print('y_train \n' ,y_train)
    print('y_test \n' , y_test)
    print('*********************')


# [6] Leave One Out
import numpy as np
from sklearn.model_selection import LeaveOneOut

X = np.array([1, 2, 3, 4])
y = np.array([5, 6, 7, 8])

loo = LeaveOneOut()

for train_index, test_index in loo.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print('X_train \n', X_train)
    print('X_test \n', X_test)
    print('y_train \n', y_train)
    print('y_test \n', y_test)
    print('*********************')


# another example

X = [1, 2, 3, 4,1, 2, 3, 4]
loo = LeaveOneOut()
for train, test in loo.split(X):
    print("%s %s" % (train, test))
    print('*********************')


# [7] Leave P Out
import numpy as np
from sklearn.model_selection import LeavePOut

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([1, 2, 3, 4])

lpo = LeavePOut(3)

lpo.get_n_splits(X)

print(lpo)
LeavePOut(p=2)
for train_index, test_index in lpo.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print('X_train \n', X_train)
    print('X_test \n', X_test)
    print('y_train \n', y_train)
    print('y_test \n', y_test)
    print('*********************')

#another example

X = np.ones(6)
lpo = LeavePOut(p=3)
for train, test in lpo.split(X):
    print("%s %s" % (train, test))


# [8] Shuffle Split
import numpy as np
from sklearn.model_selection import ShuffleSplit

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [3, 4], [5, 6]])
y = np.array([1, 2, 1, 2, 1, 2])
rs = ShuffleSplit(n_splits=5, test_size=.1, random_state=0)
rs.get_n_splits(X)

print(rs)

for train_index, test_index in rs.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)

rs = ShuffleSplit(n_splits=5, train_size=0.5, test_size=.25, random_state=0)
for train_index, test_index in rs.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)

# another example
import numpy as np
from sklearn.model_selection import ShuffleSplit

X = np.arange(10)

n = 0.3

ss = ShuffleSplit(n_splits=5, test_size=n, random_state=0)
for train_index, test_index in ss.split(X):
    print("%s %s" % (train_index, test_index))

# [9] Stratified Shuffle Split

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 0, 1, 1, 1])
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
sss.get_n_splits(X, y)
print(sss)
for train_index, test_index in sss.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print('X_train \n' , X_train)
    print('X_test \n' , X_test)
    print('y_train \n' ,y_train)
    print('y_test \n' , y_test)
    print('*********************')

# [10] Time Series Split

import numpy as np
from sklearn.model_selection import TimeSeriesSplit
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4, 5, 6])
tscv = TimeSeriesSplit(n_splits=5)
print(tscv)
for train_index, test_index in tscv.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print('X_train \n' , X_train)
    print('X_test \n' , X_test)
    print('y_train \n' ,y_train)
    print('y_test \n' , y_test)
    print('*********************')