# metrics.mean_absolute_error
# metrics.mean_squared_error
# metrics.median_absolute_error
# metrics.confusion_matrix
# metrics.accuracy_score
# metrics.f1_score
# metrics.recall_score
# metrics.precision_score
# metrics.precision_recall_fscore_support
# metrics.precision_recall_curve
# metrics.classification_report
# metrics.roc_curve
# metrics.auc
# metrics.roc_auc_score
# metrics.zero_one_loss




## mean_absolute_error
from sklearn.metrics import mean_absolute_error
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
mean_absolute_error(y_true, y_pred)

y_true = [[0.5, 1], [-1, 1], [7, -6]]
y_pred = [[0, 2], [-1, 2], [8, -5]]

print(mean_absolute_error(y_true, y_pred)) ## multioutput='uniform_average' is default
print(mean_absolute_error(y_true, y_pred, multioutput='uniform_average')) # 0.75
## take here with rows
print(mean_absolute_error(y_true, y_pred, multioutput='raw_values')) # array([0.5, 1. ])

#----------------------------------------------------

## mean_squared_error
from sklearn.metrics import mean_squared_error

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

mean_squared_error(y_true, y_pred)

y_true = [[0.5, 1],[-1, 1],[7, -6]]
y_pred = [[0, 2],[-1, 2],[8, -5]]

(y_true, y_pred)
print(mean_squared_error(y_true, y_pred, multioutput='uniform_average'))
print(mean_squared_error(y_true, y_pred, multioutput='raw_values'))

#----------------------------------------------------

## median_absolute_error
from sklearn.metrics import median_absolute_error

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

median_absolute_error(y_true, y_pred)

#----------------------------------------------------

## confusion_matrix

from sklearn.metrics import confusion_matrix

y_pred = ['a','a','b','b','a','b','a','a','a','a']
y_true  = ['a','b','b','a','b','a','a','b','a','b']
CM = confusion_matrix(y_true, y_pred)
print(CM)
## print [[3 2]
 #        [4 1]]

# import seaborn as sns
# import matplotlib.pyplot as plt

# drawing confusion matrix
# sns.heatmap(CM, center = True)
# plt.show()

#----------------------------------------------------

## accuracy_score

from sklearn.metrics import accuracy_score
y_pred = [0, 2, 1, 3,5,3]
y_true = [0, 1, 2, 3,5,3]

print(accuracy_score(y_true, y_pred)) # fraction of all Trues over everything
print(accuracy_score(y_true, y_pred, normalize=False)) #number of all Trues


#----------------------------------------------------

## f1_score
from sklearn.metrics import f1_score
y_pred = [0, 2, 1, 0, 0, 1]
y_true = [0, 1, 2, 0, 1, 2]
f1_score(y_true, y_pred, average='micro') #it can be : binary,macro,weighted,samples
#----------------------------------------------------

## recall_score
from sklearn.metrics import recall_score
y_pred =  ['a','b','c','a','b','c','a','b','c','a']
y_true =   ['a','a','b','b','a','b','c','c','b','b']
recall_score(y_true, y_pred, average=None)
#----------------------------------------------------

## precision_score
from sklearn.metrics import precision_score
y_pred = ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a']
y_true = ['a', 'a', 'b', 'b', 'a', 'b', 'c', 'c', 'b', 'b']
precision_score(y_true, y_pred, average=None)
#----------------------------------------------------

## precision_recall_fscore_support
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

y_pred = np.array(['cat', 'pig', 'dog', 'cat', 'cat', 'dog'])
y_true = np.array(['cat', 'dog', 'pig', 'cat', 'dog', 'pig'])

print(f"precision_recall_fscore_support{precision_recall_fscore_support(y_true, y_pred, average='micro')}")
#----------------------------------------------------

## precision_recall_curve
import numpy as np
from sklearn.metrics import precision_recall_curve
y_pred =  np.array([0, 0, 1, 1])
y_true =   np.array([0.1, 0.4, 0.35, 0.8])

precision, recall, thresholds = precision_recall_curve(y_pred,y_true)

print(precision)
print(recall)
print(thresholds)

#----------------------------------------------------

## classification_report
# from sklearn.metrics import classification_report
# y_true = [0, 1, 2, 2, 2,5]
# y_pred = [0, 0, 2, 2, 1,0]
# print(classification_report(y_true, y_pred ))
#


# from sklearn.metrics import classification_report
# y_true = ['a','d','a','g','a','d']
# y_pred = ['a','a','g','g','d','g']
# print(classification_report(y_true, y_pred ))
#----------------------------------------------------

## roc_curve
import numpy as np
from sklearn import metrics
y =      np.array([1    , 1     , 2     , 2])
scores = np.array([0.1  , 0.4   ,   0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)

print(f"fpr : {fpr}, tpr : {tpr}, thresholds  : {thresholds}")

##output
# fpr 		:	array([0. , 0. , 0.5, 0.5, 1. ])
# tpr 		:	Out[3]: array([0. , 0.5, 0.5, 1. , 1. ])
# thresholds  	:	Out[4]: array([1.8 , 0.8 , 0.4 , 0.35, 0.1 ])

"""
أي أنه حينما كانت الثريشهولد تساوي 1.8 , كانت tpr   وهي sensitivity  تساوي 0 , بينما كانت fpr  تساوي 0 اي ان ال specificity  تساوي 1
كذلك حينما كانت الثريشهولد تساوي 0.35 , كانت tpr   وهي sensitivity  تساوي 1 , بينما كانت fpr  تساوي 0.5 اي ان ال specificity  تساوي 0.5
"""

#----------------------------------------------------

## roc_curve
import numpy as np
from sklearn import metrics
y =      np.array([1    , 1     , 2     , 2])
scores = np.array([0.1  , 0.4   ,   0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)

print(metrics.auc(fpr, tpr))

#----------------------------------------------------

## roc_auc_score
import numpy as np
from sklearn import metrics
y =      np.array([1    , 1     , 2     , 2])
scores = np.array([0.1  , 0.4   ,   0.35, 0.8])
metrics.roc_auc_score(y, scores)

#----------------------------------------------------

## zero_one_loss
from sklearn.metrics import zero_one_loss
y_pred = [1, 2, 3, 4]
y_true = [2, 2, 3, 4]

print(zero_one_loss(y_true, y_pred))
print(zero_one_loss(y_true, y_pred, normalize=False))