from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2, f_classif
# ----------------------------------------------------


# [1] Select Percentile

from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectPercentile, chi2

X, y = load_digits(return_X_y=True)
X_new = SelectPercentile(score_func=chi2, percentile=10).fit_transform(X, y)
print(X.shape) # (1797, 64)
print(X_new.shape) # (1797, 7)

from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectPercentile, chi2

data = load_breast_cancer()
sel = SelectPercentile(score_func=chi2, percentile=20).fit_transform(X, y)

print(X.shape) #(569, 30)
print(sel.shape) #(569, 6)


from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectPercentile, chi2

X, y = load_digits(return_X_y=True)

X_new = SelectPercentile(score_func=chi2, percentile=10)
selected = X_new.fit_transform(X,y)
print(X.shape) #(1797, 64)
print(selected.shape) #(1797, 7)

print(X_new.get_support())


# [2] Generic Univariate Select

#Import Libraries
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import chi2 , f_classif
#----------------------------------------------------
X, y = load_breast_cancer(return_X_y=True)


transformer = GenericUnivariateSelect(score_func= chi2, mode= 'k_best', param=3)
X_new = transformer.fit_transform(X, y)

print(X.shape)
print(X_new.shape)

transformer.get_support()

# [3] Feature Selection by KBest

from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, chi2

X, y = load_digits(return_X_y=True)
X_new = SelectKBest(chi2, k=30).fit_transform(X, y)

print(X.shape)
print(X_new.shape)


# [4] Select From Model


#Import Libraries
from sklearn.feature_selection import SelectFromModel
#----------------------------------------------------


from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

data = load_breast_cancer()
X = data.data
y = data.target

sel = SelectFromModel(RandomForestClassifier(n_estimators = 20))
sel.fit(X,y)
selected_features = sel.transform(X)
sel.get_support()  ## True mean this feature is selected and false mean not select