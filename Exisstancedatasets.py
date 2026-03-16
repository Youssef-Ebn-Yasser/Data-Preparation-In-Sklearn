import  sklearn
from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer, load_diabetes, make_regression, \
    make_classification, load_sample_image
import numpy as np

## [1]  flowers data
IrisData = load_iris()

# this will print all feature data for X
print('dataset /n',IrisData.data)
# this will print all Column Names for feature
print(IrisData.feature_names)
# this will print the shape of the dara like ( a,b) a represent number of rows and b represent number of column
print('shape of dataset is ' , IrisData.data.shape)
# this will print all data output y
print('y Data is \n' , IrisData.target)
print('y shape is ' , IrisData.target.shape)  ## (150,)  means one column with 150 row
# this will print all Column Names for output
print('y Columns are \n' , IrisData.target_names)


## [2] Image of numbers Data (data is pixels )
DigitsData = load_digits()

X = DigitsData.data # pixels for image every image represented in 64 pixel
print('X Data is \n' , X[24])
print('X shape is ' , X.shape)
#y Data
y = DigitsData.target
print('y Data is \n' , y[22])
print('y shape is ' , y.shape)

import matplotlib.pyplot as plt
plt.gray()

print('Images of Number : ' , y[22])
# plt.matshow(DigitsData.images[y[22]])
print('------------------------------')
plt.show()


# [3] load wine data
WineData = load_wine()

#X Data
X = WineData.data
print('X Data is \n' , X[:10])
print('X shape is ' , X.shape)
print('X Features are \n' , WineData.feature_names)

#y Data
y = WineData.target
print('y Data is \n' , y[:10])
print('y shape is ' , y.shape)
print('y Columns are \n' , WineData.target_names)

#[4] Breast Cancer Data

#load breast cancer data
BreastData = load_breast_cancer()

#X Data
X = BreastData.data
print('X Data is \n' , X[:10])
print('X shape is ' , X.shape)
print('X Features are \n' , BreastData.feature_names)

#y Data
y = BreastData.target
print('y Data is \n' , y[:10])
print('y shape is ' , y.shape)
print('y Columns are \n' , BreastData.target_names)

#[5] Diabetes Data
#load diabetes data
DiabetesData= load_diabetes()

#X Data
X = DiabetesData.data
print('X Data is \n' , X[:10])
print('X shape is ' , X.shape)
print('X Features are \n' , DiabetesData.feature_names)

#y Data
y = DiabetesData.target
print('y Data is \n' , y[:10])
print('y shape is ' , y.shape)

#[6] Sample Regression Data

#load regression data
'''
X ,y = make_regression(n_samples=100, n_features=100, n_informative=10,
                       n_targets=1, bias=0.0, effective_rank=None,
                       tail_strength=0.5, noise=0.0, shuffle=True, coef=False,
                       random_state=None)
'''

X ,y = make_regression(n_samples=10000, n_features=500,shuffle=True)

#X Data
print('X Data is \n' , X[:10])
print('X shape is ' , X.shape)

#y Data
print('y Data is \n' , y[:10])
print('y shape is ' , y.shape)



#[7] Sample classification Data
#load classification data
'''
X, y = make_classification(n_samples = 100, n_features = 20, n_informative = 2, n_redundant = 2,
                           n_repeated = 0, n_classes = 2, n_clusters_per_class = 2, weights = None,
                           flip_y = 0.01, class_sep = 1.0, hypercube = True, shift = 0.0,
                           Scale() = 1.0, shuffle = True, random_state = None)
'''

X, y = make_classification(n_samples = 100, n_features = 20, shuffle = True)

#X Data
print('X Data is \n' , X[:10])
print('X shape is ' , X.shape)

#y Data
print('y Data is \n' , y[:10])
print('y shape is ' , y.shape)


#[8] Sample classification Data
china = load_sample_image('china.jpg')
china.dtype
china.shape


flower = load_sample_image('flower.jpg')
flower.dtype
flower.shape

import matplotlib.pyplot as plt
plt.imshow(china)
plt.imshow(flower)