from sklearn.preprocessing import StandardScaler

# [1] Standard Scaler    (Standardization)
data = [[33, 45], [12,34], [22, 63], [66, 24]]

scaler = StandardScaler()
print('----------------------------------')
newdata = scaler.fit_transform(data)
print(scaler.mean_)
print(newdata)
print('----------------------------------')


from sklearn.preprocessing import MinMaxScaler
# [2] MinMaxScaler
data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
scaler = MinMaxScaler()
scaler.fit(data)

print(scaler.data_range_)
print(scaler.data_min_)
print(scaler.data_max_)
print('----------------------------------')
newdata = scaler.transform(data)
print(newdata)
# we can change range
scaler = MinMaxScaler(feature_range = (1,5))
print('----------------------------------')



from sklearn.preprocessing import Normalizer

# [3] Normalizer

X = [[4, 1, 2, 2],
     [1, 3, 9, 3],
     [5, 7, 5, 1]]
# تستخدم l1  لجعل مجموع كل صف هو القيمة العظمي
#transformer = Normalizer(norm='l1' )
# تستخدم l2 لجعل جذر مجموع مربعات كل صف هو القيمة العظمي
#transformer = Normalizer(norm='l2' )
# تستخدم max   لجعل القيمة العظمي في كل صف هي القيمة العظمي
transformer = Normalizer(norm='l2' )

transformer.fit(X)
new_x = transformer.transform(X)
print(new_x)

print('----------------------------------')


from sklearn.preprocessing import MaxAbsScaler
# [4] MaxAbsScaler

X = [[ 1, 10., 2.],
     [ 2, 0., 0.],
     [ 5, 1., -1.]]
transformer = MaxAbsScaler().fit(X)  # default norm='max'
transformer
new_data = transformer.transform(X)

print(new_data)
print('----------------------------------')


import numpy as np
from sklearn.preprocessing import FunctionTransformer

# [5] FunctionTransformer

X = [[4, 1, 2, 2], [1, 3, 9, 3], [5, 7, 5, 1]]

def function1(z):
    return np.sqrt(z)

FT = FunctionTransformer(func = function1)
FT.fit(X)
new_data = FT.transform(X)
print(new_data)
print('----------------------------------')

from sklearn.preprocessing import Binarizer

# [6] Binarizer

X = [[ 1., -1., -2.],[ 2., 0., -1.], [ 0., 1., -1.]]

transformer = Binarizer(threshold=1.5)
transformer.fit(X)

new_data = transformer.transform(X)
print(new_data)
print('----------------------------------')

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# [7] PolynomialFeatures

X = np.arange(6).reshape(3, 2)
# يتم كتابة الدرجة , وهل تحتوي علي قيمة بياس (رقم 1) ام لا

poly = PolynomialFeatures(degree=4, include_bias = True)
poly.fit_transform(X)

# ولو تم اختيار interaction_only  كقيمة True  سيعرض فقط قيم a  مضروبة في b  و يحذف الاسس للقيم الوحيدة

poly = PolynomialFeatures(degree=2, interaction_only=True)
new_data = poly.fit_transform(X)
print(new_data)