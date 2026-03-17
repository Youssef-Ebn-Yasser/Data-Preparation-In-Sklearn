# Scikit-Learn (sklearn) Features Examples

This repository demonstrates some core features of **Scikit-Learn (`sklearn`)** in Python. It covers:

1. **Loading popular datasets**
2. **Cleaning data using `SimpleImputer`**
3. **Generating sample data for regression and classification**
4. **Evaluating models using various metrics**

It is meant as a practical guide for beginners to explore **datasets, data preprocessing, and model evaluation**.

---

## 1. Cleaning Data with `SimpleImputer`

`SimpleImputer` is used to handle missing values in datasets. It can replace missing values (`np.nan` or zeros) with a **mean**, **median**, or **constant value**.

**Example: Breast Cancer Data**

```python
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
import numpy as np

# Load dataset
BreastData = load_breast_cancer()
X = BreastData.data
y = BreastData.target

# Impute missing values with mean
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imputer.fit_transform(X)

print(X[:10])
print(y[:10])
```

Other examples demonstrate replacing missing values with `median` or handling zeros as missing values.

---

## 2. Loading Common Datasets

Scikit-Learn provides many built-in datasets for learning and testing:

1. **Iris (Flowers) Dataset** – classification of flower species
2. **Digits Dataset** – images of handwritten digits (8x8 pixels)
3. **Wine Dataset** – chemical analysis of wines for classification
4. **Breast Cancer Dataset** – features for cancer classification
5. **Diabetes Dataset** – features for regression tasks
6. **Regression Data** – generated using `make_regression()`
7. **Classification Data** – generated using `make_classification()`
8. **Sample Images** – `load_sample_image()` like "china.jpg" and "flower.jpg"

Each dataset includes **feature matrix `X`**, **target values `y`**, and metadata like **feature names** and **target names**.

---

## 3. Generating Sample Data

* **Regression**:

```python
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=10000, n_features=500, shuffle=True)
```

* **Classification**:

```python
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=100, n_features=20, shuffle=True)
```

These functions help create datasets for testing algorithms without needing real data.

---

## 4. Model Evaluation Metrics

Scikit-Learn provides many metrics to evaluate models. Examples include:

* **Regression Metrics**:

  * `mean_absolute_error`
  * `mean_squared_error`
  * `median_absolute_error`

* **Classification Metrics**:

  * `confusion_matrix`
  * `accuracy_score`
  * `f1_score`
  * `recall_score`
  * `precision_score`
  * `precision_recall_fscore_support`
  * `precision_recall_curve`
  * `classification_report`
  * `roc_curve`, `roc_auc_score`
  * `auc`
  * `zero_one_loss`

**Example: Mean Absolute Error**

```python
from sklearn.metrics import mean_absolute_error

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

mae = mean_absolute_error(y_true, y_pred)
print(mae)
```

**Example: Confusion Matrix**

```python
from sklearn.metrics import confusion_matrix

y_true = ['a','b','b','a','b','a','a','b','a','b']
y_pred = ['a','a','b','b','a','b','a','a','a','a']

cm = confusion_matrix(y_true, y_pred)
print(cm)
```

**Example: ROC Curve and AUC**

```python
from sklearn import metrics
import numpy as np

y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])

fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
auc_value = metrics.auc(fpr, tpr)

print(f"FPR: {fpr}, TPR: {tpr}, Thresholds: {thresholds}, AUC: {auc_value}")
```

---

## 5. Summary

This repository helps you:

* Understand **how to clean and preprocess data**
* Explore **built-in datasets** for machine learning tasks
* Generate **synthetic regression and classification datasets**
* Evaluate models using **key metrics for regression and classification**

It’s a practical starting point for anyone learning **Scikit-Learn** and preparing data for machine learning models.

Do you want me to do that?
