# Yet another regression toolkit

YART implements linear, logistic and ordinal regression with L2 regularization.

LBFGS is used to minimize the loss functions.

## Linear regression example
```python
from sklearn import datasets, cross_validation, metrics
from scipy.sparse import csr_matrix
from yart import LinearRegression
diabetes = datasets.load_diabetes()
X = csr_matrix(diabetes.data)
y = diabetes.target
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.5)
model = LinearRegression(l2=0.1)
model.fit(X_train, y_train)
print('MAE: {}'.format(metrics.mean_absolute_error(model.predict(X_test), y_test)))
```

## Logistic regression example
```python
from sklearn import datasets, cross_validation, metrics
from scipy.sparse import csr_matrix
from yart import LogisticRegression
digits = datasets.load_digits()
X = csr_matrix(digits.data)
y = digits.target
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.5)
model = LogisticRegression(l2=0.1)
model.fit(X_train, y_train)
print('Accuracy: {}'.format(metrics.accuracy_score(model.predict(X_test), y_test)))
```

## Ordinal regression example
```python
import numpy
from sklearn import datasets, cross_validation, metrics
from scipy.sparse import csr_matrix
from yart import OrdinalRegression
boston = datasets.load_boston()
X = csr_matrix(boston.data)
y = numpy.round(boston.target/10)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.5)
model = OrdinalRegression(l2=0.1)
model.fit(X_train, y_train)
print('MAE: {}'.format(metrics.mean_absolute_error(model.predict(X_test), y_test)))
```
