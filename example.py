import numpy
from sklearn import feature_extraction, metrics, datasets
import yart

COEFS = [5, 1, 0] # x, x^2, 1
def make_real_data(N):
    b, c, a = COEFS
    for i in xrange(N):
        x = float(i)/N - 0.5 + numpy.random.randn()/10
        yield {'x': x, 'x^2': x**2}, a + b*x + c*x**2 + numpy.random.randn()/10

"""
# Uncomment to see progress
import logging
logging.basicConfig(level=logging.INFO)
"""

# Continuous responses
print('Continuous - Linear regression')
data, target = zip(*make_real_data(1000))
X = feature_extraction.DictVectorizer().fit_transform(data)
y = numpy.array(target)
model = yart.LinearRegression().fit(X, y)
print 'MAE:', metrics.mean_absolute_error(model.predict(X), y)
numpy.testing.assert_almost_equal(model.coef_, COEFS, 1)

# Discrete responses
print('Discrete - Logistic regression')
import scipy.sparse
digits = datasets.load_digits()
X = scipy.sparse.csr_matrix(digits.data)
y = digits.target
model = yart.LogisticRegression().fit(X, y)
print 'Accuracy:', metrics.accuracy_score(model.predict(X), y)

# Ordinal responses
data, target = zip(*make_real_data(1000))
X = feature_extraction.DictVectorizer().fit_transform(data)
y = numpy.array(map(int, target))
print('Ordinal - Ordinal regression')
model = yart.OrdinalLogisticRegression().fit(X, y)
print 'MAE:', metrics.mean_absolute_error(model.predict(X), y)
print('Ordinal - Linear regression')
model = yart.LinearRegression().fit(X, y)
print 'MAE:', metrics.mean_absolute_error(model.predict(X), y)
