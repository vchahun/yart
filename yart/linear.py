import numpy
from  .loss import FloatDataset, SquareLoss

class LinearRegression:
    """
    Linear regression.
    Minimize regularized squared loss:
        L(x, y|w) = 1/2 ||x.w - y||^2 + l2 ||w||^2

    Parameters
    ----------
    l2: float, default=0
        L2 regularization strength
    """
    def __init__(self, l2=0):
        self.l2 = l2
        self.loss = SquareLoss()

    def fit(self, X, y):
        self.coef_ = numpy.zeros(X.shape[1] + 1, dtype=numpy.float64)
        y = numpy.asarray(y, dtype=numpy.float64)
        dataset = FloatDataset(X, y)
        self.loss.fit(dataset, self.coef_, self.l2)
        return self

    def predict(self, X):
        assert X.shape[1] == self.coef_.size - 1
        return X.dot(self.coef_[:-1]) + self.coef_[-1]
