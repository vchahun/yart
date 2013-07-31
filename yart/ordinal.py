import numpy
from .loss import IntegerDataset, OrdinalLogisticLoss

class OrdinalRegression:
    """
    Ordinal logistic regression.
    Minimize regularized log-loss:
        L(x, y|w,t) = - sum_i log p(y_i|x_i, w, t) + l2 ||w||^2

    Parameters
    ----------
    l2: float, default=0
        L2 regularization strength
    """
    def __init__(self, l2=0, loss='logistic'):
        self.l2 = l2
        if loss == 'logistic':
            self.loss = OrdinalLogisticLoss()
        else:
            raise NotImplementedError

    def fit(self, X, y):
        y = numpy.asarray(y)
        # map y to range(K) where K is the number of levels
        self.original_levels_ = numpy.unique(y)
        y_reset = numpy.zeros(y.size, dtype=numpy.int32)
        for i, u in enumerate(self.original_levels_):
            y_reset[y == u] = i
        y = y_reset
        n_thresholds = len(self.original_levels_)
        # initialize weight vector
        self.coef_ = numpy.zeros(X.shape[1] + n_thresholds - 1, dtype=numpy.float64)
        self.coef_[X.shape[1]:] = range(n_thresholds - 1)
        dataset = IntegerDataset(X, y)
        self.loss.fit(dataset, self.coef_, self.l2)
        return self

    def predict(self, X):
        n_thresholds = len(self.original_levels_) - 1
        n_features = self.coef_.size - n_thresholds
        assert X.shape[1] == n_features
        y_pred = self.loss.predict(n_features, n_thresholds, self.coef_, X)
        return self.original_levels_[y_pred]

    def predict_proba(self, X):
        n_thresholds = len(self.original_levels_) - 1
        n_features = self.coef_.size - n_thresholds
        assert X.shape[1] == n_features
        return self.loss.predict_proba(n_features, n_thresholds, self.coef_, X)
