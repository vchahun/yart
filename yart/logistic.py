import numpy
from  .loss import IntegerDataset, LogisticLoss

class LogisticRegression:
    """
    Logistic regression.
    Minimize regularized log-loss:
        L(x, y|w) = - sum_i log p(y_i|x_i, w) + l2 ||w||^2
        p(y|x, w) = exp(w[y].x) / (sum_y' exp(w[y'].x))

    Parameters
    ----------
    l2: float, default=0
        L2 regularization strength
    """
    def __init__(self, l2=0):
        self.l2 = l2
        self.loss = LogisticLoss()

    def fit(self, X, y):
        y = numpy.asarray(y, dtype=numpy.int32)
        self.n_classes = len(numpy.unique(y))
        self.coef_ = numpy.zeros((X.shape[1] + 1) * (self.n_classes - 1), dtype=numpy.float64)
        dataset = IntegerDataset(X, y)
        self.loss.fit(dataset, self.coef_, self.l2)
        return self

    def predict(self, X):
        n_features = self.coef_.size/(self.n_classes - 1) - 1
        assert X.shape[1] == n_features
        return self.loss.predict(n_features, self.n_classes, self.coef_, X)

    def predict_proba(self, X):
        n_features = self.coef_.size/(self.n_classes - 1) - 1
        assert X.shape[1] == n_features
        return self.loss.predict_proba(n_features, self.n_classes, self.coef_, X)
