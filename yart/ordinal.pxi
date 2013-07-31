from libc.math cimport exp, log1p, expm1

cdef double phi(double t):
    """
    Logistic function
    phi(t) = 1 / (1 + exp(-t))
    """
    if t > 10: return 1
    if t < -10: return 0
    return 1 / (1 + exp(-t))

cdef double log_phi(double t):
    """
    Log of the logistic function
    log phi(t) = log(1 + exp(-t))
    """
    if t > 5: return 0
    if t < -5 : return t
    return - log1p(exp(-t))

cdef int unordered_thresholds(double* thresholds, int n_thresholds):
    """
    Return False unless thresholds[i] < thresholds[i+1] for all i
    """
    cdef unsigned i
    for i in range(n_thresholds - 1):
        if thresholds[i+1] <= thresholds[i]:
            return True
    return False

class BaseOrdinalLoss:
    def predict(self, int n_features, int n_thresholds, numpy.ndarray[DOUBLE, ndim=1] coef, X):
        weights, thresholds = coef[:n_features], coef[n_features:]
        y_pred = numpy.zeros(X.shape[0], int) + n_thresholds
        cdef unsigned i, k
        cdef double wx, threshold
        for i in range(X.shape[0]):
            wx = X[i].dot(weights)
            for k, threshold in enumerate(thresholds):
                if wx < threshold:
                    y_pred[i] = k
                    break
        return y_pred

include "ordinal_logistic.pxi"
