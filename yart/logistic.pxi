cdef extern from "math.h":
    cdef enum:
        INFINITY

cdef double logaddexp(double a, double b):
    """
    compute log(exp(a) + exp(b))
    """
    if a == -INFINITY: return b
    if b == -INFINITY: return a
    if b < a: # b - a < 0
        return a + log1p(exp(b - a))
    else: # a - b < 0
        return b + log1p(exp(a - b))

cdef double log_loss(Dataset _dataset, Weight w, Weight gradient):
    """
    w.size = (K - 1) * (D + 1)  = (classes - 1)*(coefficients + intercept)
    y = 0 ... K - 1
    """
    cdef IntegerDataset dataset = <IntegerDataset> _dataset
    cdef unsigned n_classes = w.length / (dataset.n_features + 1) + 1
    cdef unsigned n_features = dataset.n_features
    cdef double loss = 0

    cdef DOUBLE *x_data_ptr
    cdef INTEGER *x_ind_ptr
    cdef unsigned x_nnz
    cdef INTEGER y

    cdef unsigned i, k, offset
    cdef double wx, z, delta
    cdef double* log_probs = <double*> malloc((n_classes - 1) * sizeof(double))
    for i in range(dataset.n_samples):
        dataset.row(i, &x_data_ptr, &x_ind_ptr, &x_nnz, &y)
        z = 0
        for k in range(n_classes - 1):
            offset = (n_features + 1) * k
            wx = w.dot(offset, x_data_ptr, x_ind_ptr, x_nnz) + w.data[offset + n_features - 1]
            log_probs[k] = wx
            z = logaddexp(z, wx)
        loss += -(log_probs[y] - z)
        for k in range(n_classes - 1):
            offset = (n_features + 1) * k
            delta = exp(log_probs[k] - z) - (y == k)
            gradient.add(offset, delta, x_data_ptr, x_ind_ptr, x_nnz) 
            gradient.data[offset + n_features - 1] += delta
    free(log_probs)

    return loss

class LogisticRegression:
    def __cinit__(self):
        pass

    def fit(self, X, y):
        y = numpy.asarray(y, dtype=numpy.int32)
        self.n_classes = len(numpy.unique(y))
        self.coef_ = numpy.zeros((X.shape[1] + 1) * (self.n_classes - 1))
        dataset = IntegerDataset(X, y)
        optimize_lbfgs(log_loss, dataset, self.coef_)
        return self

    def predict(self, X):
        cdef unsigned n_features = self.coef_.size/(self.n_classes - 1) - 1
        assert X.shape[1] == n_features
        y_pred = numpy.zeros(X.shape[0], int)
        log_probs = numpy.zeros(self.n_classes)
        cdef unsigned i, k, offset
        cdef double wx
        for i in range(X.shape[0]):
            log_probs[self.n_classes - 1] = 0
            for k in range(self.n_classes - 1):
                offset = (n_features + 1) * k
                wx = X[i].dot(self.coef_[offset:offset+n_features]) + self.coef_[offset + n_features - 1]
                log_probs[k] = wx
            y_pred[i] = log_probs.argmax()
        return y_pred
