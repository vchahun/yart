from libc.math cimport exp, log1p, expm1

cdef double log_phi(double t):
    if t > 5: return 0
    if t < -5 : return t
    return - log1p(exp(-t))

cdef double phi(double t):
    if t > 10: return 1
    if t < -10: return 0
    return 1 / (1 + exp(-t))

cdef double ordinal_log_prob(double wx, int k, double* thresholds, int n_thresholds):
    """
    log p(y = k | x)
    """
    if k == 0:
        # p(y = 1) = phi(t_1 - w.x)
        return log_phi(thresholds[0] - wx)
    if k == n_thresholds:
        # p(y = K) = 1 - phi(t_{K-1} - w.x)
        return log_phi(wx - thresholds[n_thresholds - 1])
    # u = log[exp(-t_{k-1}) - exp(-t_k)]
    cdef double u = - thresholds[k - 1] + log1p(-exp(thresholds[k - 1] - thresholds[k]))
    cdef double v = log_phi(thresholds[k] - wx)
    cdef double w = log_phi(thresholds[k - 1] - wx)
    return wx + u + v + w

cdef double cum_prob(double wx, int k, double* thresholds, int n_thresholds):
    """
    p(y <= k | x)
    """
    if k == -1:
        return 0 # p(y <= 0)
    if k == n_thresholds:
        return 1 # p(y <= K)
    return phi(thresholds[k] - wx)

cdef int unordered_thresholds(double* thresholds, int n_thresholds):
    cdef unsigned i
    for i in range(n_thresholds - 1):
        if thresholds[i+1] <= thresholds[i]:
            return True
    return False

cdef double ordinal_logistic_loss(Dataset _dataset, Weight w, Weight gradient):
    """
    w.size = D + K - 1 = coefficients + levels - 1
    y = 0 ... K - 1
    """
    cdef IntegerDataset dataset = <IntegerDataset> _dataset
    cdef Py_ssize_t n_features = dataset.n_features
    cdef Py_ssize_t n_thresholds = w.length - dataset.n_features
    cdef DOUBLE* thresholds = w.data + n_features

    if unordered_thresholds(thresholds, n_thresholds):
        return 1e10

    cdef double loss = 0

    cdef DOUBLE *x_data_ptr
    cdef INTEGER *x_ind_ptr
    cdef INTEGER x_nnz
    cdef INTEGER y

    cdef unsigned i
    cdef double wx, residual, u, v
    for i in range(dataset.n_samples):
        dataset.row(i, &x_data_ptr, &x_ind_ptr, &x_nnz, &y)
        wx = w.dot(0, x_data_ptr, x_ind_ptr, x_nnz)
        loss += - ordinal_log_prob(wx, y, thresholds, n_thresholds)
        residual = 1 - cum_prob(wx, y, thresholds, n_thresholds) - cum_prob(wx, y - 1, thresholds, n_thresholds)
        gradient.add(0, residual, x_data_ptr, x_ind_ptr, x_nnz)
        if y > 0:
            if y == n_thresholds:
                u = 1
            else:
                u = -1 / expm1(thresholds[y - 1] - thresholds[y]) # 1/(1-exp(t_j - t_{j+1}))
            v = cum_prob(wx, y - 1, thresholds, n_thresholds)
            gradient.data[n_features + y - 1] += -1 + u + v
        if y < n_thresholds:
            if y == 0:
                u = 0
            else:
                u = -1 / expm1(thresholds[y] - thresholds[y - 1]) # 1/(1-exp(t_j - t_{j-1}))
            v = cum_prob(wx, y, thresholds, n_thresholds)
            gradient.data[n_features + y] += -1 + u + v
    return loss

class OrdinalLogisticRegression:
    def __cinit__(self):
        pass

    def fit(self, X, y):
        y = numpy.asarray(y)
        # map y to range(K) where K is the number of levels
        self.original_levels = numpy.unique(y)
        cdef numpy.ndarray[INTEGER, ndim=1] y_reset = numpy.zeros(y.size, dtype=numpy.int32)
        cdef INTEGER i
        cdef int u
        for i, u in enumerate(self.original_levels):
            y_reset[y == u] = i
        y = y_reset
        K = len(self.original_levels)
        # initialize weight vector
        self.coef_ = numpy.zeros(X.shape[1] + K - 1, dtype=numpy.float64)
        self.coef_[X.shape[1]:] = range(K - 1)
        dataset = IntegerDataset(X, y)
        optimize_lbfgs(ordinal_logistic_loss, dataset, self.coef_)
        return self

    def predict(self, X):
        K = len(self.original_levels)
        n_features = self.coef_.size - K + 1
        assert X.shape[1] == n_features
        y_pred = numpy.zeros(X.shape[0], int) + K - 1
        coef, thresholds = self.coef_[:n_features], self.coef_[n_features:]
        cdef unsigned i, k
        cdef double threshold
        for i in xrange(X.shape[0]):
            wx = X[i].dot(coef)
            for k, threshold in enumerate(thresholds):
                if wx < threshold:
                    y_pred[i] = k
                    break
        return self.original_levels[y_pred]
