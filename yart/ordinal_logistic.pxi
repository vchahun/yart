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

cdef double ordinal_logistic_loss(Dataset _dataset, Weight w, Weight gradient, float l2):
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

    # Regularization term
    cdef unsigned j
    for j in range(n_features):
        loss += l2 * w.data[j]**2
        gradient.data[j] += 2 * l2 * w.data[j]

    return loss

class OrdinalLogisticLoss(BaseOrdinalLoss):
    """
    Minimize regularized ordinal logistic loss:
        L(x, y|w,t) = - sum_i log (p(k <= y_i|x_i, w,t) - p(k <= y_i - 1|x_i, w,t)) + l2 ||w||^2
        p(k <= y|x, w) = 1/(1 + exp(w.x - t_y)) 1 <= y <= K - 1
    """
    def fit(self, IntegerDataset dataset, numpy.ndarray[DOUBLE, ndim=1] coef, double l2):
        optimize_lbfgs(ordinal_logistic_loss, dataset, coef, l2)
