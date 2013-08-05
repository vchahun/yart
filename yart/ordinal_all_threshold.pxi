cdef double all_threshold_logistic_loss(Dataset _dataset, Weight w, Weight gradient, float l2):
    """
    w.size = D + K - 1 = coefficients + levels - 1
    y = 0 ... K - 1
    """
    cdef IntegerDataset dataset = <IntegerDataset> _dataset
    cdef Py_ssize_t n_features = dataset.n_features
    cdef Py_ssize_t n_thresholds = w.length - dataset.n_features
    cdef DOUBLE* thresholds = w.data + n_features

    cdef double loss = 0

    cdef DOUBLE *x_data_ptr
    cdef INTEGER *x_ind_ptr
    cdef INTEGER x_nnz
    cdef INTEGER y

    cdef unsigned i, k
    cdef double wx, residual, s
    for i in range(dataset.n_samples):
        dataset.row(i, &x_data_ptr, &x_ind_ptr, &x_nnz, &y)
        wx = w.dot(0, x_data_ptr, x_ind_ptr, x_nnz)
        residual = 0
        for k in range(n_thresholds):
            s = (1 if k < y else -1)
            loss += - log_phi(s * (wx - thresholds[k]))
            residual += - s * phi(s * (thresholds[k] - wx))
            gradient.data[n_features + k] += s * phi(s * (thresholds[k] - wx))
        gradient.add(0, residual, x_data_ptr, x_ind_ptr, x_nnz)

    # Regularization term
    cdef unsigned j
    for j in range(n_features):
        loss += l2 * w.data[j]**2
        gradient.data[j] += 2 * l2 * w.data[j]

    return loss

class OrdinalAllThresholdLoss(BaseOrdinalLoss):
    """
    Minimize regularized all-threshold logistic loss:
        L(x, y|w,t) = - sum_i sum_k log phi(sign(y_i - k) * (w.x_i - t_k)) + l2 ||w||^2
        phi(t) = 1/(1 + exp(-t))

    References
    ----------
    Ordinal Logistic Regression, Rennie 2005
    http://qwone.com/~jason/writing/olr.pdf
    """
    def fit(self, IntegerDataset dataset, numpy.ndarray[DOUBLE, ndim=1] coef, double l2):
        optimize_lbfgs(all_threshold_logistic_loss, dataset, coef, l2)
