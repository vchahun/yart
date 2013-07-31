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

cdef double log_loss(Dataset _dataset, Weight w, Weight gradient, float l2):
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
    cdef INTEGER x_nnz
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

    # Regularization term - does not include intercept
    cdef unsigned j
    for k in range(n_classes - 1):
        offset = (n_features + 1) * k
        for j in range(dataset.n_features):
            loss += l2 * w.data[offset + j]**2
            gradient.data[offset + j] += 2 * l2 * w.data[offset + j]

    return loss

class LogisticLoss:
    """
    Minimize regularized log-loss:
        L(x, y|w) = - sum_i log p(y_i|x_i, w) + l2 ||w||^2
        p(y|x, w) = exp(w[y].x) / (sum_y' exp(w[y'].x))
    """
    def fit(self, IntegerDataset dataset, numpy.ndarray[DOUBLE, ndim=1] coef, double l2):
        optimize_lbfgs(log_loss, dataset, coef, l2)

    def predict(self, int n_features, int n_classes, numpy.ndarray[DOUBLE, ndim=1] coef, X):
        y_pred = numpy.zeros(X.shape[0], int)
        log_probs = numpy.zeros(n_classes)
        cdef unsigned i, k, offset
        cdef double wx
        for i in range(X.shape[0]):
            log_probs[n_classes - 1] = 0
            for k in range(n_classes - 1):
                offset = (n_features + 1) * k
                log_probs[k] = X[i].dot(coef[offset:offset+n_features]) + coef[offset + n_features - 1]
            y_pred[i] = log_probs.argmax()
        return y_pred

    def predict_proba(self, int n_features, int n_classes, numpy.ndarray[DOUBLE, ndim=1] coef, X):
        y_proba = numpy.zeros((X.shape[0], n_classes))
        cdef unsigned i, k, offset
        cdef double wx, z
        for i in range(X.shape[0]):
            z = 0
            y_proba[i, n_classes - 1] = 0
            for k in range(n_classes - 1):
                offset = (n_features + 1) * k
                wx = X[i].dot(coef[offset:offset+n_features]) + coef[offset + n_features - 1]
                y_proba[i, k] = wx
                z = logaddexp(z, wx)
            y_proba[i] = numpy.exp(y_proba[i] - z)
        return y_proba
