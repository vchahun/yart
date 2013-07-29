cdef extern from "math.h":
    int isnan(double x)

cdef double square_loss(Dataset _dataset, Weight w, Weight gradient, float l2):
    """
    w.size = D + 1 = coefficients + intercept
    """
    cdef FloatDataset dataset = <FloatDataset> _dataset
    cdef double loss = 0

    cdef DOUBLE *x_data_ptr
    cdef INTEGER *x_ind_ptr
    cdef INTEGER x_nnz
    cdef DOUBLE y

    cdef unsigned i
    cdef double residual
    for i in range(dataset.n_samples):
        dataset.row(i, &x_data_ptr, &x_ind_ptr, &x_nnz, &y)
        residual = w.dot(0, x_data_ptr, x_ind_ptr, x_nnz) + w.data[w.length - 1] - y
        loss += 0.5 * residual**2
        gradient.add(0, residual, x_data_ptr, x_ind_ptr, x_nnz)
        gradient.data[w.length - 1] += residual

    # Regularization term
    cdef unsigned j
    for j in range(dataset.n_features):
        loss += l2 * w.data[j]**2
        gradient.data[j] += 2 * l2 * w.data[j]

    return loss

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


    def fit(self, X, y):
        self.coef_ = numpy.zeros(X.shape[1] + 1, dtype=numpy.float64)
        y = numpy.asarray(y, dtype=numpy.float64)
        dataset = FloatDataset(X, y)
        optimize_lbfgs(square_loss, dataset, self.coef_, self.l2)
        return self

    def predict(self, X):
        assert X.shape[1] == self.coef_.size - 1
        return X.dot(self.coef_[:-1]) + self.coef_[-1]
