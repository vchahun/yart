cdef double square_loss(Dataset _dataset, Weight w, Weight gradient):
    """
    w.size = D + 1 = coefficients + intercept
    """
    cdef FloatDataset dataset = <FloatDataset> _dataset
    cdef double loss = 0

    cdef DOUBLE *x_data_ptr
    cdef INTEGER *x_ind_ptr
    cdef unsigned x_nnz
    cdef DOUBLE y

    cdef unsigned i
    cdef double residual
    for i in range(dataset.n_samples):
        dataset.row(i, &x_data_ptr, &x_ind_ptr, &x_nnz, &y)
        residual = w.dot(0, x_data_ptr, x_ind_ptr, x_nnz) + w.data[w.length - 1] - y
        loss += 0.5 * residual**2
        gradient.add(0, residual, x_data_ptr, x_ind_ptr, x_nnz)
        gradient.data[w.length - 1] += residual

    return loss

class LinearRegression:
    def __cinit__(self):
        pass

    def fit(self, X, y):
        self.coef_ = numpy.zeros(X.shape[1] + 1)
        dataset = FloatDataset(X, numpy.asarray(y, dtype=numpy.float64))
        optimize_lbfgs(square_loss, dataset, self.coef_)
        return self

    def predict(self, X):
        assert X.shape[1] == self.coef_.size - 1
        return X.dot(self.coef_[:-1]) + self.coef_[-1]
