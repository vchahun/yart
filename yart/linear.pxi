cdef double square_loss(Dataset dataset, Weight w, Weight gradient):
    """
    w.size = D + 1 = coefficients + intercept
    """
    cdef double loss = 0

    cdef unsigned i
    cdef DOUBLE *x_data_ptr
    cdef INTEGER *x_ind_ptr
    cdef unsigned x_nnz
    cdef DOUBLE y

    for i in range(dataset.n_samples):
        dataset.row(i, &x_data_ptr, &x_ind_ptr, &x_nnz, &y)
        residual = w.dot(0, x_data_ptr, x_ind_ptr, x_nnz) + w.data[w.length-1] - y
        loss += 0.5 * residual**2
        gradient.add(0, residual, x_data_ptr, x_ind_ptr, x_nnz)
        gradient.data[w.length-1] += residual

    return loss

class LinearRegression:
    def __cinit__(self):
        pass

    def fit(self, X, y):
        self.coef_ = numpy.zeros(X.shape[1] + 1)
        optimize_lbfgs(square_loss, X, y, self.coef_)

    def predict(self, X):
         return X.dot(self.coef_[:-1]) + self.coef_[-1]
