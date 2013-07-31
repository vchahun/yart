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

class SquareLoss:
    """
    Minimize regularized squared loss:
        L(x, y|w) = 1/2 ||x.w - y||^2 + l2 ||w||^2
    """
    def fit(self, FloatDataset dataset, numpy.ndarray[DOUBLE, ndim=1] coef, double l2):
        optimize_lbfgs(square_loss, dataset, coef, l2)
        return self
