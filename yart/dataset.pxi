import scipy.sparse

cdef class Dataset:
    cdef Py_ssize_t n_samples
    cdef Py_ssize_t n_features
    cdef DOUBLE *X_data_ptr
    cdef INTEGER *X_indptr_ptr
    cdef INTEGER *X_indices_ptr

cdef class FloatDataset(Dataset):
    cdef DOUBLE *Y_data_ptr

    def __cinit__(self, X, numpy.ndarray[DOUBLE, ndim=1] Y):
        assert scipy.sparse.isspmatrix_csr(X)
        cdef numpy.ndarray[DOUBLE, ndim=1] X_data = X.data
        cdef numpy.ndarray[INTEGER, ndim=1] X_indptr = X.indptr
        cdef numpy.ndarray[INTEGER, ndim=1] X_indices = X.indices
        self.n_samples, self.n_features = X.shape
        self.X_data_ptr = <DOUBLE *>X_data.data
        self.X_indptr_ptr = <INTEGER *>X_indptr.data
        self.X_indices_ptr = <INTEGER *>X_indices.data
        self.Y_data_ptr = <DOUBLE *>Y.data

    cdef void row(self, unsigned index,
            DOUBLE **x_data_ptr, INTEGER **x_ind_ptr,
            unsigned *x_nnz, DOUBLE *y):
        cdef int offset = self.X_indptr_ptr[index]
        y[0] = self.Y_data_ptr[index]
        x_data_ptr[0] = self.X_data_ptr + offset
        x_ind_ptr[0] = self.X_indices_ptr + offset
        x_nnz[0] = self.X_indptr_ptr[index + 1] - offset

cdef class IntegerDataset(Dataset):
    cdef INTEGER *Y_data_ptr

    def __cinit__(self, X, numpy.ndarray[INTEGER, ndim=1] Y):
        assert scipy.sparse.isspmatrix_csr(X)
        cdef numpy.ndarray[DOUBLE, ndim=1] X_data = X.data
        cdef numpy.ndarray[INTEGER, ndim=1] X_indptr = X.indptr
        cdef numpy.ndarray[INTEGER, ndim=1] X_indices = X.indices
        self.n_samples, self.n_features = X.shape
        self.X_data_ptr = <DOUBLE *>X_data.data
        self.X_indptr_ptr = <INTEGER *>X_indptr.data
        self.X_indices_ptr = <INTEGER *>X_indices.data
        self.Y_data_ptr = <INTEGER *>Y.data

    cdef void row(self, unsigned index,
            DOUBLE **x_data_ptr, INTEGER **x_ind_ptr,
            unsigned *x_nnz, INTEGER *y):
        cdef int offset = self.X_indptr_ptr[index]
        y[0] = self.Y_data_ptr[index]
        x_data_ptr[0] = self.X_data_ptr + offset
        x_ind_ptr[0] = self.X_indices_ptr + offset
        x_nnz[0] = self.X_indptr_ptr[index + 1] - offset
