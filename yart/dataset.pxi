cdef class Dataset:
    cdef Py_ssize_t n_samples
    cdef DOUBLE *X_data_ptr
    cdef INTEGER *X_indptr_ptr
    cdef INTEGER *X_indices_ptr
    cdef DOUBLE *Y_data_ptr

    def __cinit__(self, numpy.ndarray[DOUBLE, ndim=1] X_data,
            numpy.ndarray[INTEGER, ndim=1] X_indptr,
            numpy.ndarray[INTEGER, ndim=1] X_indices,
            numpy.ndarray[DOUBLE, ndim=1] Y):
        self.n_samples = Y.shape[0]
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
