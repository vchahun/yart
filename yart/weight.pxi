cdef class Weight:
    cdef double* data
    cdef Py_ssize_t length

    def __cinit__(self, Py_ssize_t length):
        self.data = NULL
        self.length = length

    cdef double dot(self, unsigned offset,
            DOUBLE* x_data_ptr, INTEGER* x_ind_ptr, unsigned x_nnz):
        if x_nnz == 0:
            return 0
        cdef double result = 0
        for j in range(x_nnz):
            result += self.data[x_ind_ptr[j] + offset] * x_data_ptr[j]
        return result

    cdef void add(self, unsigned offset, double scale,
            DOUBLE* x_data_ptr, INTEGER* x_ind_ptr, unsigned x_nnz):
        if x_nnz == 0:
            return
        for j in range(x_nnz):
            self.data[x_ind_ptr[j] + offset] += scale * x_data_ptr[j]

    def __repr__(self):
        return '[{}]'.format(', '.join(str(self.data[i]) for i in range(self.length)))
