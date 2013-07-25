import numpy
cimport numpy
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset

ctypedef numpy.float64_t DOUBLE
ctypedef numpy.int32_t INTEGER

include "dataset.pxi"
include "weight.pxi"

ctypedef double (*loss_callback_t)(Dataset, Weight, Weight)
ctypedef double (*gradient_descent_eval_t)(void*, double*, double*, unsigned n) # data, x, g, |x|
ctypedef void (*gradient_descent_progress_t)(void*, double*, unsigned n) # data, x, |x|

cdef class CallbackData:
    cdef Dataset dataset
    cdef loss_callback_t loss_callback
    cdef Weight final_weight

    def __cinit__(self, Dataset dataset, Weight final_weight):
        self.dataset = dataset
        self.final_weight = final_weight

# Call loss function to produce gradient g and return loss value at x
cdef double gradient_descent_eval(void* data, double* x, double* g, unsigned n):
    cdef CallbackData cb_data = <CallbackData> data
    cdef Weight point = Weight(n)
    point.data = x
    cdef Weight gradient = Weight(n)
    gradient.data = g
    return cb_data.loss_callback(cb_data.dataset, point, gradient)

# Store a copy of the final weights
cdef void gradient_descent_progress(void* data, double* x, unsigned n):
    cdef CallbackData cb_data = <CallbackData> data
    memcpy(cb_data.final_weight.data, x, n * sizeof(double))

cdef double gradient_descent_pro

cdef gradient_descent(loss_callback_t loss_callback, X, y, numpy.ndarray[DOUBLE, ndim=1] w):
    cdef Dataset dataset = Dataset(X.data, X.indptr, X.indices, y)
    cdef Weight final_weight = Weight(w.size)
    final_weight.data = <double*>w.data
    cdef CallbackData cb_data = CallbackData(dataset, final_weight)
    cb_data.loss_callback = loss_callback
    _gradient_descent(w.size, <double*>w.data,
            gradient_descent_eval,
            gradient_descent_progress,
            <void*>cb_data)

cdef void _gradient_descent(unsigned n, double* x0,
        gradient_descent_eval_t call_eval,
        gradient_descent_progress_t call_progress, void* data):
    cdef double* x = <double*> malloc(n * sizeof(double))
    memcpy(x, x0, n * sizeof(double))
    cdef double* g = <double*> malloc(n * sizeof(double))
    cdef double total_diff
    cdef unsigned j
    cdef unsigned iteration
    for iteration in range(10000):
        memset(g, 0, n * sizeof(double))
        loss = call_eval(data, x, g, n)
        print('Iteration {}: loss={}'.format(iteration, loss))
        total_diff = 0
        for j in range(n):
            total_diff += g[j]**2
            x[j] -= 0.001 * g[j]
        if total_diff < 1e-3:
            break
    call_progress(data, x, n)
    free(x)

include "linear.pxi"
