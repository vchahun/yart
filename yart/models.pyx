import logging
import numpy
cimport numpy
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset

ctypedef numpy.float64_t DOUBLE
ctypedef numpy.int32_t INTEGER

include "dataset.pxi"
include "weight.pxi"

ctypedef double (*loss_callback_t)(Dataset, Weight, Weight)

cdef class CallbackData:
    cdef Dataset dataset
    cdef loss_callback_t loss_callback

    def __cinit__(self, Dataset dataset):
        self.dataset = dataset

#include "gradient_descent.pxi"
include "lbfgs.pxi"
include "linear.pxi"
include "logistic.pxi"
include "ordinal.pxi"
