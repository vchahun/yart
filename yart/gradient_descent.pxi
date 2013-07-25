# data, x, g, |x|, step
ctypedef double (*gradient_descent_eval_t)(void*, double*, double*, unsigned, unsigned)
# data, x, g, f(x), ||x||, ||g||, step, |x|, n_iter, n_eval
ctypedef void (*gradient_descent_progress_t)(void*, double*, double*, double, double, double, unsigned, unsigned, unsigned, unsigned)

# Call loss function to produce gradient g and return loss value at x
cdef double gradient_descent_eval(void* data, double* x, double* g, unsigned n, unsigned step):
    cdef CallbackData cb_data = <CallbackData> data
    cdef Weight point = Weight(n)
    point.data = x
    cdef Weight gradient = Weight(n)
    gradient.data = g
    return cb_data.loss_callback(cb_data.dataset, point, gradient)

# Progress function
cdef void gradient_descent_progress(void* data, double* x, double* g, double fx,
        double xnorm, double gnorm, unsigned step, unsigned n, unsigned k, unsigned ls):
    logging.info('Iteration {}: loss={} ||x||={} ||g||={}'.format(k, fx, xnorm, gnorm))

cdef gradient_descent(loss_callback_t loss_callback, X, y, numpy.ndarray[DOUBLE, ndim=1] w):
    cdef Dataset dataset = Dataset(X.data, X.indptr, X.indices, y)
    cdef CallbackData cb_data = CallbackData(dataset)
    cb_data.loss_callback = loss_callback
    # w -> x
    cdef double* x = <double*> malloc(w.size * sizeof(double))
    memcpy(x, w.data, w.size * sizeof(double))
    # call gradient descent
    cdef double fx
    _gradient_descent(w.size, x, &fx,
            gradient_descent_eval,
            gradient_descent_progress,
            <void*>cb_data)
    # x -> w
    memcpy(w.data, x, w.size * sizeof(double))
    free(x)

cdef void _gradient_descent(unsigned n, double* x, double* fx,
        gradient_descent_eval_t call_eval,
        gradient_descent_progress_t call_progress, void* data):
    cdef double* g = <double*> malloc(n * sizeof(double))
    cdef double xnorm
    cdef double gnorm
    cdef unsigned j
    cdef unsigned iteration
    for iteration in range(10000):
        memset(g, 0, n * sizeof(double))
        fx[0] = call_eval(data, x, g, n, 0)
        xnorm = gnorm = 0
        for j in range(n):
            xnorm += x[j]**2
            gnorm += g[j]**2
            x[j] -= 0.001 * g[j]
        call_progress(data, x, g, fx[0], xnorm, gnorm, 0, n, iteration, 0)
        if gnorm < 1e-3:
            break
    free(g)
