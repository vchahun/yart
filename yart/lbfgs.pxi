cdef extern from "lbfgs.h":
    ctypedef double lbfgsfloatval_t
    ctypedef lbfgsfloatval_t* lbfgsconst_p "const lbfgsfloatval_t *"

    ctypedef lbfgsfloatval_t (*lbfgs_evaluate_t)(void *, lbfgsconst_p,
                              lbfgsfloatval_t *, int, lbfgsfloatval_t)
    ctypedef int (*lbfgs_progress_t)(void *, lbfgsconst_p, lbfgsconst_p,
                                     lbfgsfloatval_t, lbfgsfloatval_t,
                                     lbfgsfloatval_t, lbfgsfloatval_t,
                                     int, int, int)

    cdef enum LineSearchAlgo:
        LBFGS_LINESEARCH_DEFAULT,
        LBFGS_LINESEARCH_MORETHUENTE,
        LBFGS_LINESEARCH_BACKTRACKING_ARMIJO,
        LBFGS_LINESEARCH_BACKTRACKING_WOLFE,
        LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE

    cdef enum ReturnCode:
        LBFGS_SUCCESS,
        LBFGS_ALREADY_MINIMIZED,
        LBFGSERR_UNKNOWNERROR,
        LBFGSERR_LOGICERROR,
        LBFGSERR_OUTOFMEMORY,
        LBFGSERR_CANCELED,
        LBFGSERR_INVALID_N,
        LBFGSERR_INVALID_N_SSE,
        LBFGSERR_INVALID_X_SSE,
        LBFGSERR_INVALID_EPSILON,
        LBFGSERR_INVALID_TESTPERIOD,
        LBFGSERR_INVALID_DELTA,
        LBFGSERR_INVALID_LINESEARCH,
        LBFGSERR_INVALID_MINSTEP,
        LBFGSERR_INVALID_MAXSTEP,
        LBFGSERR_INVALID_FTOL,
        LBFGSERR_INVALID_WOLFE,
        LBFGSERR_INVALID_GTOL,
        LBFGSERR_INVALID_XTOL,
        LBFGSERR_INVALID_MAXLINESEARCH,
        LBFGSERR_INVALID_ORTHANTWISE,
        LBFGSERR_INVALID_ORTHANTWISE_START,
        LBFGSERR_INVALID_ORTHANTWISE_END,
        LBFGSERR_OUTOFINTERVAL,
        LBFGSERR_INCORRECT_TMINMAX,
        LBFGSERR_ROUNDING_ERROR,
        LBFGSERR_MINIMUMSTEP,
        LBFGSERR_MAXIMUMSTEP,
        LBFGSERR_MAXIMUMLINESEARCH,
        LBFGSERR_MAXIMUMITERATION,
        LBFGSERR_WIDTHTOOSMALL,
        LBFGSERR_INVALIDPARAMETERS,
        LBFGSERR_INCREASEGRADIENT

    ctypedef struct lbfgs_parameter_t:
        int m
        lbfgsfloatval_t epsilon
        int past
        lbfgsfloatval_t delta
        int max_iterations
        int linesearch
        int max_linesearch
        lbfgsfloatval_t min_step
        lbfgsfloatval_t max_step
        lbfgsfloatval_t ftol
        lbfgsfloatval_t wolfe
        lbfgsfloatval_t gtol
        lbfgsfloatval_t xtol
        lbfgsfloatval_t orthantwise_c
        int orthantwise_start
        int orthantwise_end

    int lbfgs(int, lbfgsfloatval_t *, lbfgsfloatval_t *, lbfgs_evaluate_t,
              lbfgs_progress_t, void *, lbfgs_parameter_t *)

    void lbfgs_parameter_init(lbfgs_parameter_t *)
    lbfgsfloatval_t *lbfgs_malloc(int)
    void lbfgs_free(lbfgsfloatval_t *)

cdef lbfgsfloatval_t lbfgs_evaluate(void *cb_data_v,
        lbfgsconst_p x, lbfgsfloatval_t *g,
        int n, lbfgsfloatval_t step):
    cdef CallbackData cb_data = <CallbackData> cb_data_v
    cdef Weight point = Weight(n)
    point.data = x
    cdef Weight gradient = Weight(n)
    gradient.data = g
    # LBFGS does not set gradient to zero
    memset(g, 0, n * sizeof(double))
    return cb_data.loss_callback(cb_data.dataset, point, gradient)

cdef int lbfgs_progress(void *cb_data_v,
        lbfgsconst_p x, lbfgsconst_p g,
        lbfgsfloatval_t fx,
        lbfgsfloatval_t xnorm, lbfgsfloatval_t gnorm,
        lbfgsfloatval_t step, int n, int k, int ls):
    logging.info('Iteration {}: loss={} ||x||={} ||g||={}'.format(k, fx, xnorm, gnorm))
    return 0 # TODO if KeyboardInterrupt, non-zero

cdef optimize_lbfgs(loss_callback_t loss_callback, Dataset dataset, numpy.ndarray[DOUBLE, ndim=1] w):
    cdef CallbackData cb_data = CallbackData(dataset)
    cb_data.loss_callback = loss_callback
    # w -> x
    cdef lbfgsfloatval_t* x = lbfgs_malloc(w.size)
    memcpy(x, w.data, w.size * sizeof(double))
    # set optimization parameters
    cdef lbfgs_parameter_t params
    lbfgs_parameter_init(&params)
    # call lbfgs
    cdef double fx
    cdef int ret = lbfgs(w.size, x, &fx,
            lbfgs_evaluate,
            lbfgs_progress,
            <void*>cb_data, &params)
    # x -> w
    memcpy(w.data, x, w.size * sizeof(double))
    lbfgs_free(x)
