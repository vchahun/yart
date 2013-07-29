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

_ERROR_MESSAGES = {
    LBFGSERR_UNKNOWNERROR: "Unknown error." ,
    LBFGSERR_LOGICERROR: "Logic error.",
    LBFGSERR_OUTOFMEMORY: "Insufficient memory.",
    LBFGSERR_CANCELED: "The minimization process has been canceled.",
    LBFGSERR_INVALID_N: "Invalid number of variables specified.",
    LBFGSERR_INVALID_N_SSE: "Invalid number of variables (for SSE) specified.",
    LBFGSERR_INVALID_X_SSE: "The array x must be aligned to 16 (for SSE).",
    LBFGSERR_INVALID_EPSILON: "Invalid parameter epsilon specified.",
    LBFGSERR_INVALID_TESTPERIOD: "Invalid parameter past specified.",
    LBFGSERR_INVALID_DELTA: "Invalid parameter delta specified.",
    LBFGSERR_INVALID_LINESEARCH: "Invalid parameter linesearch specified.",
    LBFGSERR_INVALID_MINSTEP: "Invalid parameter max_step specified.",
    LBFGSERR_INVALID_MAXSTEP: "Invalid parameter max_step specified.",
    LBFGSERR_INVALID_FTOL: "Invalid parameter ftol specified.",
    LBFGSERR_INVALID_WOLFE: "Invalid parameter wolfe specified.",
    LBFGSERR_INVALID_GTOL: "Invalid parameter gtol specified.",
    LBFGSERR_INVALID_XTOL: "Invalid parameter xtol specified.",
    LBFGSERR_INVALID_MAXLINESEARCH:
        "Invalid parameter max_linesearch specified.",
    LBFGSERR_INVALID_ORTHANTWISE: "Invalid parameter orthantwise_c specified.",
    LBFGSERR_INVALID_ORTHANTWISE_START:
        "Invalid parameter orthantwise_start specified.",
    LBFGSERR_INVALID_ORTHANTWISE_END:
        "Invalid parameter orthantwise_end specified.",
    LBFGSERR_OUTOFINTERVAL:
        "The line-search step went out of the interval of uncertainty.",
    LBFGSERR_INCORRECT_TMINMAX:
        "A logic error occurred;"
        " alternatively, the interval of uncertainty became too small.",
    LBFGSERR_ROUNDING_ERROR:
        "A rounding error occurred;"
        " alternatively, no line-search step satisfies"
        " the sufficient decrease and curvature conditions.",
    LBFGSERR_MINIMUMSTEP: "The line-search step became smaller than min_step.",
    LBFGSERR_MAXIMUMSTEP: "The line-search step became larger than max_step.",
    LBFGSERR_MAXIMUMLINESEARCH:
        "The line-search routine reaches the maximum number of evaluations.",
    LBFGSERR_MAXIMUMITERATION:
        "The algorithm routine reaches the maximum number of iterations.",
    LBFGSERR_WIDTHTOOSMALL:
        "Relative width of the interval of uncertainty is at most xtol.",
    LBFGSERR_INVALIDPARAMETERS:
        "A logic error (negative line-search step) occurred.",
    LBFGSERR_INCREASEGRADIENT:
        "The current search direction increases the objective function value.",
}

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
    return cb_data.loss_callback(cb_data.dataset, point, gradient, cb_data.l2)

cdef int lbfgs_progress(void *cb_data_v,
        lbfgsconst_p x, lbfgsconst_p g,
        lbfgsfloatval_t fx,
        lbfgsfloatval_t xnorm, lbfgsfloatval_t gnorm,
        lbfgsfloatval_t step, int n, int k, int ls):
    logging.info('Iteration {}: loss={} ||x||={} ||g||={}'.format(k, fx, xnorm, gnorm))
    return 0 # TODO if KeyboardInterrupt, non-zero

class LBFGSError(Exception):
    pass

cdef optimize_lbfgs(loss_callback_t loss_callback, Dataset dataset,
        numpy.ndarray[DOUBLE, ndim=1] w, float l2):
    cdef CallbackData cb_data = CallbackData(dataset, l2)
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
    try:
        if (ret == LBFGS_SUCCESS
            or ret == LBFGS_ALREADY_MINIMIZED
            or ret == LBFGSERR_ROUNDING_ERROR):
            # x -> w
            memcpy(w.data, x, w.size * sizeof(double))
        elif ret == LBFGSERR_OUTOFMEMORY:
            raise MemoryError
        else:
            raise LBFGSError(_ERROR_MESSAGES[ret])
    finally:
        lbfgs_free(x)
