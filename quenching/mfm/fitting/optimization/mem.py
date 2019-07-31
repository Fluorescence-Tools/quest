from numpy import dot, log, sqrt
import mfm
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

settings = mfm.settings['fitting']['mem']

lower_bound = settings['lower_bound']
upper_bound = settings['upper_bound']
maxiter = settings['maxiter']
maxfun = settings['maxfun']
factr = settings['factr']
reg_scale = settings['reg_scale']



def maxent(A, b, nu, **kwargs):

    def func(x, weights, prior, A, b, l2):
        Ax = dot(A, x)
        res = (Ax - b) / weights
        grad_chi2 = 2.0 * dot(A.T, res)
        grad_S = l2 * (1 + log(prior * x))

        chi2 = sum(res**2)
        return chi2, grad_chi2 + grad_S

    # Initialization.
    n, m = A.shape
    weights = kwargs.get('w', np.ones(n))
    x0 = kwargs.get('x0', np.zeros(m))

    prior = kwargs.get('prior_distribution', np.ones(m))
    verbose = kwargs.get('verbose', False)

    # Treat each nu separately.
    bounds = [(lower_bound, upper_bound) for i in range(m)]
    l2 = (nu * reg_scale) ** 2
    result = fmin_l_bfgs_b(func, x0,
                  fprime=None,
                  args=(weights, prior, A, b, l2),
                  approx_grad=False,
                  bounds = bounds,
                  maxiter=maxiter,
                  maxfun=maxfun,
                  factr=factr,
                  pgtol=1e-6
                  )
    x = result[0]

    # Summarize results #
    res = dot(A, x) - b

    grad_chi2 = 2.0 * dot(A.T, res)
    norm_chi2 = sqrt(dot(grad_chi2, grad_chi2))

    grad_S = l2 * (1 + log(prior * x))
    norm_S = sqrt(dot(grad_S, grad_S))

    delta = (grad_chi2/norm_chi2-grad_S/norm_S)
    dgrad = 0.5*sqrt(dot(delta, delta))

    Q = norm_chi2-0.5*nu*norm_S

    if verbose:
        print 'iter #%d \tchi2 = %.6f  S = %.4f  Q = %.6f' % (norm_chi2, norm_S, Q, dgrad)

    return x


