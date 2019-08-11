"""
Functions related to random numbers
"""

from math import sqrt
import _rand
import numpy as np
from scipy.stats import norm


def weighted_choice(weights, n=1):
    """
    A weighted random number generator. The random number generator generates
    random numbers between zero and the length of the provided weight-array. The
    elements of the weight-arrays are proportional to the probability that the corresponding
    integer random number is generated.

    :param weights: array-like
    :param n: int
        number of random values to be generated
    :return: Returns an array containing the random values

    Examples
    --------

    >>> import numpy as np
    >>> weighted_choice(np.array([0.1, 0.5, 3]), 10)
    array([1, 2, 2, 2, 2, 2, 2, 2, 2, 1], dtype=uint32)

    http://eli.thegreenplace.net/2010/01/22/weighted-random-generation-in-python/
    """
    return _rand.weighted_choice(weights, n)


def brownian(x0, n, dt, delta, out=None):
    """\
    Generate an instance of Brownian motion (i.e. the Wiener process):

    .. math::
        X(t) = X(0) + N(0, \delta^2 \cdot t; 0, t)

    where N(a,b; t0, t1) is a normally distributed random variable with mean a and
    variance b.  The parameters t0 and t1 make explicit the statistical
    independence of N on different time intervals; that is, if [t0, t1) and
    [t2, t3) are disjoint intervals, then N(a, b; t0, t1) and N(a, b; t2, t3)
    are independent.

    Written as an iteration scheme,

    .. math::
        X(t + dt) = X(t) + N(0, \delta^2 \cdot dt; t, t+dt)


    If :math:`x_0` is an array (or array-like), each value in :math:`x_0` is treated as
    an initial condition, and the value returned is a numpy array with one
    more dimension than :math:`x_0`.

    Arguments
    ---------
    x0 : float or numpy array (or something that can be converted to a numpy array
         using numpy.asarray(x0)).
        The initial condition(s) (i.e. position(s)) of the Brownian motion.
    n : int
        The number of steps to take.
    dt : float
        The time step.
    delta : float
        delta determines the "speed" of the Brownian motion.  The random variable
        of the position at time t, X(t), has a normal distribution whose mean is
        the position at time t=0 and whose variance is delta**2*t.
    out : numpy array or None
        If `out` is not None, it specifies the array in which to put the
        result.  If `out` is None, a new numpy array is created and returned.

    Returns
    -------
    A numpy array of floats with shape `x0.shape + (n,)`.

    Note that the initial value `x0` is not included in the returned array.
    """

    x0 = np.asarray(x0)

    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = norm.rvs(size=x0.shape + (n,), scale=delta*sqrt(dt))

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples.
    np.cumsum(r, axis=-1, out=out)

    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)

    return out


def mc(e0, e1, kT):
    """
    Monte-Carlo acceptance criterion

    :param e0: float
        Previous energy
    :param e1: float
        Next energy
    :param kT: float
        Temperature
    :return: bool
    """
    if e1 < e0:
        return True
    else:
        r = np.random.random(1)[0]
        p = np.exp((e0 - e1) / kT)
        if r < p:
            return True
        else:
            return False