from . import datatools
from . import rdf
from . import _special


def i0(x):
    """
    Modified Bessel-function I0(x) for any real x
    (according to numerical recipes function - `bessi0`,
    Polynomal approximation Abramowitz and Stegun )

    References
    ----------

    .. [1] Abramowitz, M and Stegun, I.A. 1964, Handbook of Mathematical
       Functions, Applied Mathematics Series, Volume 55 (Washington:
       National Bureal of Standards; reprinted 1968 by Dover Publications,
       New York), Chapter 10

    :param x: float
    :return:
    """
    return _special.i0(x)


