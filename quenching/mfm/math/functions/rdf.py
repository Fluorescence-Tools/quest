import numpy as np
from . import _special


def gaussian_chain_ree(l, n):
    """
    Calculates the root mean square end-to-end distance of a Gaussian chain

    :param l: float
        The length of a segment
    :param n: int
        The number of segments
    :return:
    """
    return l * np.sqrt(n)


def gaussian_chain(r, l, n):
    """
    Calculates the radial distribution function of a Gaussian chain in three dimensions

    :param n: int
        The number of segments
    :param l: float
        The segment length
    :param r: numpy-array
        values of r should be in range [0, 1) - not including 1

    ..plot:: plots/rdf-gauss.py


    """
    r2_mean = gaussian_chain_ree(l, n) ** 2
    return 4*np.pi*r**2/(2./3. * np.pi*r2_mean)**(3./2.) * np.exp(-3./2. * r**2 / r2_mean)


def worm_like_chain(r, kappa):
    """
    Calculates the radial distribution function of a worm-like-chain given the multiple piece-solution according to:
    The radial distribution function of worm-like chain

    .. plot:: plots/rdf-wlc.py

    :param r: numpy-array
        values of r should be in range [0, 1) - not including 1
    :param kappa: float

    Examples
    --------

    Calculate the radial distirbution function for a given kappa. This function only work for distances
    smaller than *1*

    >>> import mfm.math.functions.rdf as rdf
    >>> import numpy as np
    >>> r = np.linspace(0, 0.99, 50)
    >>> kappa = 1.0
    >>> rdf.worm_like_chain(r, kappa)
    array([  4.36400392e-06,   4.54198260e-06,   4.95588702e-06,
             5.64882576e-06,   6.67141240e-06,   8.09427111e-06,
             1.00134432e-05,   1.25565315e-05,   1.58904681e-05,
             2.02314725e-05,   2.58578047e-05,   3.31260228e-05,
             4.24918528e-05,   5.45365051e-05,   7.00005025e-05,
             8.98266752e-05,   1.15215138e-04,   1.47693673e-04,
             1.89208054e-04,   2.42238267e-04,   3.09948546e-04,
             3.96381668e-04,   5.06711496e-04,   6.47572477e-04,
             8.27491272e-04,   1.05745452e-03,   1.35165891e-03,
             1.72850634e-03,   2.21192991e-03,   2.83316807e-03,
             3.63314697e-03,   4.66568936e-03,   6.00184475e-03,
             7.73573198e-03,   9.99239683e-03,   1.29382877e-02,
             1.67949663e-02,   2.18563930e-02,   2.85090497e-02,
             3.72510109e-02,   4.86977611e-02,   6.35415230e-02,
             8.23790455e-02,   1.05199154e-01,   1.30049143e-01,
             1.49953168e-01,   1.47519190e-01,   9.57787954e-02,
             1.45297018e-02,   1.53180248e-08])


    References
    ----------

    .. [1] Becker NB, Rosa A, Everaers R, Eur Phys J E Soft Matter, 2010 May;32(1):53-69,
       The radial distribution function of worm-like chains.


    """
    return _special.worm_like_chain(r, kappa)