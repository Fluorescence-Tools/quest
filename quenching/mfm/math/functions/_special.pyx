import cython
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, exp, pow, fabs, floor
cdef double eps = 1e-9


#################### WLC stuff ########################
@cython.cdivision(True)
def i0(double x):
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

    :param x:
    :return:
    """
    cdef double ax, ans
    cdef double y
    ax = fabs(x)
    if ax < 3.75:
        y = x / 3.75
        y = y * y
        ans = 1.0 + y * (3.5156299 + y * (
            3.0899424 + y * (1.2067492 + y * (
                0.2659732 + y * (0.360768e-1 + y *
                                 0.45813e-2)))))
    else:
        y = 3.75 / ax
        ans = (exp(ax) / sqrt(ax)) * \
              (0.39894228 + y * (0.1328592e-1 + y * (
                  0.225319e-2 + y * ( -0.157565e-2 + y * (
                      0.916281e-2 + y * (-0.2057706e-1 + y * (
                          0.2635537e-1 + y * ( -0.1647633e-1 + y *
                                               0.392377e-2))))))))
    return ans


def binCount(data, binWidth=16, binMin=0, binMax=4095):
    """
    Count number of occurrences of each value in array of non-negative ints.

    :param data: array_like
        1-dimensional input array
    :param binWidth:
    :param binMin:
    :param binMax:
    :return: As return values a numpy array with the bins and a array containing the counts is obtained.
    """
    # OK - rounding in C: rint, round
    nMin = np.rint(binMin / binWidth)
    nMax = np.rint(binMax / binWidth)
    nBins = nMax - nMin
    count = np.zeros(nBins, dtype=np.float32)
    bins = np.arange(nMin, nMax, dtype=np.float32)
    bins *= binWidth
    for i in range(data.shape[0]):
        bin = np.rint((data[i] / binWidth)) - nMin
        if bin < nBins:
            count[bin] += 1
    return bins, count


@cython.boundscheck(False)
def histogram1D(np.ndarray pos not None, np.ndarray data=None, long nbPt=100):
    """
    Calculates histogram of pos weighted by weights.

    :param pos: 2Theta array
    :param weights: array with intensities
    :param bins: number of output bins
    :param pixelSize_in_Pos: size of a pixels in 2theta
    :param nthread: maximum number of thread to use. By default: maximum available.
        One can also limit this with OMP_NUM_THREADS environment variable

    :return 2theta, I, weighted histogram, raw histogram

    https://github.com/kif/pyFAI/blob/master/src/histogram.pyx
    """
    if data is None:
        data = np.ones(pos.shape[0], dtype=np.float64)
    else:
        assert pos.size == data.size
    cdef np.ndarray[np.float64_t, ndim = 1] ctth = pos.astype("float64").flatten()
    cdef np.ndarray[np.float64_t, ndim = 1] cdata = data.astype("float64").flatten()
    cdef np.ndarray[np.float64_t, ndim = 1] outData = np.zeros(nbPt, dtype=np.float64)
    cdef np.ndarray[np.int64_t, ndim = 1] outCount = np.zeros(nbPt, dtype=np.int64)
    cdef long size = pos.size
    cdef double tth_min = pos.min()
    cdef double tth_max = pos.max()
    cdef double idtth = (< double > (nbPt - 1)) / (tth_max - tth_min)
    cdef long bin = 0
    cdef long i = 0
    # with nogil:
    for i in range(size):
        bin = <long> (floor(((<double> ctth[i]) - tth_min) * idtth))
        outCount[bin] += 1
        outData[bin] += cdata[i]
    return outData, outCount


def smooth(np.ndarray[np.float64_t, ndim=1] x, int l, int m):
    cdef int i, j
    cdef np.ndarray[np.float64_t, ndim = 1] xz = np.empty(x.shape[0], dtype=np.float64)
    for i in range(l-m):
        xz[i] = 0
        for j in range(i-m, i+m):
            xz[i] += x[j]
            xz[i] /= (2 * m + 1)
    return xz


@cython.cdivision(True)
cdef double Qd(double r, double kappa) nogil:
    return pow((3.0 / (4.0 * 3.14159265359 * kappa)), (3.0 / 2.0)) * \
           exp(-3.0 / 4.0 * r * r / kappa) * (
    1.0 - 5.0 / 4.0 * kappa + 2.0 * r * r - 33.0 / 80.0 * r * r * r * r / kappa)

@cython.boundscheck(False)
@cython.cdivision(True)
def worm_like_chain(np.ndarray[np.float64_t, ndim=1] r, double kappa):
    """
    Calculates the radial distribution function of a worm-like-chain given the multiple piece-solution according to:
    The radial distribution function of worm-like chain
    Eur Phys J E, 32, 53-69 (2010)

    :param r: numpy-array
        values of r should be in range [0, 1) - not including 1
    :param kappa: float

    Examples
    --------

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
    cdef double a, b, c, d
    cdef int i, j, k
    cdef double Jsyd
    cdef double ri
    cdef double f1, f2, f3, f4
    cdef np.ndarray[np.float64_t, ndim=1] Qr

    a, b = 14.054, 0.473
    c = 1.0 - (1.0 + (0.38 * kappa ** (-0.95) ** (-5.0))) ** (-1.0 / 5.0)
    d = 1.0 if kappa <= 1.0 / 8.0 else 1.0 - 1.0 / (0.177 / (kappa - 0.111) + 6.40 * (kappa - 0.111) ** 0.783)
    Jsyd = 112.04 * kappa ** 2.0 * exp(0.246 / kappa - a * kappa) if kappa > 1.0 / 8.0 else Qd(0, kappa)
    Qr = np.empty_like(r)

    cdef double** cs = [[-3. / 4., 23. / 64., -7. / 65], [-1. / 2., 17. / 16., -9. / 16.]]
    for i in range(r.shape[0]):
        ri = r[i]
        f1 = 1.0 if (1.0 - (ri - eps) ** 2) < eps else Jsyd * ((1.0 - c * ri ** ri) / (1.0 - ri ** 2)) ** (5.0 / 2.0)
        f2 = 0.0
        for k in range(2):
            for j in range(3):
                f2 += cs[k][j] * kappa ** (<double> (k - 1)) * ri ** (2 * <double> (j + 1))
        f2 = exp(f2 / (1.0 - (ri - eps) ** 2))
        f3 = exp(-d * kappa * a * b * (1.0 + b) * ri ** 2.0 / (1.0 - b ** 2.0 * ri ** 2.0))
        f4 = i0(-d * kappa * a * (1.0 + b) * ri / (1.0 - b ** 2.0 * ri ** 2))
        Qr[i] = f1 * f2 * f3 * f4
    return Qr / Qr.sum()


