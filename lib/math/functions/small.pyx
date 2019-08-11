import numpy as np
import cython
cimport numpy as np
from libc.math cimport exp, ceil, acos, acos, sqrt, atan2


cdef extern from "math.h":
    double floor(double)


#def binCount(np.ndarray[np.uint64_t, ndim=1] data, int bin_width=16, nbins=256):
def binCount(data, binWidth=16, binMin=0, binMax=4095):
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
    #https://github.com/kif/pyFAI/blob/master/src/histogram.pyx
    """calculate histogram of pos weighted by data"""
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

@cython.boundscheck(False)
def weighted_choice(np.ndarray[np.double_t, ndim=1] weights, int n=1):
    #http://eli.thegreenplace.net/2010/01/22/weighted-random-generation-in-python/
    cdef double running_total = 0.0, rnd
    cdef int nWeights = weights.shape[0], i, j
    cdef np.ndarray[np.uint32_t, ndim=1] r = np.empty(n, dtype=np.uint32)
    cdef np.ndarray[np.float64_t, ndim=1] totals = np.empty(nWeights, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] rnds = np.random.random(n)

    for i in range(nWeights):
        running_total += weights[i]
        totals[i] = running_total

    for j in range(n):
        rnd = rnds[j] * running_total
        #rnd = rmt.random0i1e() * running_total
        for i in range(nWeights):
            if rnd <= totals[i]:
                r[j] = i
                break
    return r