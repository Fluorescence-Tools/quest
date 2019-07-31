import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
from libc.stdint cimport uint32_t, uint64_t
from libc.stdlib cimport malloc, free
from libc.math cimport exp, log, floor, pow


def get_weights(np.ndarray[np.uint8_t, ndim=1] rout, np.ndarray[np.uint32_t, ndim=1] tac,
               np.ndarray[np.float32_t, ndim=2] wt, uint64_t nPh):
    cdef uint64_t i
    cdef np.ndarray[np.float32_t, ndim=1] w = np.zeros(nPh, dtype=np.float32)
    for i in xrange(nPh):
        w[i] = wt[rout[i], tac[i]]
    return w


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def count_rate_filter(np.ndarray[np.uint64_t, ndim=1] mt, uint32_t nPh, 
                      uint64_t timeWindow, uint64_t timeChunk, double tolerance):
    cdef np.ndarray[np.float32_t, ndim=1] w = np.ones(nPh, dtype=np.float32)
    cdef uint64_t i, j, k
    cdef uint64_t timeChunkLeftBorder = 0
    cdef uint64_t timeChunkRightBorder = 0
    cdef uint64_t timeWindowLeftBorder = 0
    cdef uint64_t timeWindowRightBorder = 0
    cdef double countRateChunk, countRateWindow
    for i in xrange(nPh):
        if mt[i] - mt[timeChunkLeftBorder] < timeChunk and timeChunkRightBorder < nPh - <uint64_t> 1:
            timeChunkRightBorder += 1
        else:
            countRateChunk = <double> (timeChunkRightBorder - timeWindowLeftBorder) / <double> timeChunk
            for j in xrange(timeChunkLeftBorder, timeChunkRightBorder):
                if mt[j] - mt[timeWindowLeftBorder] < timeWindow and timeWindowRightBorder < timeChunkRightBorder:
                    timeWindowRightBorder += 1
                else:
                    countRateWindow =  <double> (timeWindowRightBorder - timeWindowLeftBorder) / <double> timeWindow
                    if countRateWindow > tolerance * countRateChunk:
                        for k in range(timeWindowLeftBorder, timeWindowRightBorder):
                            w[k] = 0.0
                    timeWindowLeftBorder = timeWindowRightBorder
            timeChunkLeftBorder = timeChunkRightBorder
    return w


@cython.boundscheck(False)
cdef void make_fine(np.ndarray[np.uint64_t, ndim=1] t, np.ndarray[np.uint32_t, ndim=1] tac, uint32_t nTAC):
    cdef int i
    for i in prange(1, t.shape[0], nogil=True):
        t[i] = t[i]*nTAC + tac[i]


cdef count_photons(float[:, :] w):
    cdef int j, i
    cdef np.ndarray[np.uint64_t, ndim=1] k = np.zeros(w.shape[0], dtype=np.uint64)
    for j in range(w.shape[0]):
        for i in range(w.shape[1]):
            if w[j,i] != 0.0:
                k[j] += 1
    return k

@cython.cdivision(True)
@cython.boundscheck(False)
cdef void coarsen(uint64_t[:, :] t,  float[:, :] w):
    cdef int i, j
    for j in prange(t.shape[0], nogil=True):
        t[j,1] = t[j, 1] / 2
        for i in xrange(2, t[j, 0]):
            t[j,i] = t[j, i] / 2
            if t[j, i-1] == t[j, i]:
                w[j, i-1] += w[j, i]
                w[j, i] = 0.0
    compact(t, w, 0)


@cython.boundscheck(False)
cdef inline void compact(uint64_t[:, :] t, float[:, :] w, char full = 0):
    cdef uint64_t i, j, k, r
    for j in xrange(t.shape[0]):
        k = 1
        r = t.shape[1] if full else t[j, 0]
        for i in xrange(1, r):
            if t[j, k]!=t[j, i] and w[j,i]!=0:
                k += 1
                t[j, k] = t[j, i]
                w[j, k] = w[j, i]
        t[j, 0] = k - 1


@cython.boundscheck(False)
@cython.cdivision(True)
def correlate_tp(mt, tac, rout, cr_filter, float[:] w1, float[:] w2, uint32_t B, uint32_t nc, uint32_t fine, uint32_t nTAC):
    print "correlation algorithm: tp"
    # correlate with TAC
    if fine > 0:
        make_fine(mt, tac, nTAC)
    # make 2 corr-channels
    cdef np.ndarray[np.uint64_t, ndim=2] t = np.vstack([mt, mt])
    cdef np.ndarray[np.float32_t, ndim=2] w = np.vstack([w1*cr_filter, w2*cr_filter])
    np1, np2 = count_photons(w)
    compact(t, w, 1)
    # MACRO-Times
    mt1max, mt2max = t[0,t[0,0]], t[1,t[1,0]]
    mt1min, mt2min = t[0,1], t[1,1]
    dt1 = mt1max - mt1min
    dt2 = mt2max - mt2min
    # calculate tau axis
    cdef np.ndarray[np.uint64_t, ndim=1] taus = np.zeros(nc*B, dtype=np.uint64)
    cdef np.ndarray[np.float32_t, ndim=1] corr = np.zeros(nc*B, dtype=np.float32)
    cdef uint64_t j
    for j in xrange(1, nc * B):
        taus[j] = taus[j-1] + <long long> pow(2.0, floor(<double> j / B))
    # correlation
    cdef uint64_t n, b, pw
    cdef uint64_t ta, ti
    cdef uint64_t ca, ci, pa, pi, shift
    for n in range(nc):
        print "cascade %s\tnph1: %s\tnph2: %s" % (n, t[0,0], t[1,0])
        for b in prange(B, nogil=True):
            j = (n*B+b)
            shift = taus[j]/(<int> pow(2.0, floor(<double> j / B)))
            # STARTING CHANNEL
            ca = 0 if t[0,1]<t[1,1] else 1 # currently active correlation channel
            ci = 1 if t[0,1]<t[1,1] else 0 # currently inactive correlation channel
            # POSITION ON ARRAY
            pa, pi = 0, 1           # position on active (pa), previous (pp) and inactive (pi) channel
            while pa<t[ca,0] and pi<=t[ci,0]:
                pa += 1
                if ca == 1:
                    ta = t[ca,pa] + shift
                    ti = t[ci,pi]
                else:
                    ta = t[ca,pa]
                    ti = t[ci,pi] + shift
                if ta>=ti:
                    if ta == ti:
                        corr[j] += (w[ci,pi]*w[ca,pa])
                    ca, ci = ci, ca
                    pa, pi = pi, pa
        # COARSE - coarsening by a factor of 2 each cascade of B lag times
        coarsen(t, w)
    return np1, np2, dt1, dt2, taus, corr