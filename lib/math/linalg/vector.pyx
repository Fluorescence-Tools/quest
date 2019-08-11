import numpy as np
cimport numpy as np
from libc.math cimport exp, ceil, acos, acos, sqrt, atan2
cimport cython
from cython.parallel import parallel, prange

@cython.boundscheck(False)
cdef inline double norm3c(double[:] v):
    """
    :param v: 1D numpy-array
    :return: normalized numpy-array
    """
    cdef double sum = 0.0
    cdef int k
    for k in xrange(3):
        sum += v[k]*v[k]
    return sqrt(sum)

@cython.boundscheck(False)
def dist(double[:] u, double[:] v):
    cdef int i
    cdef float d2
    d2 = 0.0
    for i in range(u.shape[0]):
        d2 += (u[i]-v[i])*(u[i]-v[i])
    return sqrt(d2)

@cython.boundscheck(False)
def dist2(double[:] u, double[:] v):
    cdef int i
    cdef float d2
    d2 = 0.0
    for i in range(u.shape[0]):
        d2 += (u[i]-v[i])*(u[i]-v[i])
    return d2

@cython.boundscheck(False)
cdef double distc(double[:] u, double[:] v):
    cdef int i
    cdef double d2
    d2 = 0.0
    for i in range(3):
        d2 += (u[i]-v[i])*(u[i]-v[i])
    return sqrt(d2)


@cython.boundscheck(False)
def norm(double[:] v):
    """
    :param v: 1D numpy-array
    :return: normalized numpy-array
    """
    cdef double sum = 0.0
    cdef int k
    for k in xrange(v.shape[0]):
        sum += v[k]*v[k]
    return sqrt(sum)

@cython.boundscheck(False)
def dot(double[:] a, double[:] b):
    """
    :param a: 1D numpy-array
    :param b: 1D numpy-array
    :return: dot-product of the two numpy arrays
    """
    cdef double sum = 0.0
    cdef int k
    for k in xrange(a.shape[0]):
        sum += a[k]*b[k]
    return sum

@cython.boundscheck(False)
cdef inline double dot3c(double[:] a, double[:] b):
    """
    :param a: 1D numpy-array
    :param b: 1D numpy-array
    :return: dot-product of the two numpy arrays
    """
    cdef double sum = 0.0
    cdef int k
    for k in xrange(3):
        sum += a[k]*b[k]
    return sum

@cython.boundscheck(False)
def cross(double[:] a, double[:] b):
    """
    :param a: numpy array of length 3
    :param b: numpy array of length 3
    :return: cross-product of a and b
    """
    cdef np.ndarray[np.float64_t, ndim=1] o = np.empty(3, dtype=np.float64)
    o[0] = a[1]*b[2]-a[2]*b[1]
    o[1] = a[2]*b[0]-a[0]*b[2]
    o[2] = a[0]*b[1]-a[1]*b[0]
    return o

@cython.boundscheck(False)
cdef np.ndarray cross3c(double[:] a, double[:] b):
    """
    :param a: numpy array of length 3
    :param b: numpy array of length 3
    :return: cross-product of a and b
    """
    cdef np.ndarray[np.float64_t, ndim=1] o = np.empty(3, dtype=np.float64)
    o[0] = a[1]*b[2]-a[2]*b[1]
    o[1] = a[2]*b[0]-a[0]*b[2]
    o[2] = a[0]*b[1]-a[1]*b[0]
    return o

@cython.boundscheck(False)
def sub(double[:] a, double[:] b):
    cdef int k
    cdef np.ndarray[np.float64_t, ndim=1] o = np.empty(a.shape[0], dtype=np.float64)
    for k in xrange(a.shape[0]):
        o[k] = a[k]-b[k]
    return o

@cython.boundscheck(False)
cdef inline np.ndarray sub3c(double[:] a, double[:] b):
    cdef int k
    cdef np.ndarray[np.float64_t, ndim=1] o = np.empty(3, dtype=np.float64)
    for k in xrange(3):
        o[k] = a[k]-b[k]
    return o

@cython.boundscheck(False)
cdef np.ndarray add(double[:] a, double[:] b):
    cdef int k
    cdef np.ndarray[np.float64_t, ndim=1] o = np.empty(a.shape[0], dtype=np.float64)
    for k in xrange(a.shape[0]):
        o[k] = a[k]+b[k]
    return o

@cython.boundscheck(False)
@cython.cdivision(True)
def angle(double[:] a, double[:] b, double[:] c):
    """
    :param a: numpy array
    :param b: numpy array
    :param c: numpy array
    :return: angle between three vectors/points in space
    """
    cdef double r12n, r23n
    r12 = sub3c(a, b)
    r23 = sub3c(c, b)
    r12n = norm3c(r12)
    r23n = norm3c(r23)
    return acos(dot3c(r12,r23) / (r12n*r23n) )

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def dihedral(double[:] v1, double[:] v2,
              double[:] v3, double[:] v4):
    """
    Given the coordinates of the four points, obtain the vectors b1, b2, and b3 by vector subtraction.
    Let me use the nonstandard notation <v> to denote v/|v|, the unit vector in the direction of the
    vector v. Compute n1=<b1xb2> and n2=<b2xb3>, the normal vectors to the planes containing b1 and b2,
    and b2 and b3 respectively. The angle we seek is the same as the angle between n1 and n2.

    The three vectors n1, <b2>, and m1:=n1x<b2> form an orthonormal frame. Compute the coordinates of
    n2 in this frame: x=n1*n2 and y=m1*n2. (You don't need to compute <b2>*n2 as it should always be zero.)

    The dihedral angle, with the correct sign, is atan2(y,x).

    (The reason I recommend the two-argument atan2 function to the traditional cos-1 in this case is both
    because it naturally produces an angle over a range of 2pi, and because cos-1 is poorly conditioned
    when the angle is close to 0 or +-pi.)

    :param a: numpy array
    :param b: numpy array
    :param c: numpy array
    :param d: numpy array
    :return: dihedral angle between four vectors
    """
    # TODO: also calculate angle between vectors here: speed up calculations
    cdef double phi, n1Inv, n2Inv, m1Inv
    b1 = sub3c(v1, v2)
    b2 = sub3c(v2, v3)
    b3 = sub3c(v3, v4)
    n1 = cross3c(b1, b2)
    n2 = cross3c(b2, b3)
    m1 = cross3c(b2, n1)
    n1Inv = 1.0 / norm3c(n1)
    n2Inv = 1.0 / norm3c(n2)
    m1Inv = 1.0 / norm3c(m1)

    cos_phi = dot3c(n1,n2)*(n1Inv*n2Inv)
    sin_phi = dot3c(m1,n2)*(m1Inv*n2Inv)
    if cos_phi < -1: cos_phi = -1
    if cos_phi >  1: cos_phi =  1
    if sin_phi < -1: sin_phi = -1
    if sin_phi >  1: sin_phi =  1
    phi= -atan2(sin_phi,cos_phi)
    return phi

