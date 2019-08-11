import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import parallel, prange
from libc.stdlib cimport malloc, free
from libc.math cimport sin, cos, sqrt


@cython.boundscheck(False)
def atom_dist(double[:, :] aDist, int[:, :] resLookUp, double[:, :] xyz, int aID):
    cdef double d12, a1, a2, a3, b1, b2, b3
    cdef int n_residues, i, j, ia1, ia2
    n_residues = resLookUp.shape[0]
    for i in prange(n_residues, nogil=True):
        ia1 = resLookUp[i, aID]
        if ia1 < 0:
            continue
        a1 = xyz[ia1, 0]
        a2 = xyz[ia1, 1]
        a3 = xyz[ia1, 2]
        for j in range(i, n_residues):
            ia2 = resLookUp[j, aID]
            if ia2 < 0:
                continue
            b1 = xyz[ia2, 0] - a1
            b2 = xyz[ia2, 1] - a2
            b3 = xyz[ia2, 2] - a3

            d12 = sqrt(b1*b1 + b2*b2 + b3*b3)

            aDist[i, j] = d12
            aDist[j, i] = d12
    return aDist


@cython.cdivision(True)
@cython.boundscheck(False)
def internal_to_cartesian(np.ndarray internal_coordinates, double[:, :] r, int startPoint):
    cdef int i, k
    cdef int nAtoms = internal_coordinates.shape[0]

    cdef double[:] b = internal_coordinates['b']
    cdef double[:] a = internal_coordinates['a']
    cdef double[:] d = internal_coordinates['d']
    cdef int[:] ans = internal_coordinates['i']

    cdef int[:] ib = internal_coordinates['ib']
    cdef int[:] ia = internal_coordinates['ia']
    cdef int[:] id = internal_coordinates['id']

    cdef double sin_theta, cos_theta, sin_phi, cos_phi
    cdef double* p = <double *>malloc(nAtoms * sizeof(double) * 3)
    for i in prange(nAtoms, nogil=True):
        sin_theta = sin(a[i])
        cos_theta = cos(a[i])
        sin_phi = sin(d[i])
        cos_phi = cos(d[i])
        p[i*3 + 0] = b[i] * sin_theta * sin_phi
        p[i*3 + 1] = b[i] * sin_theta * cos_phi
        p[i*3 + 2] = b[i] * cos_theta

    cdef double v1, v2, v3
    cdef double u1, u2, u3
    cdef double ab1, ab2, ab3
    cdef double bc1, bc2, bc3
    cdef double cos_alpha, sin_alpha, nab, nbc
    for i in range(nAtoms):
        if ib[i] != 0 and ia[i] != 0 and id[i] != 0:
            ab1 = (r[ib[i], 0] - r[ia[i], 0])
            ab2 = (r[ib[i], 1] - r[ia[i], 1])
            ab3 = (r[ib[i], 2] - r[ia[i], 2])
            nab = sqrt(ab1*ab1+ab2*ab2+ab3*ab3)
            ab1 /= nab
            ab2 /= nab
            ab3 /= nab
            bc1 = (r[ia[i], 0] - r[id[i], 0])
            bc2 = (r[ia[i], 1] - r[id[i], 1])
            bc3 = (r[ia[i], 2] - r[id[i], 2])
            nbc = sqrt(bc1*bc1+bc2*bc2+bc3*bc3)
            bc1 /= nbc
            bc2 /= nbc
            bc3 /= nbc
            v1 = ab3 * bc2 - ab2 * bc3
            v2 = ab1 * bc3 - ab3 * bc1
            v3 = ab2 * bc1 - ab1 * bc2
            cos_alpha = ab1*bc1 + ab2*bc2 + ab3*bc3
            sin_alpha = sqrt(1.0 - cos_alpha * cos_alpha)
            v1 /= sin_alpha
            v2 /= sin_alpha
            v3 /= sin_alpha
            u1 = v2 * ab3 - v3 * ab2
            u2 = v3 * ab1 - v1 * ab3
            u3 = v1 * ab2 - v2 * ab1

            r[ans[i], 0] = r[ib[i], 0] + v1 * p[i * 3 + 0] + u1 * p[i * 3 + 1] - ab1 * p[i * 3 + 2]
            r[ans[i], 1] = r[ib[i], 1] + v2 * p[i * 3 + 0] + u2 * p[i * 3 + 1] - ab2 * p[i * 3 + 2]
            r[ans[i], 2] = r[ib[i], 2] + v3 * p[i * 3 + 0] + u3 * p[i * 3 + 1] - ab3 * p[i * 3 + 2]
    free(p)
