import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import parallel, prange
from libc.stdlib cimport malloc, free
from libc.math cimport sin, cos, sqrt, fabs


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def calculate_kappa2_distance(float[:,:,:] xyz, int aid1, int aid2, int aia1, int aia2):
    """Calculates the orientation factor kappa2 and the distance of a trajectory given the atom-indices of the
    donor and the acceptor.

    :param xyz: numpy-array (frame, atom, xyz)
    :param aid1: int, atom-index of d-dipole 1
    :param aid2: int, atom-index of d-dipole 2
    :param aia1: int, atom-index of a-dipole 1
    :param aia2: int, atom-index of a-dipole 2

    :return: distances, kappa2
    """
    cdef int n_frames, i_frame
    # coordinates of the dipole
    cdef float d11, d12, d13, d21, d22, d23
    cdef float a11, a12, a13, a21, a22, a23

    # length of the dipole
    cdef float dD21, dA21

    # normal vector of the dipoles
    cdef float muD1, muD2, muD3
    cdef float muA1, muA2, muA3

    # connection vector of the dipole
    cdef float RDA1, RDA2, RDA3

    # vector to the middle of the dipoles
    cdef float dM1, dM2, dM3
    cdef float aM1, aM2, aM3

    # normalized DA-connection vector
    cdef float nRDA1, nRDA2, nRDA3
    cdef float kappa, kappa2

    n_frames = xyz.shape[0]
    cdef np.ndarray[ndim=1, dtype=np.float32_t] k2 = np.empty(n_frames, dtype=np.float32)
    cdef np.ndarray[ndim=1, dtype=np.float32_t] d = np.empty(n_frames, dtype=np.float32)

    for i_frame in range(n_frames):
        ### Donor ###
        # cartesian coordinates of the donor-dipole
        d11 = xyz[i_frame, aid1, 0]
        d12 = xyz[i_frame, aid1, 1]
        d13 = xyz[i_frame, aid1, 2]

        d21 = xyz[i_frame, aid2, 0]
        d22 = xyz[i_frame, aid2, 1]
        d23 = xyz[i_frame, aid2, 2]

        # distance between the two end points of the donor
        dD21 = sqrt( (d11 - d21)*(d11 - d21) +
                     (d12 - d22)*(d12 - d22) +
                     (d13 - d23)*(d13 - d23)
        )

        # normal vector of the donor-dipole
        muD1 = (d21 - d11) / dD21
        muD2 = (d22 - d12) / dD21
        muD3 = (d23 - d13) / dD21

        # vector to the middle of the donor-dipole
        dM1 = d11 + dD21 * muD1 / 2.0
        dM2 = d12 + dD21 * muD2 / 2.0
        dM3 = d13 + dD21 * muD3 / 2.0

        ### Acceptor ###
        # cartesian coordinates of the acceptor
        a11 = xyz[i_frame, aia1, 0]
        a12 = xyz[i_frame, aia1, 1]
        a13 = xyz[i_frame, aia1, 2]

        a21 = xyz[i_frame, aia2, 0]
        a22 = xyz[i_frame, aia2, 1]
        a23 = xyz[i_frame, aia2, 2]

        # distance between the two end points of the acceptor
        dA21 = sqrt( (a11 - a21)*(a11 - a21) +
                     (a12 - a22)*(a12 - a22) +
                     (a13 - a23)*(a13 - a23)
        )

        # normal vector of the acceptor-dipole
        muA1 = (a21 - a11) / dA21
        muA2 = (a22 - a12) / dA21
        muA3 = (a23 - a13) / dA21

        # vector to the middle of the acceptor-dipole
        aM1 = a11 + dA21 * muA1 / 2.0
        aM2 = a12 + dA21 * muA2 / 2.0
        aM3 = a13 + dA21 * muA3 / 2.0

        # vector connecting the middle of the dipoles
        RDA1 = dM1 - aM1
        RDA2 = dM2 - aM2
        RDA3 = dM3 - aM3

        # Length of the dipole-dipole vector (distance)
        dRDA = sqrt(RDA1*RDA1 + RDA2*RDA2 + RDA3*RDA3)

        # Normalized dipole-diple vector
        nRDA1 = RDA1 / dRDA
        nRDA2 = RDA2 / dRDA
        nRDA3 = RDA3 / dRDA

        # Orientation factor kappa2
        kappa = muA1*muD1 + muA2*muD2 + muA3*muD3 - 3.0 * (muD1*nRDA1+muD2*nRDA2+muD3*nRDA3) * (muA1*nRDA1+muA2*nRDA2+muA3*nRDA3)
        kappa2 = kappa * kappa

        k2[i_frame] = kappa2
        d[i_frame] = dRDA

    return d, k2

@cython.wraparound(False)
@cython.boundscheck(False)
def below_min_distance(float[:,:,:] xyz, float min_distance, int[:] atom_selection=np.empty(0, dtype=np.int32)):
    """Takes the xyz-coordinates (frame, atom, xyz) of a trajectory as an argument an returns a vector of booleans
    of length of the number of frames. The bool is False if the frame contains a atomic distance smaller than the
    min distance.

    :param xyz: numpy array
        The coordinates (frame nbr, atom nbr, coord)

    :param min_distance: float
        Minimum distance if a distance

    :return: numpy-array
        If a atom-atom distance within a frame is smaller than min_distance the value within the array is True otherwise
        it is False.

    """
    cdef int i, j, n_atoms, n_frames
    cdef int i_frame, i_atom, j_atom
    cdef float x1, y1, z1
    cdef float x2, y2, z2
    cdef float dx, dy, dz

    n_frames = xyz.shape[0]
    cdef np.ndarray[ndim=1, dtype=np.uint8_t] re = np.zeros(n_frames, dtype=np.uint8)

    cdef int[:] atoms = range(xyz.shape[1]) if atom_selection.shape[0] == 0 else atom_selection
    n_atoms = atoms.shape[0]

    for i_frame in prange(n_frames, nogil=True):
    #for i_frame in range(n_frames):

        for i in range(n_atoms):
            i_atom = atoms[i]
            x1 = xyz[i_frame, i_atom, 0]
            y1 = xyz[i_frame, i_atom, 1]
            z1 = xyz[i_frame, i_atom, 2]

            for j in range(i + 1, n_atoms):
                j_atom = atoms[j]

                x2 = xyz[i_frame, j_atom, 0]
                y2 = xyz[i_frame, j_atom, 1]
                z2 = xyz[i_frame, j_atom, 2]

                dx = fabs(x1-x2)
                dy = fabs(y1-y2)
                dz = fabs(z1-z2)

                if dx + dy + dz < min_distance:
                    re[i_frame] += 1
                    break

            if re[i_frame] > 0:
                break
    return re



@cython.wraparound(False)
@cython.boundscheck(False)
def min_distance_sq(float[:,:,:] xyz):
    """Determines the minimum distance in each frame of a trajectory

    :param xyz: numpy array
        The coordinates (frame nbr, atom nbr, coord)

    :return: numpy-array

    """
    cdef int i_frame, i_atom, j_atom
    cdef float x1, y1, z1
    cdef float x2, y2, z2
    cdef float distance_sq

    n_frames = xyz.shape[0]
    n_atoms = xyz.shape[1]

    cdef np.ndarray[ndim=1, dtype=np.float32_t] re = np.zeros(n_frames, dtype=np.float32)
    for i_frame in prange(n_frames, nogil=True):

        for i_atom in range(n_atoms):
            x1 = xyz[i_frame, i_atom, 0]
            y1 = xyz[i_frame, i_atom, 1]
            z1 = xyz[i_frame, i_atom, 2]

            for j_atom in range(i_atom + 1, n_atoms):
                x2 = xyz[i_frame, j_atom, 0]
                y2 = xyz[i_frame, j_atom, 1]
                z2 = xyz[i_frame, j_atom, 2]

                distance_sq = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2)
                re[i_frame] = min(re[i_frame], distance_sq)
    return re


@cython.wraparound(False)
@cython.boundscheck(False)
def rotate(float[:,:,:] xyz, float[:,:] rm):
    """ Rotates a trajectory (frame, atom, coord)

    :param xyz: numpy array
        The coordinates (frame nbr, atom nbr, coord)

    :param rm: numpy array 3x3 dtpye np.float32 - the rotation matrix
    :return:

    Examples
    --------

    >>> from mfm.structure.trajectory import TrajectoryFile
    >>> from mfm.structure.cStructure import rotate
    >>> import numpy as np
    >>> traj = TrajectoryFile('stride_100.h5')
    >>> xyz = traj.xyz
    >>> b = np.array([[-0.856274009, 0.513258278, -0.057972118], [0.513934493, 0.835381866, -0.194957629], [-0.051634759, -0.196731016, -0.979096889]], dtype=np.float32)
    >>> rotate(xyz, b)

    """
    cdef int i_frame, i_atom
    cdef int n_frames, n_atoms
    cdef int i, j
    cdef float t1, t2, t3
    cdef x, y, z

    n_frames = xyz.shape[0]
    n_atoms = xyz.shape[1]
    for i_frame in range(n_frames):
        for i_atom in range(n_atoms):
            # matrix vector product
            x = xyz[i_frame, i_atom, 0]
            y = xyz[i_frame, i_atom, 1]
            z = xyz[i_frame, i_atom, 2]

            t1 = rm[0, 0] * x + rm[0, 1] * y + rm[0, 2] * z
            t2 = rm[1, 0] * x + rm[1, 1] * y + rm[1, 2] * z
            t3 = rm[2, 0] * x + rm[2, 1] * y + rm[2, 2] * z

            xyz[i_frame, i_atom, 0] = t1
            xyz[i_frame, i_atom, 1] = t2
            xyz[i_frame, i_atom, 2] = t3



@cython.wraparound(False)
@cython.boundscheck(False)
def translate(np.ndarray[ndim=3, dtype=np.float32_t] xyz, np.ndarray[ndim=1, dtype=np.float32_t] vector):
    """ Translate a trajectory by an vector

    :param xyz: numpy array
        (frame nbr, atom_number, coord)
    :param vector:
    :return:
    """
    cdef int i_frame, i_atom, i_dim
    cdef int n_frames, n_atoms, n_dim
    n_frames = xyz.shape[0]
    n_atoms = xyz.shape[1]

    for i_frame in range(n_frames):
        for i_atom in range(n_atoms):
            for i_dim in range(3):
                xyz[i_frame, i_atom, i_dim] += vector[i_dim]


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

