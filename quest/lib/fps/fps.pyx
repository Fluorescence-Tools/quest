import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport abort, malloc, free
from libc.math cimport sqrt, exp, cos, sin
from libc.stdint cimport int32_t, uint32_t, uint8_t
from cython.parallel import prange


cdef inline double dist3c2(double r1x, double r1y, double r1z, double r2x, double r2y, double r2z) nogil:
    return (r1x - r2x)*(r1x - r2x) + (r1y - r2y)*(r1y - r2y) + (r1z - r2z)*(r1z - r2z)


cdef extern from "mtrandom.h":
    cdef cppclass MTrandoms:
        void seedMT()
        double random0i1i() nogil
        double random0i1e() nogil


cdef MTrandoms rmt
rmt.seedMT()


@cython.boundscheck(False)
def density2points(int32_t n, int32_t npm, double dg, char[:] density, double[:] r0, int32_t ng):
    cdef int32_t ix, iy, iz, offset

    cdef np.ndarray[dtype=np.float64_t, ndim=2] _r = np.empty((n, 3), dtype=np.float64, order='C')
    cdef double[:] gd = np.arange(-npm, npm, dtype=np.float64) * dg

    n = 0
    for ix in range(-npm, npm):
        offset = ng * (ng * (ix + npm)) + npm
        for iy in range(-npm, npm):
            for iz in range(-npm, npm):
                if density[iz + offset] > 0:
                    _r[n, 0] = gd[ix + npm] + r0[0]
                    _r[n, 1] = gd[iy + npm] + r0[1]
                    _r[n, 2] = gd[iz + npm] + r0[2]
                    n += 1
            offset += ng
    return _r


@cython.cdivision(True)
@cython.boundscheck(False)
cdef inline ranDist(av1, av2, double* distances, uint32_t nSamples):
    cdef uint32_t i, i1, i2
    cdef int32_t lp1, lp2
    cdef double sTemp = 0.0

    cdef double[:, :] p1 = av1.points
    cdef double[:, :] p2 = av2.points
    lp1 = p1.shape[0]
    lp2 = p2.shape[0]
    for i in range(nSamples):
        i1 = <int>(rmt.random0i1e() * lp1)
        i2 = <int>(rmt.random0i1e() * lp2)
        distances[i] = sqrt(
            (p1[i1, 0] - p2[i2, 0]) * (p1[i1, 0] - p2[i2, 0]) + \
            (p1[i1, 1] - p2[i2, 1]) * (p1[i1, 1] - p2[i2, 1]) + \
            (p1[i1, 2] - p2[i2, 2]) * (p1[i1, 2] - p2[i2, 2])
        )


@cython.cdivision(True)
@cython.boundscheck(False)
def RDAMeanE(av1, av2, double R0=52.0, uint32_t nSamples=50000):
    """
    >>> import lib
    >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
    >>> structure = lib.Structure(pdb_filename)
    >>> av1 = lib.fps.AV(structure, residue_seq_number=72, atom_name='CB')
    >>> av2 = lib.fps.AV(structure, residue_seq_number=134, atom_name='CB')
    >>> lib.fps.RDAMeanE(av1, av2)
    52.602731299544686
    """
    cdef uint32_t i
    cdef double Esum = 0.0
    cdef double* d = <double*>malloc(nSamples * sizeof(double))
    ranDist(av1, av2, d, nSamples)
    for i in prange(nSamples, nogil=True):
        Esum += (1./(1.+(d[i]/R0)**6.0))
    Esum /= nSamples
    free(d)
    return (1./Esum - 1.)**(1./6.) * R0


@cython.cdivision(True)
@cython.boundscheck(False)
def RDAMean(av1, av2, uint32_t nSamples=50000):
    """
    >>> import lib
    >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
    >>> structure = lib.Structure(pdb_filename)
    >>> av1 = lib.fps.AV(structure, residue_seq_number=72, atom_name='CB')
    >>> av2 = lib.fps.AV(structure, residue_seq_number=134, atom_name='CB')
    >>> lib.fps.RDAMean(av1, av2)
    52.93390285282142
    """
    cdef uint32_t i
    cdef double RDA = 0.0

    cdef double* d = <double*>malloc(nSamples * sizeof(double))
    ranDist(av1, av2, d, nSamples)
    for i in prange(nSamples, nogil=True):
        RDA += d[i]
    free(d)
    return RDA / nSamples


def dRmp(av1, av2):
    """
    >>> import lib
    >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
    >>> structure = lib.Structure(pdb_filename)
    >>> av1 = lib.fps.AV(structure, residue_seq_number=72, atom_name='CB')
    >>> av2 = lib.fps.AV(structure, residue_seq_number=134, atom_name='CB')
    >>> lib.fps.dRmp(av1, av2)
    49.724995634807691
    """
    return np.sqrt(((av1.Rmp-av2.Rmp)**2).sum())



@cython.cdivision(True)
@cython.boundscheck(False)
def subav(np.ndarray density, int32_t ng, double dg, double[:] slow_radius, double[:, :] rs, double[:] r0):
    """
    density: density of the accessible volume dimension ng, ng, ng uint8 numpy array as obtained of fps
    dg: grid resolution of the density grid
    slow_radius: radius around the list of points. all points within a radius of r0 around the points in rs are part
    of the subav. each slow point is assiciated to one slow-radius
    rs: list of points (x,y,z) defining the subav
    r0: is the position of the accessible volume
    """
    cdef int32_t ix0, iy0, iz0, ix, iy, iz, slow_radius_idx, isa
    cdef char is_slow
    cdef int32_t n_slow_center = rs.shape[0]
    cdef np.ndarray[np.uint8_t, ndim=3] subav_density = np.copy(density)

    for ix in prange(ng, nogil=True):
        for iy in range(ng):
            for iz in range(ng):
                if subav_density[ix, iy, iz] == 0:
                    continue
                # count the overlaps with slow-centers
                is_slow = 0
                for isa in range(n_slow_center):
                    ix0 = <int>(((rs[isa, 0]-r0[0])/dg)) + (ng - 1)/2
                    iy0 = <int>(((rs[isa, 1]-r0[1])/dg)) + (ng - 1)/2
                    iz0 = <int>(((rs[isa, 2]-r0[2])/dg)) + (ng - 1)/2
                    slow_radius_idx = <int>(slow_radius[isa] / dg)
                    if ((ix - ix0)**2 + (iy - iy0)**2 + (iz - iz0)**2) < slow_radius_idx**2:
                        is_slow = 1
                        break
                if is_slow == 0:
                    subav_density[ix, iy, iz] = 0
    return subav_density


@cython.cdivision(True)
@cython.boundscheck(False)
def simulate_traj(np.ndarray[dtype=np.uint8_t, ndim=3] d, np.ndarray[dtype=np.uint8_t, ndim=3] ds,
                  double dg, double t_max, double t_step, double D, double slow_fact):
    """
    `d` density of whole av_macos_xcode in shape ng, ng, ng (as generated by fps library)
    `ds` density_slow of slow av_macos_xcode (has to be same shape as fast av_macos_xcode only different occupancy)
    dimensions = 2;         % two dimensional simulation
    tau = .1;               % time interval in seconds
    time = tau * 1:N;       % create a time vector for plotting

    k = sqrt(D * dimensions * tau);

    http://labs.physics.berkeley.edu/mediawiki/index.php/Simulating_Brownian_Motion
    """
    cdef int32_t n_accepted, n_rejected, i_accepted
    cdef int32_t i, x_idx, y_idx, z_idx
    cdef int32_t n_samples = int(t_max/t_step)
    cdef int32_t ng = d.shape[0]
    sigma = np.sqrt(2 * D * 3 * t_step) / dg
    cdef np.ndarray[dtype=np.float64_t, ndim=2] pos = np.zeros([n_samples, 3], dtype=np.float64)
    cdef char[:] accepted = np.zeros(n_samples, dtype=np.uint8)
    cdef double[:, :] r = np.random.normal(loc=0, scale=sigma, size=(n_samples, 3))
    slow_fact = np.sqrt(slow_fact)

    # find random point with density > 0 in density_av
    # take random points and use lookup-table to check if point is really
    # within the accessible volume
    while True:
        op = np.array(np.where(d > 0))
        # No point in AV found return (-1 indicates error)
        if op[0].shape[0] == 0:
            return (pos - (ng - 1) / 2) * dg, -1, -1, -1

        rnd_idx = np.random.randint(0, op[0].shape[0], 3)
        pos[0, 0] = op[0, rnd_idx[0]]
        pos[0, 1] = op[1, rnd_idx[1]]
        pos[0, 2] = op[2, rnd_idx[2]]

        x_idx = <int32_t>(pos[0, 0])
        y_idx = <int32_t>(pos[0, 1])
        z_idx = <int32_t>(pos[0, 2])

        if d[x_idx, y_idx, z_idx] > 0:
            break

    # MCMC-integration
    n_accepted = 1
    n_rejected = 0
    i_accepted = 0
    for i in range(n_samples):
        x_idx = <int32_t>(pos[i_accepted, 0])
        y_idx = <int32_t>(pos[i_accepted, 1])
        z_idx = <int32_t>(pos[i_accepted, 2])

        if ds[x_idx, y_idx, z_idx] > 0:
            r[i, 0] *= slow_fact
            r[i, 1] *= slow_fact
            r[i, 2] *= slow_fact

        pos[i, 0] = pos[i_accepted, 0] + r[i, 0]
        pos[i, 1] = pos[i_accepted, 1] + r[i, 1]
        pos[i, 2] = pos[i_accepted, 2] + r[i, 2]

        x_idx = <int32_t>(pos[i, 0])
        y_idx = <int32_t>(pos[i, 1])
        z_idx = <int32_t>(pos[i, 2])

        if 0 < x_idx < ng and 0 < y_idx < ng and 0 < z_idx < ng:
            if d[x_idx, y_idx, z_idx] > 0:
                i_accepted = i
                n_accepted += 1
                accepted[i] = True
            else:
                n_rejected += 1
                accepted[i] = False
    return (pos - (ng - 1) / 2) * dg, accepted, n_accepted, n_rejected


def spherePoints(int nSphere):
    # generate sphere points Returns list of 3d coordinates of points on a sphere using the
    # Golden Section Spiral algorithm.
    cdef double offset, rd, y, phi
    cdef np.ndarray[dtype=np.float32_t, ndim=2] sphere_points = np.zeros((nSphere,3), dtype=np.float32)
    cdef double inc = 3.14159265359 * (3 - sqrt(5))
    cdef int32_t k
    offset = 2.0 / (<double> nSphere)
    for k in range(nSphere):
        y = k * offset - 1.0 + (offset / 2.0)
        rd = sqrt(1 - y * y)
        phi = k * inc
        sphere_points[k, 0] = cos(phi) * rd
        sphere_points[k, 1] = y
        sphere_points[k, 2] = sin(phi) * rd
    return sphere_points


@cython.cdivision(True)
@cython.boundscheck(False)
def asa(double[:, :] r, double[:] vdw, uint32_t[:] probe_atom_indices, float[:, :] sphere_points,
        double probe=1.0, double radius = 2.5):
    """
    Returns list of accessible surface areas of the atoms, using the probe
    and atom radius to define the surface.

    Routines to calculate the Accessible Surface Area of a set of atoms.
    The algorithm is adapted from the Rose lab's chasa.py, which uses
    the dot density technique found in:

    Shrake, A., and J. A. Rupley. "Environment and Exposure to Solvent
    of Protein Atoms. Lysozyme and Insulin." JMB (1973) 79:351-371.
    """
    cdef char is_accessible
    cdef int32_t n_accessible_point, n_neighbor
    cdef int32_t i, j, k, nSphere, n_probe_atoms, n_atoms, probe_atom_index, atom_index

    cdef double aX, aY, aZ, bX, bY, bZ, dist2

    nSphere = sphere_points.shape[0]
    n_probe_atoms = probe_atom_indices.shape[0]
    n_atoms = r.shape[0]

    cdef np.ndarray[dtype=np.uint32_t, ndim=1] neighbor_indices = np.zeros(n_atoms, dtype=np.uint32)
    cdef np.ndarray[dtype=np.float32_t, ndim=1] probe_atom_asa = np.zeros(n_probe_atoms, dtype=np.float32)

    # calcuate asa
    cdef double c = 4.0 * 3.14159265359 / (<double> nSphere)
    for i, probe_atom_index in enumerate(probe_atom_indices):

        # find neighbors!
        n_neighbor = 0
        aX = r[probe_atom_index, 0]
        aY = r[probe_atom_index, 1]
        aZ = r[probe_atom_index, 2]
        for atom_index in range(n_atoms):
            if probe_atom_index != atom_index:
                bX = r[atom_index, 0]
                bY = r[atom_index, 1]
                bZ = r[atom_index, 2]
                dist2 = dist3c2(aX, aY, aZ, bX, bY, bZ)

                if dist2 < 2 * (vdw[probe_atom_index] + probe):
                    neighbor_indices[n_neighbor] = atom_index
                    n_neighbor += 1

        # accessible?
        n_accessible_point = 0
        for j in range(nSphere):
            is_accessible = 1
            aX = sphere_points[j, 0] * radius + r[probe_atom_index, 0]
            aY = sphere_points[j, 1] * radius + r[probe_atom_index, 1]
            aZ = sphere_points[j, 2] * radius + r[probe_atom_index, 2]

            for k in range(n_neighbor):
                bX = r[neighbor_indices[k], 0]
                bY = r[neighbor_indices[k], 1]
                bZ = r[neighbor_indices[k], 2]
                dist2 = dist3c2(aX, aY, aZ, bX, bY, bZ)
                if dist2 < (radius + probe) * (radius + probe):
                    is_accessible = 0
                    break
            if is_accessible > 0:
                n_accessible_point += 1

        probe_atom_asa[i] += c * n_accessible_point * vdw[probe_atom_index]**2
    return probe_atom_asa

