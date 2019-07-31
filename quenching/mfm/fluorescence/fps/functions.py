from mfm.fluorescence.fps import _fps
import numpy as np


def RDAMean(av1, av2, nSamples=50000):
    """Calculate the mean distance between two accessible volumes

    >>> import mfm
    >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
    >>> structure = mfm.Structure(pdb_filename)
    >>> av1 = mfm.fps.AV(structure, residue_seq_number=72, atom_name='CB')
    >>> av2 = mfm.fps.AV(structure, residue_seq_number=134, atom_name='CB')
    >>> mfm.fps.RDAMean(av1, av2)
    52.93390285282142
    """
    _fps.RDAMean(av1, av2, nSamples)


def RDAMeanE(av1, av2, R0=52.0, nSamples=50000):
    """Calculate the FRET-averaged (PDA/Intensity) distance between two accessible volumes

    >>> import mfm
    >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
    >>> structure = mfm.Structure(pdb_filename)
    >>> av1 = mfm.fps.AV(structure, residue_seq_number=72, atom_name='CB')
    >>> av2 = mfm.fps.AV(structure, residue_seq_number=134, atom_name='CB')
    >>> mfm.fps.RDAMeanE(av1, av2)
    52.602731299544686
    """
    return _fps.RDAMeanE(av1, av2, R0, nSamples)


def dRmp(av1, av2):
    """Calculate the distance between the mean position of two accessible volumes

    >>> import mfm
    >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
    >>> structure = mfm.Structure(pdb_filename)
    >>> av1 = mfm.fps.AV(structure, residue_seq_number=72, atom_name='CB')
    >>> av2 = mfm.fps.AV(structure, residue_seq_number=134, atom_name='CB')
    >>> mfm.fps.dRmp(av1, av2)
    49.724995634807691
    """
    return _fps.dRmp(av1, av2)


def density2points(n, npm, dg, density, r0, ng):
    """
    :param n:
    :param npm:
    :param dg:
    :param density:
    :param r0:
    :param ng:
    :return:
    """
    return _fps.density2points(n, npm, dg, density, r0, ng)


def reset_density_av(density):
    """Sets all densities in av to 1.0 if the density is bigger than 0.0

    :param density: numpy-array
    :return:
    """
    ng = density.shape[0]
    _fps.reset_density_av(density, ng)


def random_distances(av1, av2, n_samples=10000):
    """Generates a set of random distances
    """
    return _fps.random_distances(av1, av2, n_samples)


def make_subav(density, dg, radius, rs, r0):
    """

    :param density: numpy-array
        density of the accessible volume dimension ng, ng, ng uint8 numpy array as obtained of fps
    :param ng: int
        number of grid points
    :param dg: float
        grid resolution of the density grid
    :param radius: float
        radius around the list of points. all points within a radius of r0 around the points in rs are part
        of the subav. each slow point is assiciated to one slow-radius
    :param rs: list
        list of points (x,y,z) defining the subav
    :param r0:
        is the position of the accessible volume


    Examples
    --------

    >>> import mfm
    >>> import numpy as np
    >>> av = np.
    >>> av =
    >>> mfm.fluorescence.fps.subav()
    """
    ng = density.shape[0]
    n_radii = rs.shape[0]

    radius = np.array(radius, dtype=np.float64)
    if len(radius) != n_radii:
        radius = np.zeros(n_radii, dtpye=np.float64) + radius[0]

    density = np.copy(density)
    _fps.make_subav(density, ng, dg, radius, rs, r0, n_radii)
    return density


def modify_av(density, dg, radius, rs, r0, factor):
    """
    Multiplies density by factor if within radius

    :param density: numpy-array
        density of the accessible volume dimension ng, ng, ng uint8 numpy array as obtained of fps
    :param dg: float
        grid resolution of the density grid
    :param radius: numpy-array/list
        radius around the list of points. all points within a radius of r0 around the points in rs are part
        of the subav. each slow point is associated to one slow-radius
    :param rs: numpy-array/list
        list of points (x,y,z) defining the subav
    :param r0: numpy-array
        is the position of the accessible volume
    :param factor: float
        factor by which density is multiplied
    """

    ng = density.shape[0]
    n_radii = rs.shape[0]

    radius = np.array(radius, dtype=np.float64)
    if len(radius) != n_radii:
        radius = np.zeros(n_radii, dtpye=np.float64) + radius[0]

    density = np.copy(density)

    _fps.modify_av(density, ng, dg, radius, rs, r0, n_radii, factor)
