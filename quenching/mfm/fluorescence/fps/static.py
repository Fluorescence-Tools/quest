import ctypes as C
import platform
import os

import numpy as np

from mfm.fluorescence.fps import functions


b, o = platform.architecture()
package_directory = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(package_directory, './dll')


if 'Windows' in o:
    if '32' in b:
        fpslibrary = 'fpsnative.win32.dll' # os.path.join(package_directory, './dll/fpsnative.win32.dll')
    elif '64' in b:
        fpslibrary = 'fpsnative.win64.dll' # os.path.join(package_directory, './dll/fpsnative.win64.dll')
else:
    if platform.system() == 'Linux':
        fpslibrary = 'liblinux_fps.so' # os.path.join(package_directory, './dll/liblinux_fps.so')
    else:
        fpslibrary = 'libav.dylib' # os.path.join(package_directory, './dll/libav.dylib')

try:
    fps_dll_file = os.path.join(path, fpslibrary)
    _fps = np.ctypeslib.load_library(fps_dll_file, './dll')
    _fps.calculate1R.restype = C.c_int
    _fps.calculate1R.argtypes = [C.c_double, C.c_double, C.c_double,
                                 C.c_int, C.c_double,
                                 C.POINTER(C.c_double), C.POINTER(C.c_double), C.POINTER(C.c_double),
                                 C.POINTER(C.c_double), C.c_int, C.c_double,
                                 C.c_double, C.c_int,
                                 C.POINTER(C.c_char)]
    _fps.calculate3R.argtypes = [C.c_double, C.c_double, C.c_double, C.c_double, C.c_double,
                                 C.c_int, C.c_double,
                                 C.POINTER(C.c_double), C.POINTER(C.c_double), C.POINTER(C.c_double),
                                 C.POINTER(C.c_double), C.c_int, C.c_double,
                                 C.c_double, C.c_int,
                                 C.POINTER(C.c_char)]
except OSError:
    _fps = None
    print "Operating system not supported."


def calculate_1_radius(l, w, r, atom_i, x, y, z, vdw, linkersphere=0.5, linknodes=3, vdwRMax=1.8, dg=0.5, verbose=False):
    """
    :param l: float
        linker length
    :param w: float
        linker width
    :param r: float
        dye-radius
    :param atom_i: int
        attachment-atom index
    :param x: array
        Cartesian coordinates of atoms (x)
    :param y: array
        Cartesian coordinates of atoms (y)
    :param z: array
        Cartesian coordinates of atoms (z)
    :param vdw:
        Van der Waals radii (same length as number of atoms)
    :param linkersphere: float
        Initial linker-sphere to start search of allowed dye positions
    :param linknodes: int
        By default 3
    :param vdwRMax: float
        Maximal Van der Waals radius
    :param dg: float
        Resolution of accessible volume in Angstrom
    :param verbose: bool
        If true informative output is printed on std-out

    Examples
    --------
    Calculating accessible volume using provided pdb-file

    >>> import mfm
    >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
    >>> residue_number = 18
    >>> atom_name = 'CB'
    >>> attachment_atom = 1
    >>> av = mfm.fps.AV(pdb_filename, attachment_atom=1, verbose=True)

    Calculating accessible volume using provided Structure object

    >>> import mfm
    >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
    >>> structure = mfm.Structure(pdb_filename)
    >>> av = mfm.fps.AV(structure, attachment_atom=1, verbose=True)
    Calculating accessible volume
    -----------------------------
    Loading PDB
    Calculating initial-AV
    Linker-length  : 20.00
    Linker-width   : 0.50
    Linker-radius  : 5.00
    Attachment-atom: 1
    AV-resolution  : 0.50
    AV: calculate1R
    Number of atoms: 2647
    Attachment atom: [ 33.28   58.678  40.397]
    Points in AV: 111911
    Points in total-AV: 111911

    Using residue_seq_number and atom_name to calculate accessible volume, this also works without
    chain_identifier. However, only if a single-chain is present.

    >>> import mfm
    >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
    >>> structure = mfm.Structure(pdb_filename)
    >>> av = mfm.fps.AV(structure, residue_seq_number=11, atom_name='CB', verbose=True)

    If save_av is True the calculated accessible volume is save to disk. The filename of the calculated
    accessible volume is determined by output_file

    >>> import mfm
    >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
    >>> structure = mfm.Structure(pdb_filename)
    >>> av = mfm.fps.AV(structure, residue_seq_number=11, atom_name='CB', verbose=True, save_av=True, output_file='test')
    Calculating accessible volume
    -----------------------------
    Loading PDB
    Calculating initial-AV
    Linker-length  : 20.00
    Linker-width   : 0.50
    Linker-radius  : 5.00
    Attachment-atom: 174
    AV-resolution  : 0.50
    AV: calculate1R
    Number of atoms: 2647
    Attachment atom: [ 41.606  44.953  36.625]
    Points in AV: 22212
    Points in total-AV: 22212

    write_xyz
    ---------
    Filename: test.xyz
    -------------------

    """
    if verbose:
        print("AV: calculate1R")
    n_atoms = len(vdw)

    npm = int(np.floor(l / dg))
    ng = 2 * npm + 1
    ng3 = ng * ng * ng
    density = np.zeros(ng3, dtype=np.uint8)
    x0, y0, z0 = x[atom_i], y[atom_i], z[atom_i]
    r0 = np.array([x0, y0, z0])

    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.ctypes.html
    # Be careful using the ctypes attribute - especially on temporary arrays or arrays
    # constructed on the fly. For example, calling (a+b).ctypes.data_as(ctypes.c_void_p)
    # returns a pointer to memory that is invalid because the array created as (a+b) is
    # deallocated before the next Python statement.

    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)
    z = np.ascontiguousarray(z)
    vdw = np.ascontiguousarray(vdw)

    _x = x.ctypes.data_as(C.POINTER(C.c_double))
    _y = y.ctypes.data_as(C.POINTER(C.c_double))
    _z = z.ctypes.data_as(C.POINTER(C.c_double))
    _vdw = vdw.ctypes.data_as(C.POINTER(C.c_double))

    _density = density.ctypes.data_as(C.POINTER(C.c_char))
    n = _fps.calculate1R(l, w, r, atom_i, dg, _x, _y, _z, _vdw,
                         n_atoms, vdwRMax, linkersphere, linknodes, _density)
    if verbose:
        print("Number of atoms: %i" % n_atoms)
        print("Attachment atom coordinates: %s" % r0)
        print("Points in AV: %i" % n)

    density = density.astype(np.float32)
    points = functions.density2points(n, npm, dg, density, r0, ng)
    density = density.reshape([ng, ng, ng])
    return points, density, ng, r0


def calculate_3_radius(l, w, r1, r2, r3, atom_i, x, y, z, vdw, linkersphere=0.5, linknodes=3,
                       vdwRMax=1.8, dg=0.4, verbose=False):
    """
    :param l: float
        linker length
    :param w: float
        linker width
    :param r1: float
        Dye-radius 1
    :param r2: float
        Dye-radius 2
    :param r3: float
        Dye-radius 3
    :param atom_i: int
        attachment-atom index
    :param x: array
        Cartesian coordinates of atoms (x)
    :param y: array
        Cartesian coordinates of atoms (y)
    :param z: array
        Cartesian coordinates of atoms (z)
    :param vdw:
        Van der Waals radii (same length as number of atoms)
    :param linkersphere: float
        Initial linker-sphere to start search of allowed dye positions
    :param linknodes: int
        By default 3
    :param vdwRMax: float
        Maximal Van der Waals radius
    :param dg: float
        Resolution of accessible volume in Angstrom
    :param verbose: bool
        If true informative output is printed on std-out
    :return:

    """
    if verbose:
        print("AV: calculate1R")
    n_atoms = len(vdw)

    npm = int(np.floor(l / dg))
    ng = 2 * npm + 1
    ng3 = ng * ng * ng
    density = np.zeros(ng3, dtype=np.uint8)
    x0, y0, z0 = x[atom_i], y[atom_i], z[atom_i]
    r0 = np.array([x0, y0, z0])

    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)
    z = np.ascontiguousarray(z)
    vdw = np.ascontiguousarray(vdw)

    _x = x.ctypes.data_as(C.POINTER(C.c_double))
    _y = y.ctypes.data_as(C.POINTER(C.c_double))
    _z = z.ctypes.data_as(C.POINTER(C.c_double))
    _vdw = vdw.ctypes.data_as(C.POINTER(C.c_double))

    _density = density.ctypes.data_as(C.POINTER(C.c_char))
    n = _fps.calculate3R(l, w, r1, r2, r3, atom_i, dg, _x, _y, _z, _vdw,
                         n_atoms, vdwRMax, linkersphere, linknodes, _density)
    if verbose:
        print("Number of atoms: %i" % n_atoms)
        print("Attachment atom: %s" % r0)
        print("Points in AV: %i" % n)

    density = density.astype(np.float32)
    points = functions.density2points(n, npm, dg, density, r0, ng)
    density = density.reshape([ng, ng, ng])
    return points, density, ng, r0