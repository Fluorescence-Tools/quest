import os.path
package_directory = os.path.dirname(os.path.abspath(__file__))

import ctypes as C
import platform
import json
import os
from collections import OrderedDict
import numpy as np
from PyQt5 import QtGui, QtCore, uic, QtWidgets
from lib.fps.fps import subav, simulate_traj, RDAMeanE, RDAMean, dRmp
from lib.fps.fps import density2points, spherePoints, asa
from lib.io import PDB
from lib import Structure


b, o = platform.architecture()
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

_fps = np.ctypeslib.load_library(fpslibrary, './dll')
_fps.calculate1R.restype = C.c_int
_fps.calculate1R.argtypes = [C.c_double, C.c_double, C.c_double,
                             C.c_int, C.c_double,
                             C.POINTER(C.c_double), C.POINTER(C.c_double), C.POINTER(C.c_double),
                             C.POINTER(C.c_double), C.c_int, C.c_double,
                             C.c_double, C.c_int,
                             C.POINTER(C.c_char)]


def calculate1R(l, w, r, atom_i, x, y, z, vdw, linkersphere=0.5, linknodes=3, vdwRMax=1.8, dg=0.5, verbose=False):
    """
    Parameters
    ----------
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

    points = density2points(n, npm, dg, density, r0, ng)
    density = density.reshape([ng, ng, ng])
    return points, density, ng, r0


_fps.calculate3R.argtypes = [C.c_double, C.c_double, C.c_double, C.c_double, C.c_double,
                             C.c_int, C.c_double,
                             C.POINTER(C.c_double), C.POINTER(C.c_double), C.POINTER(C.c_double),
                             C.POINTER(C.c_double), C.c_int, C.c_double,
                             C.c_double, C.c_int,
                             C.POINTER(C.c_char)]


def calculate3R(l, w, r1, r2, r3, atom_i, x, y, z, vdw,
                linkersphere=0.5, linknodes=3, vdwRMax=1.8, dg=0.4, verbose=False):
    """
    Parameters
    ----------
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

    points = density2points(n, npm, dg, density, r0, ng)
    density = density.reshape([ng, ng, ng])
    return points, density, ng, r0


def write_xyz(filename, points, verbose=False):
    """
    Writes the points as xyz-format file. The xyz-format file can be opened and displayed for instance
    in PyMol
    :param filename: string
        Filename the cartesian coordinates in points are written to as xyz-format file
    :param points:
    :param verbose:
    """
    if verbose:
        print("\nwrite_xyz")
        print("---------")
        print("Filename: %s" % filename)
    fp = open(filename, 'w')
    npoints = len(points)
    fp.write('%i\n' % npoints)
    fp.write('Name\n')
    for p in points:
        fp.write('D %.3f %.3f %.3f\n' % (p[0], p[1], p[2]))
    fp.close()
    if verbose:
        print("-------------------")


class AV(object):

    def __init__(self, structure, residue_seq_number=None, atom_name=None, chain_identifier=None,
                 linker_length=20.0, linker_width=0.5, radius1=5.0,
                 radius2=4.5, radius3=3.5, simulation_grid_resolution=0.5, save_av=False, output_file='out',
                 attachment_atom_index=None, verbose=False, allowed_sphere_radius=0.5,
                 simulation_type='AV1', position_name=None, residue_name=None):

        """
        Parameters
        ----------
        :param structure: Structure of str
            either string with path of pdb-file or pdb object
        :param residue_seq_number:
        :param atom_name:
        :param chain_identifier:
        :param linker_length:
        :param linker_width:
        :param radius1:
        :param radius2:
        :param radius3:
        :param simulation_grid_resolution:
        :param save_av:
        :param output_file:
        :param attachment_atom_index:
        :param verbose:
        :param allowed_sphere_radius:
        :param simulation_type:
        :param position_name:
        :param residue_name:
        :raise ValueError:

        Examples
        --------
        Calculating accessible volume using provided pdb-file
        >>> import lib
        >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
        >>> residue_number = 18
        >>> atom_name = 'CB'
        >>> attachment_atom = 1
        >>> av = lib.fps.AV(pdb_filename, attachment_atom=1, verbose=True)

        Calculating accessible volume using provided Structure object
        >>> import lib
        >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
        >>> structure = lib.Structure(pdb_filename)
        >>> av = lib.fps.AV(structure, attachment_atom=1, verbose=True)
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
        >>> import lib
        >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
        >>> structure = lib.Structure(pdb_filename)
        >>> av = lib.fps.AV(structure, residue_seq_number=11, atom_name='CB', verbose=True)

        If save_av is True the calculated accessible volume is save to disk. The filename of the calculated
        accessible volume is determined by output_file
        >>> import lib
        >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
        >>> structure = lib.Structure(pdb_filename)
        >>> av = lib.fps.AV(structure, residue_seq_number=11, atom_name='CB', verbose=True, save_av=True, output_file='test')
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
        self.verbose = verbose
        self.points_slow = None
        self.position_name = position_name
        self.residue_name = residue_name
        if verbose:
            print("")
            print("Calculating accessible volume")
            print("-----------------------------")
        if isinstance(structure, Structure):
            atoms = structure.atoms
        else:
            if verbose:
                print("Loading PDB")
            atoms = PDB.read(structure, verbose=verbose)
        x = atoms['coord'][:, 0]
        y = atoms['coord'][:, 1]
        z = atoms['coord'][:, 2]
        vdw = atoms['radius']
        self.structure = structure
        self.dg = simulation_grid_resolution
        self.output_file = output_file

        # Determine Labeling position
        if verbose:
            print("Labeling position")
            print("Chain ID: %s" % chain_identifier)
            print("Residue seq. number: %s" % residue_seq_number)
            print("Residue name: %s" % residue_name)
            print("Atom name: %s" % atom_name)

        if attachment_atom_index is None:
            if residue_seq_number is None or atom_name is None:
                raise ValueError("either attachment_atom number or residue and atom_name have to be provided.")
            self.attachment_residue = residue_seq_number
            self.attachment_atom = atom_name
            if chain_identifier is None:
                attachment_atom_index = np.where((atoms['res_id'] == residue_seq_number) &
                                                (atoms['atom_name'] == atom_name))[0]
            else:
                attachment_atom_index = np.where((atoms['res_id'] == residue_seq_number) &
                                                 (atoms['atom_name'] == atom_name) &
                                                 (atoms['chain'] == chain_identifier))[0]
            if len(attachment_atom_index) != 1:
                raise ValueError("Invalid selection of attachment atom")
            else:
                attachment_atom_index = attachment_atom_index[0]
            if verbose:
                print("Atom index: %s" %attachment_atom_index)

        if verbose:
            print("Calculating initial-AV")
            print("Linker-length  : %.2f" % linker_length)
            print("Linker-width   : %.2f" % linker_width)
            print("Linker-radius  : %.2f" % radius1)
            print("Attachment-atom: %i" % attachment_atom_index)
            print("AV-resolution  : %.2f" % simulation_grid_resolution)
        if simulation_type == 'AV1':
            points, density, ng, x0 = calculate1R(linker_length, linker_width, radius1, attachment_atom_index, x, y, z, vdw, verbose=verbose,
                                                  linkersphere=allowed_sphere_radius, dg=simulation_grid_resolution)
        elif simulation_type == 'AV3':
            points, density, ng, x0 = calculate3R(linker_length, linker_width, radius1, radius2, radius3,
                                                  attachment_atom_index, x, y, z, vdw, dg=simulation_grid_resolution, verbose=verbose,
                                                  linkersphere=allowed_sphere_radius)
        self.ng = ng
        self.points = points
        self.points_fast = points
        self.density_fast = density
        self.density_slow = None
        self.density = density
        self.x0 = x0
        if verbose:
            print("Points in total-AV: %i" % density.sum())
        if save_av:
            write_xyz(output_file+'.xyz', points, verbose=verbose)

    def calc_slow_av(self, slow_centers=None, slow_radius=None, save=True, verbose=True, replace_density=False):
        """
        Calculates a subav given the accessible volume determined during class initiation.
        : list/numpy array of points dimension (npoints, 3)


        Parameters
        ----------
        :param slow_centers : array_like
             A list of Cartesian coordinated representing the slow centers. The shape should be (n,3)
             where n is the number of slow centers.
        :param slow_radius : array_like, or float
            A list of radii the shape
        :param replace_density : bool
            If replace_density is True the AV-density will be replaced by the density of the slow
            accessible volume
        :param verbose : bool
        :param save : bool
            If save is True the slow-accessible volume is saved to an xyz-file with the suffix '_slow'

        """
        verbose = verbose or self.verbose
        if verbose:
            print("Calculating slow-AV:")
            print(slow_centers)
        if isinstance(slow_radius, (int, long, float)):
            slow_radii = np.ones(slow_centers.shape[0])*slow_radius
        elif len(slow_radius) != slow_centers.shape[0]:
            raise ValueError("The size of the slow_radius doesnt match the number of slow_centers")

        if verbose or self.verbose:
            print("Calculating slow AV")
            print("slow centers: \n%s" % slow_centers)

        ng = self.ng
        npm = (ng - 1) / 2
        density = self.density_fast
        dg = self.dg
        x0 = self.x0
        slow_density = subav(density, ng, dg, slow_radii, slow_centers, x0)
        n = slow_density.sum()
        points = fps.density2points(n, npm, dg, slow_density.reshape(ng*ng*ng), x0, ng)
        if verbose or self.verbose:
            print("Points in slow-AV: %i" % n)
        if save:
            write_xyz(self.output_file+'_slow.xyz', points, verbose=verbose)
        self.density_slow = slow_density
        self.points_slow = points
        if replace_density:
            self.density = slow_density

    @property
    def Rmp(self):
        return self.points.mean(axis=0)

    def dRmp(self, av):
        """
        Calculate the distance between the mean positions with respect to the accessible volume `av`

        Parameters
        ----------
        av : accessible volume object

        Example
        -------
        >>> import lib
        >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
        >>> structure = lib.Structure(pdb_filename)
        >>> av1 = lib.fps.AV(structure, residue_seq_number=72, atom_name='CB')
        >>> av2 = lib.fps.AV(structure, residue_seq_number=134, atom_name='CB')
        >>> av1.dRmp(av2)
        """
        return dRmp(self, av)

    def dRDA(self, av):
        """
        Calculate the mean distance to the second accessible volume `av`

        Parameters
        ----------
        av : accessible volume object

        Example
        -------
        >>> import lib
        >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
        >>> structure = lib.Structure(pdb_filename)
        >>> av1 = lib.fps.AV(structure, residue_seq_number=72, atom_name='CB')
        >>> av2 = lib.fps.AV(structure, residue_seq_number=134, atom_name='CB')
        >>> av1.dRDA(av2)
        """
        return RDAMean(self, av)


    def dRDAE(self, av):
        """
        Calculate the FRET-averaged mean distance to the second accessible volume `av`

        Parameters
        ----------
        av : accessible volume object

        Example
        -------
        >>> import lib
        >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
        >>> structure = lib.Structure(pdb_filename)
        >>> av1 = lib.fps.AV(structure, residue_seq_number=72, atom_name='CB')
        >>> av2 = lib.fps.AV(structure, residue_seq_number=134, atom_name='CB')
        >>> av1.dRDAE(av2)
        """
        return RDAMeanE(self, av)

    def __repr__(self):
        s = '\n'
        s += 'Accessible Volume\n'
        s += '-----------------\n'
        s += 'Attachment residue: %s\n' % self.attachment_residue
        s += 'Attachment atom: %s\n' % self.attachment_atom
        s += '\n'
        s += 'Fast-AV:\n'
        s += 'n-points: %s\n' % self.points_fast.shape[0]
        if self.points_slow is not None:
            s += 'Slow-AV:\n'
            s += 'n-points: %s\n' % self.points_slow.shape[0]
        else:
            s += 'Slow-AV: not calculated'
        return s


class AvPotential(object):

    name = 'Av'

    def __init__(self, distances=None, positions=None, av_samples=10000, min_av=150, verbose=False):
        """
        parameters should be a dictionary determining the
         system (labeling positions etc.)
        >>> import json
        >>> labeling_file = './sample_data/model/labeling.json'
        >>> labeling = json.load(open(labeling_file, 'r'))
        >>> distances = labeling['Distances']
        >>> positions = labeling['Positions']
        >>> import lib
        >>> av_potential = lib.fps.AvPotential(distances=distances, positions=positions)
        >>> structure = lib.Structure('/sample_data/model/HM_1FN5_Naming.pdb')
        >>> av_potential.getChi2(structure)
        """
        self.verbose = verbose
        self.distances = distances
        self.positions = positions
        self.n_av_samples = av_samples
        self.min_av = min_av
        self.avs = OrderedDict()

    @property
    def structure(self):
        return self._structure

    @structure.setter
    def structure(self, structure):
        self._structure = structure
        self.calc_avs()

    def calc_avs(self):
        if self.positions is None:
            raise ValueError("Positions not set unable to calculate AVs")
        arguments = [
            dict(
                {'structure': self.structure,
                'verbose': self.verbose,
                },
                **self.positions[position_key]
            )
            for position_key in self.positions
        ]
        avs = map(lambda x: AV(**x), arguments)
        for i, position_key in enumerate(self.positions):
            self.avs[position_key] = avs[i]

    def calc_distances(self, structure=None, verbose=False):
        verbose = verbose or self.verbose
        if isinstance(structure, Structure):
            self.structure = structure
        for distance_key in self.distances:
            distance = self.distances[distance_key]
            av1 = self.avs[distance['position1_name']]
            av2 = self.avs[distance['position2_name']]
            distance_type = distance['distance_type']
            R0 = distance['Forster_radius']
            if distance_type == 'RDAMean':
                d12 = RDAMean(av1, av2, self.n_av_samples)
            elif distance_type == 'Rmp':
                d12 = dRmp(av1, av2)
            elif distance_type == 'RDAMeanE':
                d12 = RDAMeanE(av1, av2, R0, self.n_av_samples)
            distance['model_distance'] = d12
            if verbose:
                print("-------------")
                print("Distance: %s" % distance_key)
                print("Forster-Radius %.1f" % distance['Forster_radius'])
                print("Distance type: %s" % distance_type)
                print("Model distance: %.1f" % d12)
                print("Experimental distance: %.1f (-%.1f, +%.1f)" % (distance['distance'],
                                                                      distance['error_neg'], distance['error_pos']))

    def getChi2(self, structure=None, reduced=False, verbose=False):
        verbose = self.verbose or verbose
        if isinstance(structure, Structure):
            self.structure = structure

        chi2 = 0.0
        self.calc_distances(verbose=verbose)
        for distance in list(self.distances.values()):
            dm = distance['model_distance']
            de = distance['distance']
            error_neg = distance['error_neg']
            error_pos = distance['error_pos']
            d = dm - de
            chi2 += (d / error_neg)**2 if d < 0 else (d / error_pos)**2
        if reduced:
            return chi2 / (len(list(self.distances.keys())) - 1.0)
        else:
            return chi2

    def getEnergy(self, structure=None):
        if isinstance(structure, Structure):
            self.structure = structure
        return self.getChi2()


class AvWidget(AvPotential, QtWidgets.QWidget):

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent=parent)
        uic.loadUi('./lib/fps/avWidget.ui', self)
        AvPotential.__init__(self)
        self._filename = None
        self.connect(self.actionOpenLabeling, QtCore.SIGNAL("triggered()"), self.onLoadAvJSON)

    def onLoadAvJSON(self):
        self.filename = str(QtWidgets.QFileDialog.getOpenFileName(None, 'Open FRET-JSON', '', 'link file (*.fps.json)')[0])

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, v):
        if os.path.isfile(v):
            self._filename = v
            p = json.load(open(v))
            self.distances = p["Distances"]
            self.positions = p["Positions"]
        else:
            v = 'None'
        self.lineEdit_2.setText(v)

    @property
    def n_av_samples(self):
        return int(self.spinBox_2.value())

    @n_av_samples.setter
    def n_av_samples(self, v):
        return self.spinBox_2.setValue(int(v))

    @property
    def min_av(self):
        return int(self.spinBox_2.value())

    @min_av.setter
    def min_av(self, v):
        return self.spinBox.setValue(int(v))

