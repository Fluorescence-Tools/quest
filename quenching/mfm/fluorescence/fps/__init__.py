import os
from collections import OrderedDict
import json

import mfm
import numpy as np
from PyQt4 import QtCore, QtGui, uic

from .static import calculate_1_radius, calculate_3_radius
from . import functions
import mfm.io.pdb as pdb
from mfm import Structure
import static
import dynamic


package_directory = os.path.dirname(__file__)
dye_file = './settings/dye_definition.json'
try:
    dye_definition = json.load(open(dye_file))
except IOError:
    dye_definition = dict()
    dye_definition['a'] = 0
dye_names = dye_definition.keys()


class AV(object):
    """

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

    def __init__(self, structure, residue_seq_number=None, atom_name=None, chain_identifier=None,
                 linker_length=20.0, linker_width=0.5, radius1=5.0,
                 radius2=4.5, radius3=3.5, simulation_grid_resolution=0.5, save_av=False, output_file='out',
                 attachment_atom_index=None, allowed_sphere_radius=0.5,
                 simulation_type='AV1', position_name=None, residue_name=None, **kwargs):
        """

        :param structure: :class:`mfm.structure.structure` or str
            Structure object of filename pointing to PDB-File
        :param residue_seq_number: int
            Attachment residue of the accessible volume
        :param atom_name: string
            Atom name where the dye is attached to
        :param chain_identifier: string
            Chain identifier
        :param linker_length: float
            Length of the dye linker
        :param linker_width: float
            Width of the dye linker
        :param radius1: float
            Radius of the dye
        :param radius2: float
            Radius of the dye
        :param radius3: float
            Radius of the dye
        :param simulation_grid_resolution: float
            Resolution of the simulation grid
        :param save_av: bool
            If True than the accessible volume is saved after performing the simulation
        :param output_file: string
            Filename used to save the accessible volume
        :param attachment_atom_index: int
            Optional parameter. If an attachment atom index is provided the parameters atom_name, chain_identifier,
            and residue_seq_number are ignored to look-up the atom
        :param verbose: bool
            If True an output to the stdout is generated
        :param allowed_sphere_radius: float
        :param simulation_type:
        :param position_name:
        :param residue_name:
        :return:

        """
        self.verbose = kwargs.get('verbose', mfm.verbose)
        if isinstance(structure, Structure):
            self.structure = structure
        elif isinstance(structure, str):
            self.structure = Structure(structure, verbose=self.verbose)

        self.points_slow = None
        self.position_name = position_name
        self.residue_name = residue_name
        self.dg = simulation_grid_resolution
        self.output_file = output_file
        self.attachment_residue = residue_seq_number
        self.attachment_atom = atom_name

        if isinstance(self.atoms, np.ndarray):
            if attachment_atom_index is None:
                if residue_seq_number is None or atom_name is None:
                    raise ValueError("either attachment_atom number or residue and atom_name have to be provided.")
            if attachment_atom_index is None:
                attachment_atom_index = pdb.get_attachment_atom_index(self.atoms, chain_identifier, residue_seq_number,
                                                                      atom_name, residue_name)
            if self.verbose:
                print("Calculating accessible volume")
                print("-----------------------------")

            x = self.atoms['coord'][:, 0]
            y = self.atoms['coord'][:, 1]
            z = self.atoms['coord'][:, 2]
            vdw = self.atoms['radius']

            if self.verbose:
                print("Calculating initial-AV")
                print("Linker-length  : %.2f" % linker_length)
                print("Linker-width   : %.2f" % linker_width)
                print("Linker-radius  : %.2f" % radius1)
                print("Attachment-atom: %i" % attachment_atom_index)
                print("AV-resolution  : %.2f" % simulation_grid_resolution)

            if simulation_type == 'AV1':
                points, density, ng, x0 = calculate_1_radius(linker_length, linker_width, radius1, attachment_atom_index, x, y, z,
                                                      vdw, verbose=self.verbose,
                                                      linkersphere=allowed_sphere_radius, dg=simulation_grid_resolution)
            elif simulation_type == 'AV3':
                points, density, ng, x0 = calculate_3_radius(linker_length, linker_width, radius1, radius2, radius3,
                                                      attachment_atom_index, x, y, z, vdw, dg=simulation_grid_resolution,
                                                      verbose=self.verbose,
                                                      linkersphere=allowed_sphere_radius)
            self.points = points
            self.points_fast = points
            self.density_slow = np.zeros((0, 0, 0), dtype=np.uint8)
            self._density = density
            self.x0 = x0
            if self.verbose:
                print("Points in total-AV: %i" % density.sum())
            if save_av:
                pdb.write_xyz(output_file + '.xyz', points, verbose=self.verbose)

    def __str__(self):
        s = "\nAccessible volume\n"
        s += "-----------------\n"
        s += "\n"
        s += "x0: %s\n" % self.x0
        s += "Shape slow (x,y,z): %s %s %s\n" % self.density_slow.shape
        s += "Shape fast (x,y,z): %s %s %s\n" % self.density.shape
        return s

    @property
    def atoms(self):
        return self.structure.atoms

    @property
    def ng(self):
        return self.density.shape[0]

    @property
    def density(self):
        return self._density

    @density.setter
    def density(self, v):
        self._density = v

    def save(self, filename, mode='xyz'):
        """Saves the accessible volume as xyz-file
        """
        if mode == 'xyz':
            with open(filename+'_fast.xyz', 'w') as fp:
                points = self.points_fast
                npoints = len(points)
                fp.write('%i\n' % npoints)
                fp.write('Name\n')
                for p in points:
                    fp.write('D %.3f %.3f %.3f\n' % (p[0], p[1], p[2]))
                fp.close()
            with open(filename+'_slow.xyz', 'w') as fp:
                points = self.points_slow
                npoints = len(points)
                fp.write('%i\n' % npoints)
                fp.write('Name\n')
                for p in points:
                    fp.write('D %.3f %.3f %.3f\n' % (p[0], p[1], p[2]))
                fp.close()

    def calc_slow_av(self, slow_centers=None, slow_radius=None, save=False, replace_density=False, **kwargs):
        """
        Calculates a subav given the accessible volume determined during class initiation.

        :param slow_centers: array_like
            A list of Cartesian coordinated representing the slow centers. The shape should be (n,3)
            where n is the number of slow centers.
        :param slow_radius: array_like, or float
            A list of radii the shape
        :param save: bool
            If save is True the slow-accessible volume is saved to an xyz-file with the suffix '_slow'
        :param verbose: bool
        :param replace_density: bool
            If replace_density is True the AV-density will be replaced by the density of the slow
            accessible volume
        :return:

        Examples
        --------

        Using residue_seq_number and atom_name to calculate accessible volume, this also works without
        chain_identifier. However, only if a single-chain is present.

        >>> import mfm
        >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
        >>> structure = mfm.Structure(pdb_filename)
        >>> av = mfm.fps.AV(structure, residue_seq_number=11, atom_name='CB', verbose=True)

        Now lets determine the points within the AV within a certain radius

        >>> av.calc_slow_av(slow_centers=np.array([[10, 5.0, 2.1]]), slow_radius=5.0, verbose=True)
        Calculating slow-AV:
        [[ 10.    5.    2.1]]
        Calculating slow AV
        slow centers:
        [[ 10.    5.    2.1]]
        Points in slow-AV: 0

        There are no points within a radius of 5.0. At a bigger radius will find some points.

        >>> av.calc_slow_av(slow_centers=np.array([[10, 5.0, 2.1]]), slow_radius=70.0, verbose=True)
        Calculating slow-AV:
        [[ 10.    5.    2.1]]
        Calculating slow AV
        slow centers:
        [[ 10.    5.    2.1]]
        Points in slow-AV: 2809

        """
        verbose = kwargs.get('verbose', mfm.verbose)
        density = kwargs.get('density', self.density)
        dg = kwargs.get('dg', self.dg)
        ng = kwargs.get('ng', self.ng)
        x0 = kwargs.get('ng', self.x0)
        output_file = kwargs.get('output_file', self.output_file)

        if verbose:
            print("Calculating slow-AV:")
            print(slow_centers)
        if isinstance(slow_radius, (int, long, float)):
            slow_radii = np.ones(slow_centers.shape[0]) * slow_radius
        else:
            slow_radii = np.array(slow_radius)
            if slow_radii.shape[0] != slow_centers.shape[0]:
                raise ValueError("The size of the slow_radius doesnt match the number of slow_centers")

        if verbose:
            print("Calculating slow AV")
            print("slow centers: \n%s" % slow_centers)

        npm = (ng - 1) / 2
        slow_density = functions.make_subav(density, dg, slow_radii, slow_centers, x0)

        n = slow_density.sum()
        points = functions.density2points(n, npm, dg, slow_density.reshape(ng * ng * ng), x0, ng)
        if verbose:
            print("Points in slow-AV: %i" % n)
        if save:
            pdb.write_xyz(output_file + '_slow.xyz', points, verbose=verbose)

        self.density_slow = slow_density
        self.points_slow = points
        if replace_density:
            self.density = slow_density

    @property
    def Rmp(self):
        """
        The mean position of the accessible volume (average x, y, z coordinate)
        """
        return self.points.mean_xyz(axis=0)

    def dRmp(self, av):
        """
        Calculate the distance between the mean positions with respect to the accessible volume `av`

        :param av: accessible volume object
        :return:

        Examples
        --------

        >>> import mfm
        >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
        >>> structure = mfm.Structure(pdb_filename)
        >>> av1 = mfm.fps.AV(structure, residue_seq_number=72, atom_name='CB')
        >>> av2 = mfm.fps.AV(structure, residue_seq_number=134, atom_name='CB')
        >>> av1.dRmp(av2)
        """
        return functions.dRmp(self, av)

    def dRDA(self, av):
        """
        Calculate the mean distance to the second accessible volume

        :param av:
        :return:

        Examples
        --------

        >>> import mfm
        >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
        >>> structure = mfm.Structure(pdb_filename)
        >>> av1 = mfm.fps.AV(structure, residue_seq_number=72, atom_name='CB')
        >>> av2 = mfm.fps.AV(structure, residue_seq_number=134, atom_name='CB')
        >>> av1.dRDA(av2)
        """
        return functions.RDAMean(self, av)

    def dRDAE(self, av):
        """
        Calculate the FRET-averaged mean distance to the second accessible volume

        :param av:
        :return:

        Examples
        --------

        >>> import mfm
        >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
        >>> structure = mfm.Structure(pdb_filename)
        >>> av1 = mfm.fps.AV(structure, residue_seq_number=72, atom_name='CB')
        >>> av2 = mfm.fps.AV(structure, residue_seq_number=134, atom_name='CB')
        >>> av1.dRDAE(av2)
        """
        return functions.RDAMeanE(self, av)

    def __repr__(self):
        s = '\n'
        s += 'Accessible Volume\n'
        s += '-----------------\n'
        s += 'Attachment residue: %s\n' % self.attachment_residue
        s += 'Attachment atom: %s\n' % self.attachment_atom
        s += '\n'
        s += 'Fast-AV:\n'
        s += '\tn-points: %s\n' % self.points_fast.shape[0]
        s += 'Slow-AV:\n'
        if self.points_slow is not None:
            s += '\tn-points: %s\n' % self.points_slow.shape[0]
        else:
            s += '\t!!not calculated!!\n'
        return s


class AvPotential(object):
    """
    The AvPotential class provides the possibility to calculate the reduced or unreduced chi2 given a set of
    labeling positions and experimental distances. Here the labeling positions and distances are provided as
    dictionaries.

    Examples
    --------

    >>> import json
    >>> labeling_file = './sample_data/model/labeling.json'
    >>> labeling = json.load(open(labeling_file, 'r'))
    >>> distances = labeling['Distances']
    >>> positions = labeling['Positions']
    >>> import mfm
    >>> av_potential = mfm.fps.AvPotential(distances=distances, positions=positions)
    >>> structure = mfm.Structure('/sample_data/model/HM_1FN5_Naming.pdb')
    >>> av_potential.getChi2(structure)

    """
    name = 'Av'

    def __init__(self, distances=None, positions=None, av_samples=10000, min_av=150, verbose=False):
        self.verbose = verbose
        self.distances = distances
        self.positions = positions
        self.n_av_samples = av_samples
        self.min_av = min_av
        self.avs = OrderedDict()

    @property
    def structure(self):
        """
        The Structure object used for the calculation of the accessible volumes
        """
        return self._structure

    @structure.setter
    def structure(self, structure):
        self._structure = structure
        self.calc_avs()

    @property
    def chi2(self):
        """
        The current unreduced chi2 (recalculated at each call)
        """
        return self.getChi2()

    def calc_avs(self):
        """
        Calculates/recalculates the accessible volumes.
        """
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
        """

        :param structure: Structure
            If this object is provided the attributes regarding dye-attachment are kept constant
            and the structure is changed prior calculation of the distances.
        :param verbose: bool
            If this is True output to stdout is generated
        """
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
                d12 = functions.RDAMean(av1, av2, self.n_av_samples)
            elif distance_type == 'Rmp':
                d12 = functions.dRmp(av1, av2)
            elif distance_type == 'RDAMeanE':
                d12 = functions.RDAMeanE(av1, av2, R0, self.n_av_samples)
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
        """

        :param structure: Structure
            A Structure object if provided the attributes regarding dye-attachment are kept constant
            and the structure is changed prior calculation of the distances.
        :param reduced: bool
            If True the reduced chi2 is calculated (by default False)
        :param verbose: bool
            Output to stdout
        :return: A float containig the chi2 (reduced or unreduced) of the current or provided structure.
        """
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
            chi2 += (d / error_neg) ** 2 if d < 0 else (d / error_pos) ** 2
        if reduced:
            return chi2 / (len(list(self.distances.keys())) - 1.0)
        else:
            return chi2

    def getEnergy(self, structure=None):
        if isinstance(structure, Structure):
            self.structure = structure
        return self.getChi2()


class AvWidget(AvPotential, QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent=parent)
        uic.loadUi('./mfm/ui/fluorescence/avWidget.ui', self)
        AvPotential.__init__(self)
        self._filename = None
        self.connect(self.actionOpenLabeling, QtCore.SIGNAL("triggered()"), self.onLoadAvJSON)

    def onLoadAvJSON(self):
        self.filename = str(QtGui.QFileDialog.getOpenFileName(None, 'Open FRET-JSON', '', 'link file (*.fps.json)'))

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
