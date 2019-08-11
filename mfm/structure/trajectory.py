import os
import copy
import tempfile

import mdtraj
import numpy as np

import mfm
from mfm.curve import Genealogy
from structure import Structure, average


class Universe(object):
    def __init__(self, structure=None):
        self.structures = [] if structure is None else [structure]
        self.potentials = []
        self.scaling = []
        self.Es = []

    def addPotential(self, potential, scale=1.0):
        print("addPotential")
        self.potentials.append(potential)
        self.scaling.append(scale)

    def removePotential(self, potentialNbr=None):
        print("removePotential")
        if potentialNbr == -1:
            self.potentials.pop()
            self.scaling.pop()
        else:
            self.potentials.pop(potentialNbr)
            self.scaling.pop(potentialNbr)

    def clearPotentials(self):
        self.potentials = []
        self.scaling = []

    def getEnergy(self, structure=None):
        if isinstance(structure, Structure):
            for p in self.potentials:
                p.structure = structure
        Es = self.getEnergies()
        E = Es.sum()
        self.E = E
        if E < -10000:
            print(Es)
        return E

    def getEnergies(self, structure=None):
        if isinstance(structure, Structure):
            for p in self.potentials:
                p.structure = structure

        scales = np.array(self.scaling)
        Es = np.array([pot.getEnergy() for pot in self.potentials])
        self.Es = np.dot(scales, Es)
        return Es


class TrajectoryFile(Genealogy, mdtraj.Trajectory):

    """
    Creates an Trajectory of Structures given a HDF5-File using mdtraj.HDF5TrajectoryFile

    Parameters
    ----------

    structure : string / Structure
        determines the topology
        is either a string containing the filename of a PDB-File or an instance of Structure()
        Obligatory in write mode, not needed in reading mode

    filename_hdf : string
        the filename of the HDF5-file

    Other Parameters
    ----------------
    verbose : bool

    stride : int, default=None
        Only read every stride-th frame.

    frame : integer or None
        If frame is an integer only the frame number provided by the integer is loaded otherwise the whole trajectory
        is loaded.

    See Also
    --------

    mfm.Structure

    Examples
    --------

    Making new h5-Trajectory file

    >>> import mfm
    >>> s = mfm.Structure('/sample_data/structure/T4L_Topology.pdb', verbose=True, make_coarse=False)
    >>> t = mfm.TrajectoryFile('test.h5', s, mode='w')
    >>> t[0]
    <mfm.structure.structure.Structure at 0x11f34e10>
    >>> print t[0]
    ATOM   2621  HB2 LEU   164      -0.380   1.259  -1.178  0.00  0.00             H
    ATOM   2622  HB3 LEU   164      -0.418   1.377  -1.295  0.00  0.00             H
    ATOM   2623   CG LEU   164      -0.497   1.413  -1.099  0.00  0.00             C
    ATOM   2624   HG LEU   164      -0.577   1.479  -1.133  0.00  0.00             H
    ATOM   2625  CD1 LEU   164      -0.558   1.332  -0.991  0.00  0.00             C
    ATOM   2626 HD11 LEU   164      -0.589   1.399  -0.910  0.00  0.00             H
    ATOM   2627 HD12 LEU   164      -0.654   1.294  -1.026  0.00  0.00             H
    ATOM   2628 HD13 LEU   164      -0.481   1.261  -0.961  0.00  0.00             H
    ATOM   2629  CD2 LEU   164      -0.380   1.493  -1.044  0.00  0.00             C
    ATOM   2630 HD21 LEU   164      -0.307   1.435  -0.987  0.00  0.00             H
    ATOM   2631 HD22 LEU   164      -0.336   1.535  -1.135  0.00  0.00             H
    ATOM   2632 HD23 LEU   164      -0.417   1.578  -0.986  0.00  0.00             H
    ATOM   2633    C LEU   164      -0.703   1.322  -1.307  0.00  0.00             C
    ATOM   2634    O LEU   164      -0.699   1.418  -1.389  0.00  0.00             O
    ATOM   2635  OXT LEU   164      -0.814   1.287  -1.254  0.00  0.00             O

    Opening h5-Trajectory file

    >>> import mfm
    >>> t = mfm.TrajectoryFile('./sample_data/structure/2807_8_9_b.h5', mode='r', stride=1)
    >>> print t[0:3]
    [<mfm.structure.structure.Structure at 0x1345d5d0>,
    <mfm.structure.structure.Structure at 0x1345d610>,
    <mfm.structure.structure.Structure at 0x132d2230>]

    Name of the trajectory

    >>> print t.name
    '/ sample_data/ structure/ sample_data/ structure/ T4L_Trajectory.h5'

    initialize with mdtraj.Trajectory

    >>> import mfm
    >>> t = mfm.TrajectoryFile('./sample_data/structure/2807_8_9_b.h5', mode='r', stride=1)
    >>> t2 = mfm.TrajectoryFile(t.mdtraj, filename='test.h5')

    Attributes:
    -----------
    rmsd : array/list containing the rmsd vs the reference structure of -new- / added structures upon addition
    of the strucutre

    """

    parameterNames = ['rmsd', 'drmsd', 'energy', 'chi2']

    def __init__(self, p_object, structure=None, **kwargs):
        frame = kwargs.get('frame', None)
        self.mode = kwargs.get('mode', 'r')
        self.atom_indices = kwargs.get('atom_indices', None)
        self.make_coarse = kwargs.get('make_coarse', False)
        self.stride = kwargs.get('stride', 1)
        self._rmsd_ref_state = kwargs.get('rmsd_ref_state', 0)
        self.verbose = kwargs.get('stride', True)
        self.center = kwargs.get('center', False)
        self._invert = kwargs.get('inverse_trajectory', False)

        pdb_tmp = tempfile.mktemp(".pdb")
        if isinstance(p_object, str):
            if p_object.endswith('.pdb'):
                self.structure = Structure(p_object, self.make_coarse, self.verbose)
                self._filename = kwargs.get('filename', tempfile.mktemp(".h5"))
                if frame is None:
                    self._mdtraj = mdtraj.Trajectory.load(p_object)
                elif isinstance(frame, int):
                    self._mdtraj = mdtraj.load_frame(p_object, frame)
            elif p_object.endswith('.h5'):
                self._filename = p_object
                if self.mode == 'r':
                    self._mdtraj = mdtraj.Trajectory.load(p_object, stride=self.stride)
                    mdtraj.Trajectory.__init__(self, self._mdtraj.xyz, self._mdtraj.topology)
                    self._mdtraj[0].save_pdb(pdb_tmp)
                    self.structure = Structure(pdb_tmp, self.make_coarse, self.verbose)
                elif self.mode == 'w':
                    structure.write(pdb_tmp)
                    self._mdtraj = mdtraj.Trajectory.load(pdb_tmp)
                    self._mdtraj.save_hdf5(p_object)
                    self._filename = p_object
                    self.structure = structure
                    mdtraj.Trajectory.__init__(self, self._mdtraj.xyz, self._mdtraj.topology)
        elif isinstance(p_object, mdtraj.Trajectory):
            self._mdtraj = p_object
            mdtraj.Trajectory.__init__(self, self._mdtraj.xyz, self._mdtraj.topology)
            self._mdtraj[0].save_pdb(pdb_tmp)
            self.structure = Structure(pdb_tmp, self.make_coarse, self.verbose)

        elif isinstance(p_object, mdtraj.Trajectory):
            self._mdtraj = p_object
            self._mdtraj[0].save_pdb(pdb_tmp)
            self.structure = Structure(pdb_tmp, False, self.verbose)
            mdtraj.Trajectory.__init__(self, xyz=p_object.xyz, topology=p_object.topology)
            self._filename = kwargs.get('filename', tempfile.mktemp(".h5"))
        elif isinstance(p_object, Structure):
            self.structure = p_object
            self.structure.write(pdb_tmp)
            self._mdtraj = mdtraj.Trajectory.load(pdb_tmp)
            mdtraj.Trajectory.__init__(self, xyz=self._mdtraj.xyz, topology=self._mdtraj.topology)
            self._filename = kwargs.get('filename', tempfile.mktemp(".h5"))

        if self.center:
            self._mdtraj.center_coordinates()

        Genealogy.__init__(self)

        self.rmsd_ref_state = kwargs.get('ref_state', 0)
        self.rmsd = []
        self.drmsd = []
        self.energy = []
        self.chi2r = []
        self.offset = 0

    def clear(self):
        self.rmsd = []
        self.drmsd = []
        self.energy = []
        self.chi2r = []
        self.rmsd_ref_state = 0

    '''
    @property
    def xyz(self):
        """Cartesian coordinates of each atom in each simulation frame

        If the attribute :py:attribute:`.TrajectoryFile.invert` is True the oder of the trajectory is inverted
        """
        if self.invert:
            return self._xyz[::-1]
        else:
            return self._xyz

    @xyz.setter
    def xyz(self, v):
        self._xyz = v
    '''

    @property
    def invert(self):
        """
        If True the oder of the trajectory is inverted (by default False)
        """
        return self._invert

    @invert.setter
    def invert(self, v):
        self._invert = bool(v)

    @property
    def filename(self):
        """
        The filename of the trajectory
        """
        return self._filename

    @filename.setter
    def filename(self, v):
        if isinstance(v, str):
            self._filename = v
            self.save(filename=v)

    @property
    def mdtraj(self):
        return self._mdtraj

    @property
    def name(self):
        """
        The name of the trajectory composed of the directory and the filename
        """
        try:
            fn = copy(self.directory + self.filename)
            return fn.replace('/', '/ ')
        except AttributeError:
            return "None"

    @property
    def rmsd_ref_state(self):
        """
        The index (frame number) of the reference state used for the RMSD calculation
        :return:
        """
        return self._rmsd_ref_state

    @rmsd_ref_state.setter
    def rmsd_ref_state(self, ref_frame):
        self._rmsd_ref_state = ref_frame
        self.rmsd = mdtraj.rmsd(self, self, ref_frame)

    @property
    def directory(self):
        """
        Directory in which the filename of the trajectory is located in
        """
        return os.path.dirname(self.filename)

    @property
    def reference(self):
        """
        The reference structure used for RMSD-calculation. This cannot be set directly but has to be set via the number
        of the refernece state :py:attribute`.rmsd_ref_state`
        """
        if self.rmsd_ref_state == 'average':
            return self.average
        else:
            return self[int(self.rmsd_ref_state)]

    @property
    def average(self):
        """
        The average structure (:py:class:`~mfm.structure.Structure`) of the trajectory
        """
        return average(self[:len(self)])

    @property
    def values(self):
        """A 2D-numpy array containing the RMSD, dRMSD, energy and the chi2 values of the trajectory

        Examples
        --------

        >>> import mfm
        >>> t = mfm.TrajectoryFile('./sample_data/structure/2807_8_9_b.h5', mode='r', stride=1)
        >>> t
        <mdtraj.Trajectory with 92 frames, 2495 atoms, 164 residues, without unitcells at 0x117f3b70>
        >>> t.values
        array([], shape=(4, 0), dtype=float64)
        >>> t.append(t[0])
        inf     inf     0.0000  0.6678
        >>> t.append(t[0])
        inf     inf     0.0000  0.0000
        >>> t.values
        array([ [  5.96507968e-09,   5.96508192e-09],
                [  6.67834069e-01,   5.96507944e-09],
                [             inf,              inf],
                [             inf,              inf]])
        """
        rmsd = np.array(self.rmsd)
        drmsd = np.array(self.drmsd)
        energy = np.array(self.energy)
        chi2 = np.array(self.chi2r)
        return np.vstack([rmsd, drmsd, energy, chi2])

    def append(self, xyz, update_rmsd=True, energy=np.inf, energy_fret=np.inf, verbose=True):
        """Append a structure of type :py::class`mfm.Structure` to the trajectory

        :param structure: Structure
        :param update_rmsd: bool
        :param energy: float
            Energy of the system
        :param energy_fret: float
            Energy of the FRET-potential
        :param verbose: bool
            By default True. If True energy, energy_fret, RMSD and dRMSD are printed to std-out.

        Examples
        --------

        >>> import mfm
        >>> t = mfm.TrajectoryFile('./sample_data/structure/2807_8_9_b.h5', mode='r', stride=1)
        >>> t
        <mdtraj.Trajectory with 92 frames, 2495 atoms, 164 residues, without unitcells at 0x11762b70>
        >>> t.append(t[0])
        <mdtraj.Trajectory with 93 frames, 2495 atoms, 164 residues, without unitcells at 0x11762b70>
        """
        verbose = verbose or self.verbose
        if isinstance(xyz, Structure):
            xyz = xyz.xyz

        xyz = xyz.reshape((1, xyz.shape[0], 3)) / 10.0
        # write to trajectory file
        mode = 'a' if os.path.isfile(self.filename) else 'w'
        t = mdtraj.formats.hdf5.HDF5TrajectoryFile(self.filename, mode=mode)
        t.write(xyz, time=len(t))
        t.close()

        next_rmsd = 0.0
        next_drmsd = 0.0
        if update_rmsd:
            self._xyz = np.append(self._xyz, xyz, axis=0)
            self._time = np.arange(len(self._xyz))
            if len(self) > 1:
                self.mdtraj._xyz = self._xyz
                self.mdtraj._time = self.time
                new = self.mdtraj[-1]
                previous = self.mdtraj[-2]
                next_drmsd = mdtraj.rmsd(new, previous) * 10.0
                next_rmsd = mdtraj.rmsd(new, self.mdtraj[self.rmsd_ref_state]) * 10.0

        self.drmsd.append(next_drmsd)
        self.rmsd.append(next_rmsd)
        self.energy.append(energy)
        self.chi2r.append(energy_fret)
        if verbose:
            print("%.3f\t%.3f\t%.4f\t%.4f" % (energy, energy_fret, next_rmsd, next_drmsd))

    def __iter__(self):
        """
        Implements iterator
        >>> import mfm
        >>> t = mfm.TrajectoryFile('./sample_data/structure/2807_8_9_b.h5', mode='r', stride=1)
        >>> for s in t:
        >>>     print s
        [<mfm.structure.structure.Structure object at 0x12FAE330>, <mfm.structure.structure.Structure object at 0x12FAE3B0>, <li
        b.structure.Structure.Structure object at 0x11852070>, <mfm.structure.structure.Structure object at 0x131052D0>, <mfm.st
        ructure.Structure.Structure object at 0x13195270>, <mfm.structure.structure.Structure object at 0x13228210>]
        """
        for i in range(len(self)):
            yield self[i]

    def __next__(self):
        """
        Iterate trough the trajectory. The current frame is stored in the
        trajectory property ``offset``

        Returns
        -------
        next : Structure
            Returns the next structure in the trajectory

        Example
        -------

        >>> import mfm
        >>> t = mfm.TrajectoryFile('./sample_data/structure/2807_8_9_b.h5', mode='r', stride=1)
        >>> s = str(t.next())
        >>> print s[:500]
        ATOM      1    N MET A   1       7.332 -10.706 -15.034  0.00  0.00             N
        ATOM      2    H MET A   1       7.280 -10.088 -15.830  0.00  0.00             H
        ATOM      3   H2 MET A   1       7.007 -11.615 -15.330  0.00  0.00             H
        ATOM      4   H3 MET A   1       8.267 -10.697 -14.653  0.00  0.00             H
        ATOM      5   CA MET A   1       6.341 -10.257 -14.033  0.00  0.00             C
        ATOM      6   HA MET A   1       5.441  -9.927 -14.551  0.00  0.00             H
        >>> s = str(t.next())
        >>> print s[:500]
        ATOM      1    N MET A   1      12.234  -5.443 -11.675  0.00  0.00             N
        ATOM      2    H MET A   1      12.560  -5.462 -10.719  0.00  0.00             H
        ATOM      3   H2 MET A   1      12.359  -4.507 -12.036  0.00  0.00             H
        ATOM      4   H3 MET A   1      12.767  -6.064 -12.265  0.00  0.00             H
        ATOM      5   CA MET A   1      10.824  -5.798 -11.763  0.00  0.00             C
        ATOM      6   HA MET A   1      10.490  -5.577 -12.777  0.00  0.00             H
        ATOM      7
        """
        if self.offset == len(self):
            raise StopIteration
        element = self[self.offset]
        self.offset = self.offset + 1
        return element

    def __getitem__(self, key):
        if isinstance(key, int):
            self.structure.xyz = self.mdtraj[key].xyz * 10.0
            self.structure.update()
            return self.structure
        else:
            start = 0 if key.start is None else key.start
            stop = None if key.stop is None else key.stop
            step = 1 if key.step is None else key.step
            re = []
            for i in range(start, min(stop, len(self)), step):
                s = copy.deepcopy(self.structure)
                s.atoms = np.copy(self.structure.atoms)
                s.xyz = self.mdtraj[i].xyz * 10.0
                s.update()
                re.append(s)
            return re
