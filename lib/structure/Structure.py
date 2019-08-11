import os
from copy import deepcopy, copy
from collections import OrderedDict
import tempfile

import numpy as np
import mdtraj
from scipy.cluster.hierarchy import fcluster

import lib
from lib.math.linalg import vector
from lib.structure import cStructure
from lib.io.PDB import Pdb

try:
    import fastcluster as hclust
except ImportError:
    import scipy.cluster.hierarchy as hclust
import os.path


clusterCriteria = ['maxclust', 'inconsistent', 'distance']


def onRMSF(structures, selectedNbrs, atomName=None, weights=None):
    """
    Calcualtes the root mean square deviation with respect to the average structure
    for a given set of structures. The structures do not have to be aligned.
    :param structures: a list of structure object of type Structure
    :param selectedNbrs: list of integers with the selected structures out of the structure list
    :param atomName: atom-name used for calcualtion (e.g. 'CA') if not specified all atoms are used
    :return:
    """
    print("onRMSF")
    if weights is None:
        print("using no weights")
        weights = np.ones(len(selectedNbrs), dtype=np.float32)
    else:
        print("using weights")
    weights /= sum(weights)
    candidateStructures = [deepcopy(structures[i]) for i in selectedNbrs]
    print("calculating average structure as a reference")
    reference = average(candidateStructures, weights=weights)
    print("aligning selected structures with respect to reference")
    for s in candidateStructures:
        super_impose(reference, s)
    print("Getting %s-atoms of reference" % atomName)
    ar = reference.getAtoms(atomName=atomName)
    cr = ar['coord']
    msf = np.zeros(len(ar), dtype=np.float32)
    for i, s in enumerate(candidateStructures):
        a = s.getAtoms(atomName=atomName)
        ca = a['coord']
        msf += weights[i] * np.sum((cr - ca) ** 2, axis=1)
    return np.sqrt(msf)


def rmsd(sa, sb, atom_indices=None):
    """
    Takes two structures and returns the rmsd-value
    >>> import lib
    >>> t = lib.TrajectoryFile('./sample_data/structure/2807_8_9_b.h5', mode='r', stride=1)
    >>> s1 = t[10]
    >>> s1
    <lib.structure.Structure.Structure at 0x135f3ad0>
    >>> s2 = t[0]
    >>> s2
    <lib.structure.Structure.Structure at 0x1348fbb0>
    >>> rmsd(s1, s2)
    6.960082250440536
    """
    if atom_indices is not None:
        a = sa.xyz[atom_indices]
        b = sb.xyz[atom_indices]
    else:
        a = sa.xyz
        b = sb.xyz
    rmsd = np.sqrt(1. / a.shape[0] * ((a - b) ** 2).sum())
    return float(rmsd)


def find_representative(trajectory, cl):
    """
    :param trajectory: a list of structures
    :param c: a list of numbers (positions in structures) belonging to one cluster
    :return: index of representative structure of cluster
    """
    structuresInCluster = [trajectory[i] for i in cl]
    averageStructureInCluster = average(structuresInCluster)
    idx, representativeStructureInCluster = find_best(averageStructureInCluster, structuresInCluster)
    idxOfRepresentativeStructure = cl[idx]
    return idxOfRepresentativeStructure


def cluster(structures, threshold=5000, criterion='maxclust', Z=None, distances=None, directory=None):
    # http://www.mathworks.de/de/help/stats/hierarchical-clustering.html
    print("Performing cluster-analysis")
    k = 0
    #start_time = time.time()
    nStructures = len(structures)
    if distances is None:
        distances = np.empty(nStructures * (nStructures - 1) / 2)
        for i in range(nStructures):
            for j in range(i + 1, nStructures):
                distances[k] = rmsd(structures[j], structures[i])
                k += 1
            m = (nStructures * nStructures - 1) / 2
            print('RMSD computation %s/%s : %.1f%%' % (k, m, float(k) / m * 100.0))
        if directory is not None:
            print("Saving distance-matrix")
            np.save(directory + '/' + 'clDistances.npy', distances)

    print('mean pairwise distance ', np.mean(distances))
    print('stddev pairwise distance', np.std(distances))

    if Z is None:
        # run hierarchical clustering on the distance matrix
        print('\n\nRunning hierarchical clustering (UPGMA)...')
        Z = hclust.linkage(distances, method='average', preserve_input=True)
        # get flat clusters from the linkage matrix corresponding to states
        if directory is not None:
            print("Saving cluster-results")
            np.save(directory + '/' + 'clLinkage.npy', Z)

    print('\n\nFlattening the clusters...')
    assignments = fcluster(Z, t=threshold, criterion=criterion)
    cl = dict()
    for c in np.unique(assignments):
        cl[c] = []
    for i, a in enumerate(assignments):
        cl[a] += [i]
        #print "Needed time: %.3f seconds" % (time.time() - start_time)
    print('Number of clusters found', len(np.unique(assignments)))
    return Z, cl, assignments, distances


def findSmallestCluster(clusters):
    print("findSmallestCluster")
    minCl = list(clusters.keys())[0]
    for clName in clusters:
        if len(clusters[clName]) < len(clusters[minCl]):
            minCl = clName
    return minCl


def super_impose(structure_ref, structure_align, atom_indices=None):
    if atom_indices is not None:
        a_atoms = structure_align.xyz[atom_indices]
        r_atoms = structure_ref.xyz[atom_indices]
    else:
        a_atoms = structure_align.xyz
        r_atoms = structure_ref.xyz

    # Center coordinates
    n = r_atoms.shape[0]
    av1 = a_atoms.sum(axis=0) / n
    av2 = r_atoms.sum(axis=0) / n
    re = structure_ref.xyz - av2
    al = structure_align.xyz - av1

    # Calculate rotation matrix
    a = np.dot(np.transpose(al), re)
    u, d, vt = np.linalg.svd(a)

    rot = np.transpose(np.dot(np.transpose(vt), np.transpose(u)))
    if np.linalg.det(rot) < 0:
        vt[2] = -vt[2]
        rot = np.transpose(np.dot(np.transpose(vt), np.transpose(u)))

    # Rotate structure
    structure_align.xyz = np.dot(al, rot)


def average(structures, weights=None, write=True, filename=None):
    """
    Calculates weighted average of a list of structures.
    saves to filename if write is True
    if filename not provided makes new "average.pdb" file in temp-folder
    of the system

    Example:
    >>> import lib
    >>> t = lib.TrajectoryFile('./sample_data/structure/2807_8_9_b.h5', mode='r', stride=1)
    >>> avg = t.average
    >>> avg
    <lib.structure.Structure.Structure at 0x117ff770>
    >>> avg.filename
    'c:\\users\\peulen\\appdata\\local\\temp\\average.pdb'
    """
    if weights is None:
        weights = np.ones(len(structures), dtype=np.float64)
        weights /= weights.sum()
    else:
        weights = np.array(weights)
    avg = Structure()
    avg.atoms = np.copy(structures[0].atoms)
    avg.xyz *= 0.0
    for i, s in enumerate(structures):
        avg.xyz += weights[i] * s.xyz
    filename = os.path.join(tempfile.tempdir, "average.pdb") if filename is None else filename
    if write:
        avg.filename = filename
        avg.write()
    return avg


def find_best(target, reference, atom_indices=None):
    """
    target and reference are both of type mdtraj.Trajectory
    reference is of length 1, target of arbitrary length

    returns a Structure object and the index within the trajectory

    >>> import lib
    >>> t = t = lib.TrajectoryFile('./sample_data/structure/2807_8_9_b.h5', mode='r', stride=1)
    >>> find_best(t.mdtraj, t.mdtraj[2])
    (2,
     <mdtraj.Trajectory with 1 frames, 2495 atoms, 164 residues, without unitcells at 0x13570b30>)
    """
    rmsds = mdtraj.rmsd(target, reference, atom_indices=atom_indices)
    iMin = np.argmin(rmsds)
    return iMin, target[iMin]


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


class TrajectoryFile(lib.Genealogy, mdtraj.Trajectory):
    parameterNames = ['rmsd', 'drmsd', 'energy', 'chi2']

    def __init__(self, p_object, structure=None, **kwargs):
        """Creates an Trajectory of Structures given a HDF5-File using mdtraj.HDF5TrajectoryFile

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

        See Also
        --------
        lib.Structure

        Examples
        --------

        Making new h5-Trajectory file

        >>> import lib
        >>> s = lib.Structure('/sample_data/structure/T4L_Topology.pdb', verbose=True, make_coarse=False)
        >>> t = lib.TrajectoryFile('test.h5', s, mode='w')
        >>> t[0]
        <lib.structure.Structure.Structure at 0x11f34e10>
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

        >>> import lib
        >>> t = lib.TrajectoryFile('./sample_data/structure/2807_8_9_b.h5', mode='r', stride=1)
        >>> print t[0:3]
        [<lib.structure.Structure.Structure at 0x1345d5d0>,
        <lib.structure.Structure.Structure at 0x1345d610>,
        <lib.structure.Structure.Structure at 0x132d2230>]

        Name of the trajectory
        >>> print t.name
        '/ sample_data/ structure/ sample_data/ structure/ T4L_Trajectory.h5'

        initialize with mdtraj.Trajectory
        >>> import lib
        >>> t = lib.TrajectoryFile('./sample_data/structure/2807_8_9_b.h5', mode='r', stride=1)
        >>> t2 = lib.TrajectoryFile(t.mdtraj, filename='test.h5')

        Attributes:
        -----------
        rmsd : array/list containing the rmsd vs the reference structure of -new- / added structures upon addition
        of the strucutre

        """
        self.mode = kwargs.get('mode', 'r')
        self.atom_indices = kwargs.get('atom_indices', None)
        self.make_coarse = kwargs.get('make_coarse', False)
        self.stride = kwargs.get('stride', 1)
        self._rmsd_ref_state = kwargs.get('rmsd_ref_state', 0)
        self.verbose = kwargs.get('stride', True)
        self.center = kwargs.get('center', True)

        pdb_tmp = tempfile.mktemp(".pdb")
        if isinstance(p_object, str):
            if p_object.endswith('.pdb'):
                self.structure = Structure(p_object, self.make_coarse, self.verbose)
                self._filename = kwargs.get('filename', tempfile.mktemp(".h5"))
                self._mdtraj = mdtraj.Trajectory.load(p_object)
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
            mdtraj.Trajectory.__init__(self, p_object.xyz, p_object.topology)
            self._filename = kwargs.get('filename', tempfile.mktemp(".h5"))
        elif isinstance(p_object, Structure):
            self.structure = p_object
            self.structure.write(pdb_tmp)
            self._mdtraj = mdtraj.Trajectory.load(pdb_tmp)
            mdtraj.Trajectory.__init__(self, self._mdtraj.xyz, self._mdtraj.topology)
            self._filename = kwargs.get('filename', tempfile.mktemp(".h5"))

        if self.center:
            self._mdtraj.center_coordinates()

        lib.Genealogy.__init__(self)

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

    @property
    def filename(self):
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
        try:
            fn = copy(self.directory + self.filename)
            return fn.replace('/', '/ ')
        except AttributeError:
            return "None"

    @property
    def rmsd_ref_state(self):
        return self._rmsd_ref_state

    @rmsd_ref_state.setter
    def rmsd_ref_state(self, ref_frame):
        self._rmsd_ref_state = ref_frame
        self.rmsd = mdtraj.rmsd(self, self, ref_frame)

    @property
    def directory(self):
        return os.path.dirname(self.filename)

    @property
    def reference(self):
        if self.rmsd_ref_state == 'average':
            return self.average
        else:
            return self[int(self.rmsd_ref_state)]

    @property
    def average(self):
        return average(self[:len(self)])

    @property
    def values(self):
        """
        >>> import lib
        >>> t = lib.TrajectoryFile('./sample_data/structure/2807_8_9_b.h5', mode='r', stride=1)
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
        """
        Append a structure of type :py::class`lib.Structure` to the trajectory

        :param structure: Structure
        :param update_rmsd: bool
        :param energy: float
            Energy of the system
        :param energy_fret: float
            Energy of the FRET-potential
        :param verbose: bool
            By default True. If True energy, energy_fret, RMSD and dRMSD are printed to std-out.

        Example
        -------
        >>> import lib
        >>> t = lib.TrajectoryFile('./sample_data/structure/2807_8_9_b.h5', mode='r', stride=1)
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
        >>> import lib
        >>> t = lib.TrajectoryFile('./sample_data/structure/2807_8_9_b.h5', mode='r', stride=1)
        >>> for s in t:
        >>>     print s
        [<lib.structure.Structure.Structure object at 0x12FAE330>, <lib.structure.Structure.Structure object at 0x12FAE3B0>, <li
        b.structure.Structure.Structure object at 0x11852070>, <lib.structure.Structure.Structure object at 0x131052D0>, <lib.st
        ructure.Structure.Structure object at 0x13195270>, <lib.structure.Structure.Structure object at 0x13228210>]
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

        >>> import lib
        >>> t = lib.TrajectoryFile('./sample_data/structure/2807_8_9_b.h5', mode='r', stride=1)
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
                s = deepcopy(self.structure)
                s.atoms = np.copy(self.structure.atoms)
                s.xyz = self.mdtraj[i].xyz * 10.0
                s.update()
                re.append(s)
            return re


internal_atom_numbers = [
    ('N', 0),
    ('CA', 1),
    ('C', 2),
    ('O', 3),
    ('CB', 4),
    ('H', 5),
    ('CG', 6),
    ('CD', 7),
]

residue_atoms_internal = OrderedDict([
    ('CYS', ['N', 'C', 'CA', 'CB', 'O', 'H']),
    ('MET', ['N', 'C', 'CA', 'CB', 'O', 'H']),
    ('PHE', ['N', 'C', 'CA', 'CB', 'O', 'H']),
    ('ILE', ['N', 'C', 'CA', 'CB', 'O', 'H']),
    ('LEU', ['N', 'C', 'CA', 'CB', 'O', 'H']),
    ('VAL', ['N', 'C', 'CA', 'CB', 'O', 'H']),
    ('TRP', ['N', 'C', 'CA', 'CB', 'O', 'H']),
    ('TYR', ['N', 'C', 'CA', 'CB', 'O', 'H']),
    ('ALA', ['N', 'C', 'CA', 'CB', 'O', 'H']),
    ('GLY', ['N', 'C', 'CA', 'O', 'H']),
    ('THR', ['N', 'C', 'CA', 'CB', 'O', 'H']),
    ('SER', ['N', 'C', 'CA', 'CB', 'O', 'H']),
    ('GLN', ['N', 'C', 'CA', 'CB', 'O', 'H']),
    ('ASN', ['N', 'C', 'CA', 'CB', 'O', 'H']),
    ('GLU', ['N', 'C', 'CA', 'CB', 'O', 'H']),
    ('ASP', ['N', 'C', 'CA', 'CB', 'O', 'H']),
    ('HIS', ['N', 'C', 'CA', 'CB', 'O', 'H']),
    ('ARG', ['N', 'C', 'CA', 'CB', 'O', 'H']),
    ('LYS', ['N', 'C', 'CA', 'CB', 'O', 'H']),
    ('PRO', ['N', 'C', 'CA', 'CB', 'O', 'H', 'CG', 'CD']),
]
)

cartesian_keys = ['i', 'chain', 'res_id', 'res_name', 'atom_id', 'atom_name', 'element', 'coord',
                  'charge', 'radius', 'bfactor']

cartesian_formats = ['i4', '|S1', 'i4', '|S5', 'i4', '|S5', '|S1', '3f8', 'f4int', 'f4', 'f4']

internal_keys = ['i', 'ib', 'ia', 'id', 'b', 'a', 'd']
internal_formats = ['i4', 'i4', 'i4', 'i4', 'f8', 'f8', 'f8']

a2id = dict(internal_atom_numbers)
id2a = dict([(a[1], a[0]) for a in internal_atom_numbers])
res2id = dict([(aa, i) for i, aa in enumerate(residue_atoms_internal)])


def r2i(coord_i, a1, a2, a3, an, ai):
    """
    Cartisian coordinates to internal-coordinates
    """
    vn = an['coord']
    v1 = a1['coord']
    v2 = a2['coord']
    v3 = a3['coord']
    b = vector.norm(v3 - vn)
    a = vector.angle(v2, v3, vn)
    d = vector.dihedral(v1, v2, v3, vn)
    coord_i[ai] = an['i'], a3['i'], a2['i'], a1['i'], b, a, d
    return ai + 1


def move_center_of_mass(structure, all_atoms):
    for i, res in enumerate(structure.residue_ids):
        at_nbr = np.where(all_atoms['res_id'] == res)[0]
        cb_nbr = structure.l_cb[i]
        if cb_nbr > 0:
            cb = structure.atoms[cb_nbr]
            cb['coord'] *= cb['mass']
            for at in at_nbr:
                atom = all_atoms[at]
                residue_name = atom['res_name']
                if atom['atom_name'] not in residue_atoms_internal[residue_name]:
                    cb['coord'] += atom['coord'] * atom['mass']
                    cb['mass'] += atom['mass']
            cb['coord'] /= cb['mass']
            structure.atoms[cb_nbr] = cb


def calc_internal_coordinates(structure):
    structure.coord_i = np.zeros(structure.atoms.shape[0], dtype={'names': internal_keys, 'formats': internal_formats})
    rp, ai = None, 0
    res_nr = 0
    for rn in list(structure.residue_dict.values()):
        res_nr += 1
        # BACKBONE
        if rp is None:
            structure.coord_i[ai] = rn['N']['i'], 0, 0, 0, 0.0, 0.0, 0.0
            ai += 1
            structure.coord_i[ai] = rn['CA']['i'], rn['N']['i'], 0, 0, \
                                    vector.norm(rn['N']['coord'] - rn['CA']['coord']), 0.0, 0.0
            ai += 1
            structure.coord_i[ai] = rn['C']['i'], rn['CA']['i'], rn['N']['i'], 0, \
                                    vector.norm(rn['CA']['coord'] - rn['C']['coord']), \
                                    vector.angle(rn['C']['coord'], rn['CA']['coord'], rn['N']['coord']), \
                                    0.0
            ai += 1
        else:
            ai = r2i(structure.coord_i, rp['N'], rp['CA'], rp['C'], rn['N'], ai)
            ai = r2i(structure.coord_i, rp['CA'], rp['C'], rn['N'], rn['CA'], ai)
            ai = r2i(structure.coord_i, rp['C'], rn['N'], rn['CA'], rn['C'], ai)
        ai = r2i(structure.coord_i, rn['N'], rn['CA'], rn['C'], rn['O'], ai)  # O
        # SIDECHAIN
        resName = rn['CA']['res_name']
        if resName != 'GLY':
            ai = r2i(structure.coord_i, rn['O'], rn['C'], rn['CA'], rn['CB'], ai)  # CB
        if resName != 'PRO':
            ai = r2i(structure.coord_i, rn['N'], rn['CA'], rn['C'], rn['H'], ai)  # H
        else:
            ai = r2i(structure.coord_i, rn['N'], rn['CA'], rn['CB'], rn['CG'], ai)  # CG
            ai = r2i(structure.coord_i, rn['CA'], rn['CB'], rn['CG'], rn['CD'], ai)  # CD
        rp = rn
    print("Atoms internal: %s" % (ai + 1))
    print("--------------------------------------")
    structure.coord_i = structure.coord_i[:ai]
    structure._phi_indices = [list(structure.coord_i['i']).index(x) for x in structure.l_c]
    structure._omega_indices = [list(structure.coord_i['i']).index(x) for x in structure.l_ca]
    structure._psi_indices = [list(structure.coord_i['i']).index(x) for x in structure.l_n]
    structure._chi_indices = [list(structure.coord_i['i']).index(x) for x in structure.l_cb if x >= 0]


def make_residue_lookup_table(structure):
    l_residue = np.zeros((structure.n_residues, structure.max_atom_residue), dtype=np.int32) - 1
    n = 0
    res_dict = structure.residue_dict
    for residue in list(res_dict.values()):
        for atom in list(residue.values()):
            atom_name = atom['atom_name']
            if atom_name in list(a2id.keys()):
                l_residue[n, a2id[atom_name]] = atom['i']
        n += 1
    l_ca = l_residue[:, a2id['CA']]
    l_cb = l_residue[:, a2id['CB']]
    l_c = l_residue[:, a2id['C']]
    l_n = l_residue[:, a2id['N']]
    l_h = l_residue[:, a2id['H']]
    return l_residue, l_ca, l_cb, l_c, l_n, l_h


def get_residue_sequence(structure):
    try:
        residue_dict = structure.residue_dict
        return [residue_dict[key]['C']['res_name'] for key in list(residue_dict.keys())]
    except KeyError:
        return structure.io.sequence(structure)


class Structure(lib.Data):

    max_atom_residue = 16

    def __init__(self, filename=None, make_coarse=True, verbose=False, auto_update=True):
        """

        Attributes
        ----------
        max_atom_residue : int
            Maximum atoms per residue. The maximum atoms per residue have to be limited, because of the
            residue/atom lookup tables. By default maximum 16 atoms per residue are allowed.

        xyz : numpy-array
            The xyz-attribute is an array of the cartesian coordinates represented by the attribute atom

        residue_names: list
            residue names is a list of the residue-names. Here the residues are represented by the same
            name as the initial PDB-file, usually 3-letter amino-acid code. The residue_names attribute
            may look like: ['TYR', 'HIS', 'ARG']

        n_atoms: int
            The number of atoms

        b_factos: array
            An array of the b-factors.

        radius_gyration: float
            The attribute radius_gyration returns the radius of gyration of the structure. The radius of gyration
            is given by: rG = (np.sqrt((coord - rM) ** 2).sum(axis=1)).mean()
            Here rM are the mean coordinates of the structure.


        Parameters
        ----------
        :param filename: str
            Path to the pdb file on disk
        :param make_coarse: bool
            Conversion to coarse representation using internal coordinates. Side-chains are
            not considered. The Cbeta-atoms is moved to center of mass of the side-chain.
        :param verbose: bool
            print output to stdout
        :param auto_update: bool
            update cartesian-coordiantes automatically after change of internal coordinates.
            This only applies for coarse-grained coordinates

        Examples
        --------
        >>> import lib
        >>> s_coarse = lib.Structure('/sample_data/structure/HM_1FN5_Naming.pdb', verbose=True, make_coarse=True)
        ======================================
        Filename: /sample_data/structure/HM_1FN5_Naming.pdb
        Path: /sample_data/structure
        Number of atoms: 9316
        --------------------------------------
        Atoms internal: 3457
        --------------------------------------
        >>> print s_coarse
        ATOM   2274    O ALA   386      52.927  10.468 -17.263  0.00  0.00             O
        ATOM   2275   CB ALA   386      53.143  12.198 -14.414  0.00  0.00             C
        >>> s_coarse.omega *= 0.0
        >>> s_coarse.omega += 3.14
        >>> print s_coarse
        ATOM   2273    C ALA   386      47.799  59.970  21.123  0.00  0.00             C
        ATOM   2274    O ALA   386      47.600  59.096  20.280  0.00  0.00             O
        >>> s_coarse.write('test_out.pdb')
        >>> s_aa = lib.Structure('/sample_data/structure/HM_1FN5_Naming.pdb', verbose=True, make_coarse=False)
        >>> print s_aa
        ATOM   9312    H MET   583      40.848  10.075  17.847  0.00  0.00             H
        ATOM   9313   HA MET   583      40.666   8.204  15.667  0.00  0.00             H
        ATOM   9314  HB3 MET   583      38.898   7.206  16.889  0.00  0.00             H
        ATOM   9315  HB2 MET   583      38.796   8.525  17.846  0.00  0.00             H
        >>> print s_aa.omega
        array([], dtype=float64)
        >>> s_aa.to_coarse()
        >>> print s_aa
        ATOM   3451   CA MET   583      40.059   8.800  16.208  0.00  0.00             C
        ATOM   3452    C MET   583      38.993   9.376  15.256  0.00  0.00             C
        ATOM   3453    O MET   583      38.405  10.421  15.616  0.00  0.00             O
        ATOM   3454   CB MET   583      39.408   7.952  17.308  0.00  0.00             C
        ATOM   3455    H MET   583      40.848  10.075  17.847  0.00  0.00             H
        print s_aa.omega
        array([ 0.        ,  3.09665806, -3.08322105,  3.13562203,  3.09102453,...])
        """

        lib.Data.__init__(self)
        self.is_coarse = make_coarse
        self.verbose = verbose
        self.auto_update = auto_update
        self.io = lib.io.PDB

        if filename is not None:
            self.filename = filename

            ####################################################
            ######         PREPARE   COORDINATES          ######
            ####################################################
            self.atoms = self.io.read(filename, verbose=self.verbose)
            self.coord_i = np.zeros(self.atoms.shape[0], dtype={'names': internal_keys,
                                                                'formats': internal_formats})

            ####################################################
            ######         LOOKUP TABLES                  ######
            ####################################################
            self.l_res, self.l_ca, self.l_cb, self.l_c, self.l_n, self.l_h = make_residue_lookup_table(self)
            self.residue_types = np.array([res2id[res] for res in list(self.atoms['res_name'])
                                           if res in list(res2id.keys())], dtype=np.int32)
            self.dist_ca = np.zeros((self.n_residues, self.n_residues), dtype=np.float64)

            self._phi_indices = []
            self._omega_indices = []
            self._psi_indices = []
            self._chi_indices = []

            self.sequence = get_residue_sequence(self)

        if make_coarse:
            self.to_coarse()

    def to_coarse(self):
        """
        Converts the structure-instance into a coarse structure.

        Examples
        --------

        >>> s_aa = lib.Structure('/sample_data/structure/HM_1FN5_Naming.pdb', verbose=True, make_coarse=False)
        >>> print s_aa
        ATOM   9312    H MET   583      40.848  10.075  17.847  0.00  0.00             H
        ATOM   9313   HA MET   583      40.666   8.204  15.667  0.00  0.00             H
        ATOM   9314  HB3 MET   583      38.898   7.206  16.889  0.00  0.00             H
        ATOM   9315  HB2 MET   583      38.796   8.525  17.846  0.00  0.00             H
        >>> s_aa.to_coarse()
        >>> print s_aa
        ATOM   3451   CA MET   583      40.059   8.800  16.208  0.00  0.00             C
        ATOM   3452    C MET   583      38.993   9.376  15.256  0.00  0.00             C
        ATOM   3453    O MET   583      38.405  10.421  15.616  0.00  0.00             O
        ATOM   3454   CB MET   583      39.408   7.952  17.308  0.00  0.00             C
        ATOM   3455    H MET   583      40.848  10.075  17.847  0.00  0.00             H
        print s_aa.omega
        array([ 0.        ,  3.09665806, -3.08322105,  3.13562203,  3.09102453,...])
        """
        self.is_coarse = True

        ####################################################
        ######       TAKE ONLY INTERNAL ATOMS         ######
        ####################################################
        all_atoms = np.copy(self.atoms)
        tmp = [atom for atom in all_atoms if atom['atom_name'] in residue_atoms_internal[atom['res_name']]]
        atoms = np.empty(len(tmp), dtype={'names': lib.io.PDB.keys, 'formats': lib.io.PDB.formats})
        atoms[:] = tmp
        atoms['i'] = np.arange(atoms.shape[0])
        atoms['atom_id'] = np.arange(atoms.shape[0])
        self.atoms = atoms

        ####################################################
        ######         LOOKUP TABLES                  ######
        ####################################################
        self.l_res, self.l_ca, self.l_cb, self.l_c, self.l_n, self.l_h = make_residue_lookup_table(self)
        tmp = [res2id[res] for res in list(self.atoms['res_name']) if res in list(res2id.keys())]
        self.residue_types = np.array(tmp, dtype=np.int32)
        self.dist_ca = np.zeros((self.n_residues, self.n_residues), dtype=np.float64)

        ####################################################
        ######         REASSIGN COORDINATES           ######
        ####################################################
        move_center_of_mass(self, all_atoms)

        ####################################################
        ######         INTERNAL  COORDINATES          ######
        ####################################################
        self.coord_i = np.zeros(self.atoms.shape[0], dtype={'names': internal_keys, 'formats': internal_formats})
        calc_internal_coordinates(self)
        self.update_coordinates()
        self.sequence = get_residue_sequence(self)

    @property
    def internal_coordinates(self):
        return self.coord_i

    @property
    def xyz(self):
        return self.atoms['coord']

    @xyz.setter
    def xyz(self, v):
        self.atoms['coord'] = v

    @property
    def vdw(self):
        return self.atoms['radius']

    @vdw.setter
    def vdw(self, v):
        self.atoms['radius'] = v

    @property
    def residue_names(self):
        res_name = list(set(self.atoms['res_name']))
        res_name.sort()
        return res_name

    @property
    def residue_dict(self):
        residue_dict = OrderedDict()
        for res in list(set(self.atoms['res_id'])):
            at_nbr = np.where(self.atoms['res_id'] == res)
            residue_dict[res] = OrderedDict()
            for atom in self.atoms[at_nbr]:
                residue_dict[res][atom['atom_name']] = atom
        return residue_dict

    @property  # OK
    def n_atoms(self):
        return len(self.atoms)

    @property  # OK
    def n_residues(self):
        return len(self.residue_ids)

    @property  # OK
    def atom_types(self):
        return set(self.atoms['atom_name'])

    @property  # OK
    def residue_ids(self):
        residue_ids = list(set(self.atoms['res_id']))
        return residue_ids

    @property  # OK
    def b_factors(self):
        bfac = self.atoms[self.l_ca]['bfactor']
        return bfac

    @property
    def radius_gyration(self):
        coord = self.xyz
        rM = coord[:, :].mean(axis=0)
        rG = (np.sqrt((coord - rM) ** 2).sum(axis=1)).mean()
        return float(rG)

    def update_coordinates(self):
        if self.auto_update:
            self.update()

    def update_dist(self):
        #cStructure.atom_dist(self.dist_ca, self.l_res, self.atoms, a2id['CA'])
        cStructure.atom_dist(self.dist_ca, self.l_res, self.xyz, a2id['CA'])

    def update(self, start_point=0):
        cStructure.internal_to_cartesian(self.internal_coordinates, self.xyz, start_point)
        self.update_dist()

    @property
    def phi(self):
        return self.internal_coordinates[self._phi_indices]['d']

    @phi.setter
    def phi(self, v):
        self.internal_coordinates['d'][self._phi_indices] = v
        self.update_coordinates()

    @property
    def omega(self):
        return self.internal_coordinates[self._omega_indices]['d']

    @omega.setter
    def omega(self, v):
        self.internal_coordinates['d'][self._omega_indices] = v
        self.update_coordinates()

    @property
    def chi(self):
        return self.internal_coordinates[self._chi_indices]['d']

    @chi.setter
    def chi(self, v):
        self.internal_coordinates['d'][self._chi_indices] = v
        self.update_coordinates()

    @property
    def psi(self):
        return self.internal_coordinates[self._psi_indices]['d']

    @psi.setter
    def psi(self, v):
        self.internal_coordinates['d'][self._psi_indices] = v
        self.update_coordinates()

    def write(self, filename=None, append=False):
        if filename is None:
            filename = self.filename
        aw = np.copy(self.atoms)
        aw['coord'] = self.xyz
        self.io.write(filename, aw, append=append)

    def __str__(self):
        if self.atoms is None:
            return ""
        else:
            s = ""
            for at in self.atoms:
                s += "%-6s%5d %4s%1s%3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s%2s\n" % \
                     ("ATOM ",
                      at['atom_id'], at['atom_name'], " ", at['res_name'], at['chain'],
                      at['res_id'], " ",
                      at['coord'][0], at['coord'][1], at['coord'][2],
                      0.0, 0.0, "  ", at['element'])
            return s

    def __deepcopy__(self, memo):
        new = copy(self)
        new.atoms = np.copy(self.atoms)
        new.dist_ca = np.copy(self.dist_ca)
        new.filename = copy(self.filename)
        new.coord_i = np.copy(self.internal_coordinates)
        new.io = self.io

        new.l_ca = deepcopy(self.l_ca)
        new.l_cb = deepcopy(self.l_cb)
        new.l_c = deepcopy(self.l_c)
        new.l_n = deepcopy(self.l_n)
        new.l_h = deepcopy(self.l_h)
        new.l_res = deepcopy(self.l_res)

        return new
