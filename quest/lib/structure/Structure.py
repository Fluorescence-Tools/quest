from __future__ import annotations
import os
from copy import deepcopy, copy
from collections import OrderedDict
import tempfile

import numpy as np

import quest.lib
from quest.lib.structure import cStructure
from quest.lib.io.pdb import Pdb


clusterCriteria = ['maxclust', 'inconsistent', 'distance']


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


class Structure(object):

    max_atom_residue = 16

    def __init__(
            self,
            filename=None,
            make_coarse=True,
            verbose=False,
            auto_update=True
    ):
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
        >>> import quest.lib
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
        >>> s_aa = quest.lib.Structure('/sample_data/structure/HM_1FN5_Naming.pdb', verbose=True, make_coarse=False)
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

        super(Structure, self).__init__()
        self.is_coarse = make_coarse
        self.verbose = verbose
        self.auto_update = auto_update
        self.io = quest.lib.io.pdb

        if filename is not None:
            self.filename = filename

            ####################################################
            ######         PREPARE   COORDINATES          ######
            ####################################################
            self.atoms = self.io.read(filename, verbose=self.verbose)
            self.residue_types = np.array([res2id[res] for res in list(self.atoms['res_name'])
                                           if res in list(res2id.keys())], dtype=np.int32)
            self.dist_ca = np.zeros((self.n_residues, self.n_residues), dtype=np.float64)

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
        new.io = self.io

        return new
