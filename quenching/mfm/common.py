"""
common.py

Common data for all scripts in the SuRF-toolbox. In the common
library all kinds of constants are defined. For instance colors
in plots but also constants as the Avogadros number.
"""
from itertools import tee
import json


structure_data = json.load(open('./settings/structure.json'))
quencher = {'TRP': ['CB'], 'TYR': ['CB'], 'HIS': ['CB'], 'PRO': ['CB']}
quencher_names = quencher.keys()

constants = {'kappa2': 2. / 3.,
             'Na': 6.0221415e23, # Avogadro's number - particles / mol
             'nH2O': 1.33, # refractiveIndexWater
             'kB': 1.3806503e-23, # Boltzmann constant m2 kg/(s2 K)
             'T': 298.0}                # Temperature in Kelvin
"""Physical an chemical constants"""

MAX_BONDS = structure_data['MAX_BONDS']
"""Dictionary of maximum number of bonds"""

atom_weights = dict((key, structure_data["Periodic Table"][key]["Atomic weight"])
                    for key in structure_data["Periodic Table"].keys())
"""Atomic weights (http://www.chem.qmul.ac.uk/iupac/AtWt/ & PyMol) """


PKA_DICT = structure_data['PKA_DICT']
"""Dictionary of pKa values and un-protonated charge state."""

CHARGE_DICT = structure_data['CHARGE_DICT']
"""Default charges of amino acids"""

TITR_ATOM = structure_data['TITR_ATOM']
"""Atom on which to place charge in amino-acid"""

TITR_ATOM_COARSE = structure_data['TITR_ATOM_COARSE']
"""Atom on which to place charge in amino-acid (Coarse grained default position C-Beta)"""

MW_DICT = structure_data['MW_DICT']
"""Dictionary of amino acid molecular weights.  The the molecular weight of water
should be subtracted for each peptide bond to calculate a protein molecular
weight."""

MW_H2O = 18.0
"""Molecular weight of water"""

VDW_DICT = dict((key, structure_data["Periodic Table"][key]["vdW radius"]) for key in structure_data["Periodic Table"].keys())
"""Dictionary of van der Waal radii
CR - coarse grained Carbon/Calpha
"""

atom_colors = {
    'C': [0, 1, 0],
    'N': [0, 0, 1],
    'H': [1, 1, 1],
    'O': [1, 0, 0],
    'S': [1, 1, 0],
    'P': [1, 0.5, 0]
}
"""Dictionary to assign atom-types to colors"""

# --------------------------------------------------------------------------- #
# Amino acid name data
# DON'T CHANGE ORDER!!!
# --------------------------------------------------------------------------- #

_aa_index = [('ALA', 'A'), # 0
             ('CYS', 'C'), # 1
             ('ASP', 'D'), # 2
             ('GLU', 'E'), # 3
             ('PHE', 'F'), # 4
             ('GLY', 'G'), # 5
             ('HIS', 'H'), # 6
             ('ILE', 'I'), # 7
             ('LYS', 'K'), # 8
             ('LEU', 'L'), # 9
             ('MET', 'M'), # 10
             ('ASN', 'N'), # 11
             ('PRO', 'P'), # 12
             ('GLN', 'Q'), # 13
             ('ARG', 'R'), # 14
             ('SER', 'S'), # 15
             ('THR', 'T'), # 16
             ('VAL', 'V'), # 17
             ('TRP', 'W'), # 18
             ('TYR', 'Y'), # 19
             ('cisPro', 'cP'), # 20
             ('transPro', 'tP'), # 21
             ('CYX', 'C'), # 22  in Amber CYS with disulfide-bridge
             ('HIE', 'H'), # 22  in Amber CYS with disulfide-bridge
]

AA3_TO_AA1 = dict(_aa_index)
AA1_TO_AA3 = dict([(aa[1], aa[0]) for aa in _aa_index])
AA3_TO_ID = dict([(aa[0], i) for i, aa in enumerate(_aa_index)])

# --------------------------------------------------------------------------- #
# PDB record data
# --------------------------------------------------------------------------- #

# Types of coordinate entries
COORD_RECORDS = ["ATOM  ", "HETATM"]


conversion_coefficents = {"RDA2Rmp": [-14.454, 1.45246, -0.0052, 1.9e-5],
        "Rmp2RDA": [12.57637, 0.584808, 0.005067, 0.000021],
        "RDAE2Rmp": [-33.132620, 1.963202, -0.008030, 0.000021],
        "Rmp2RDAE": [19.455314, 0.507799, 0.002925, -0.000003]
}
"""Dictionary of FRET-conversion functions (polynomal coefficents)"""


def convFRET2Rmp(type, dFRET):
    """
    Convert different types of distances (<RDA>, <RDA>E)using a standard polynomial as defined in the
    common.py file.

    :param type: either RDA or RDAE
    :param dFRET: float (the distance)
    :return: float distance between mean-positions
    """
    if type == 'RDA':
        c = conversion_coefficents['RDA2Rmp']
    elif type == 'RDAE':
        c = conversion_coefficents['RDAE2Rmp']
    Rmp = c[0] + c[1] * dFRET + c[2] * dFRET ** 2 + c[3] * dFRET ** 3
    return Rmp


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
