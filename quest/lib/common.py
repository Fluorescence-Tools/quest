"""
common.py

Common data for all scripts in the SuRF-toolbox. In the common
library all kinds of constants are defined. For instance colors
in plots but also constants as the Avogadros number.
"""

supported_coordinate_files = ['PDB', 'PQR']

# --------------------------------------------------------------------------- #
# Chemical data
# --------------------------------------------------------------------------- #
constants = {'kappa2': 2. / 3.,
             'Na': 6.0221415e23, # Avogadro's number - particles / mol
             'nH2O': 1.33, # refractiveIndexWater
             'kB': 1.3806503e-23, # Boltzmann constant m2 kg/(s2 K)
             'T': 298.0}                # Temperature in Kelvin

# Dictionary of maximum bonds
MAX_BONDS = {'H': 1,
             'C': 4,
             'O': 3,
             'N': 4,
             'P': 5
}

# For Center of Mass Calculation.
# Taken from http://www.chem.qmul.ac.uk/iupac/AtWt/ & PyMol
atom_weights = {
    'H': 1.00794,
    'He': 4.002602,
    'Li': 6.941,
    'Be': 9.012182,
    'B': 10.811,
    'C': 12.0107,
    'N': 14.0067,
    'O': 15.9994,
    'F': 18.9984032,
    'Ne': 20.1797,
    'Na': 22.989770,
    'Mg': 24.3050,
    'Al': 26.981538,
    'Si': 28.0855,
    'P': 30.973761,
    'S': 32.065,
    'Cl': 35.453,
    'Ar': 39.948,
    'K': 39.0983,
    'Ca': 40.078,
    'Sc': 44.955910,
    'Ti': 47.867,
    'V': 50.9415,
    'Cr': 51.9961,
    'Mn': 54.938049,
    'Fe': 55.845,
    'Co': 58.933200,
    'Ni': 58.6934,
    'Cu': 63.546,
    'Zn': 65.39,
    'Ga': 69.723,
    'Ge': 72.64,
    'As': 74.92160,
    'Se': 78.96,
    'Br': 79.904,
    'Kr': 83.80,
    'Rb': 85.4678,
    'Sr': 87.62,
    'Y': 88.90585,
    'Zr': 91.224,
    'Nb': 92.90638,
    'Mo': 95.94,
    'Tc': 98.0,
    'Ru': 101.07,
    'Rh': 102.90550,
    'Pd': 106.42,
    'Ag': 107.8682,
    'Cd': 112.411,
    'In': 114.818,
    'Sn': 118.710,
    'Sb': 121.760,
    'Te': 127.60,
    'I': 126.90447,
    'Xe': 131.293,
    'Cs': 132.90545,
    'Ba': 137.327,
    'La': 138.9055,
    'Ce': 140.116,
    'Pr': 140.90765,
    'Nd': 144.24,
    'Pm': 145.0,
    'Sm': 150.36,
    'Eu': 151.964,
    'Gd': 157.25,
    'Tb': 158.92534,
    'Dy': 162.50,
    'Ho': 164.93032,
    'Er': 167.259,
    'Tm': 168.93421,
    'Yb': 173.04,
    'Lu': 174.967,
    'Hf': 178.49,
    'Ta': 180.9479,
    'W': 183.84,
    'Re': 186.207,
    'Os': 190.23,
    'Ir': 192.217,
    'Pt': 195.078,
    'Au': 196.96655,
    'Hg': 200.59,
    'Tl': 204.3833,
    'Pb': 207.2,
    'Bi': 208.98038,
    'Po': 208.98,
    'At': 209.99,
    'Rn': 222.02,
    'Fr': 223.02,
    'Ra': 226.03,
    'Ac': 227.03,
    'Th': 232.0381,
    'Pa': 231.03588,
    'U': 238.02891,
    'Np': 237.05,
    'Pu': 244.06,
    'Am': 243.06,
    'Cm': 247.07,
    'Bk': 247.07,
    'Cf': 251.08,
    'Es': 252.08,
    'Fm': 257.10,
    'Md': 258.10,
    'No': 259.10,
    'Lr': 262.11,
    'Rf': 261.11,
    'Db': 262.11,
    'Sg': 266.12,
    'Bh': 264.12,
    'Hs': 269.13,
    'Mt': 268.14,
}


# Dictionary of pKa values and un-protonated charge state.
PKA_DICT = {"ASP": 4.0,
            "CYS": 8.5,
            "GLU": 4.4,
            "TYR": 10.0,
            "CTERM": 3.1,
            "ARG": 12.0,
            "HIS": 6.5,
            "LYS": 10.4,
            "NTERM": 8.0}

CHARGE_DICT = {"ASP": -1.,
               "CYS": -1.,
               "GLU": -1.,
               "TYR": -1.,
               "CTERM": -1.,
               "ARG": 1.,
               "HIS": 1.,
               "LYS": 1.,
               "NTERM": 1.}

# Atom on which to place charge
TITR_ATOM = {"ASP": "CG ",
             "GLU": "CD ",
             "TYR": "OH ",
             "ARG": "CZ ",
             "HIS": "NE2",
             "LYS": "NZ "}

# Atom on which to place charge (in coarse mode always Cb for now)
TITR_ATOM_COARSE = {"ASP": "CB",
                    "GLU": "CB",
                    "TYR": "CB",
                    "ARG": "CB",
                    "HIS": "CB",
                    "LYS": "CB",
                    "CYS": "CB"}

# Dictionary of amino acid molecular weights.  The the molecular weight of water
# should be subtracted for each peptide bond to calculate a protein molecular
# weight.
MW_DICT = {"ALA": 89.09,
           "ARG": 174.20,
           "ASN": 132.12,
           "ASP": 133.10,
           "ASX": 132.61,
           "CYS": 121.15,
           "GLN": 146.15,
           "GLU": 147.13,
           "GLX": 146.64,
           "GLY": 75.07,
           "HIS": 155.16,
           "ILE": 131.17,
           "LEU": 131.17,
           "LYS": 146.19,
           "MET": 149.21,
           "PHE": 165.19,
           "PRO": 115.13,
           "SER": 105.09,
           "THR": 119.12,
           "TRP": 204.23,
           "TYR": 181.19,
           "VAL": 117.15,
           "IVA": 85.12,
           "STA": 157.15,
           "ACE": 43.30}
MW_H2O = 18.0

# Dictionary of van der Waal radii
VDW_DICT = {"CA": 1.870, "C": 1.760, "O": 1.400,
            "AD": 1.700, "AD": 1.700, "AE": 1.700, "AE": 1.700, "C0": 1.870,
            "CB": 1.870, "CD": 1.870, "CG": 1.870, "CE": 1.870, "CH": 1.760, "CZ": 1.870,
            "CR": 3.5, # coarse grained Carbon/Calpha
            "C": 1.76,
            # Nitrogen
            "N": 1.650, "ND": 1.650, "NE": 1.650, "NZ": 1.650, "NH": 1.650,
            # Oxygen
            "OD": 1.400, "O": 1.400, "OE": 1.400, "OG": 1.400, "OXT": 1.400,
            "OH": 1.400, "OP": 1.400,
            # Sulphur
            "SD": 1.850, "S": 1.850, "SG": 1.850,
            # Hydrogen
            "H": 1.1, "HB": 1.1, "HD": 1.1, "HG": 1.1, "HE": 1.1,
            "HZ": 1.1, "CH": 1.1, "HH": 1.1, "HA": 1.1, "HT": 1.1,
            # Phosphor
            "P": 1.900}

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

