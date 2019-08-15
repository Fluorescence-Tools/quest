import os
import locale
import numpy as np
import lib.common as common


keys = ['i', 'chain', 'res_id', 'res_name',
        'atom_id', 'atom_name', 'element',
        'coord',
        'charge', 'radius', 'bfactor', 'mass']

formats = ['i4', '|U1', 'i4', '|U5',
           'i4', '|U5', '|U1',
           '3f8',
           'f8', 'f8', 'f8', 'f8']


def assign_element_to_atom_name(
        atom_name: str
):
    """Tries to guess element from atom name if not recognised.

    :param atom_name: string

    Examples
    --------

    >>> assign_element_to_atom_name('CA')
    C
    """
    element = ""
    if atom_name.upper() not in common.atom_weights:
        # Inorganic elements have their name shifted left by one position
        #  (is a convention in PDB, but not part of the standard).
        # isdigit() check on last two characters to avoid mis-assignment of
        # hydrogens atoms (GLN HE21 for example)
        # Hs may have digit in [0]
        putative_element = atom_name[1] if atom_name[0].isdigit() else atom_name[0]
        if putative_element.capitalize() in common.atom_weights.keys():
            element = putative_element
    return element


def sequence(structure, use_atoms=False):
    """
    ParseModel the SEQRES entries in a pdb file.  If this fails, use the ATOM
    entries.  Return dictionary of sequences keyed to chain and type of
    sequence used.
    :param structure: a `Structure` object
    :param use_atoms: boolean
    """
    # Try using SEQRES
    pdb = open(structure.filename, 'r').readlines()
    # get residue-sequence
    seq = [l for l in pdb if l[0:6] == "SEQRES"]
    if len(seq) != 0 and not use_atoms:
        chain_dict = dict([(l[11], []) for l in seq])
        for c in list(chain_dict.keys()):
            chain_seq = [l[19:70].split() for l in seq if l[11] == c]
            for x in chain_seq:
                chain_dict[c].extend(x)
    else:
        # Check to see if there are multiple models.  If there are, only look
        # at the first model.
        models = [i for i, l in enumerate(pdb) if l.startswith("MODEL")]
        if len(models) > 1:
            pdb = pdb[models[0]:models[1]]
        atoms = [l for l in pdb if l[0:6] == "ATOM  " and l[13:16] == "CA "]
        chain_dict = dict([(l[21], []) for l in atoms])
        for c in list(chain_dict.keys()):
            chain_dict[c] = [l[17:20] for l in atoms if l[21] == c]
    return chain_dict



def parse_string_pdb(string, assignCharge=False, **kwargs):
    rows = string.splitlines()
    verbose = kwargs.get('verbose', True)
    atoms = np.zeros(len(rows), dtype={'names': keys, 'formats': formats})
    ni = 0
    for line in rows:
        if line.startswith('ATOM'):
            atom_name = line[12:16].strip().upper()
            atoms['i'][ni] = ni
            atoms['chain'][ni] = line[21]
            atoms['res_name'][ni] = line[17:20].strip().upper()
            atoms['atom_name'][ni] = atom_name
            atoms['res_id'][ni] = line[22:26]
            atoms['atom_id'][ni] = line[6:11]
            atoms['coord'][ni][0] = line[30:38]
            atoms['coord'][ni][1] = line[38:46]
            atoms['coord'][ni][2] = line[46:54]
            atoms['bfactor'][ni] = line[60:65]
            atoms['element'][ni] = assign_element_to_atom_name(atom_name)
            try:
                if assignCharge:
                    if atoms['res_name'][ni] in common.CHARGE_DICT:
                        if atoms['atom_name'][ni] == common.TITR_ATOM_COARSE[atoms['res_name'][ni]]:
                            atoms['charge'][ni] = common.CHARGE_DICT[atoms['res_name'][ni]]
                atoms['mass'][ni] = common.atom_weights[atoms['element'][ni]]
                atoms['radius'][ni] = common.VDW_DICT[atoms['element'][ni]]
            except KeyError:
                print("Cloud not assign parameters to: %s" % line)
            ni += 1
    atoms = atoms[:ni]
    if verbose:
        print("Number of atoms: %s" % (ni + 1))
    return atoms


def parse_string_pqr(string, **kwargs):
    rows = string.splitlines()
    verbose = kwargs.get('verbose', True)
    atoms = np.zeros(len(rows), dtype={'names': keys, 'formats': formats})
    ni = 0

    for line in rows:
        if line[:4] == "ATOM" or line[:6] == "HETATM":
            # Extract x, y, z, r from pqr to xyzr file
            atom_name = line[12:16].strip().upper()
            atoms['i'][ni] = ni
            atoms['chain'][ni] = line[21]
            atoms['atom_name'][ni] = atom_name.upper()
            atoms['res_name'][ni] = line[17:20].strip().upper()
            atoms['res_id'][ni] = line[21:27]
            atoms['atom_id'][ni] = line[6:11]
            atoms['coord'][ni][0] = "%10.5f" % float(line[30:38].strip())
            atoms['coord'][ni][1] = "%10.5f" % float(line[38:46].strip())
            atoms['coord'][ni][2] = "%10.5f" % float(line[46:54].strip())
            atoms['radius'][ni] = "%10.5f" % float(line[63:70].strip())
            atoms['element'][ni] = assign_element_to_atom_name(atom_name)
            atoms['charge'][ni] = "%10.5f" % float(line[55:62].strip())
            ni += 1

    atoms = atoms[:ni]
    if verbose:
        print("Number of atoms: %s" % (ni + 1))
    return atoms


def assign_element_to_atom_name(
        atom_name: str
):
    """Tries to guess element from atom name if not recognised.

    :param atom_name: string

    Examples
    --------

    >>> assign_element_to_atom_name('CA')
    C
    """
    if atom_name.upper() not in common.atom_weights:
        # Inorganic elements have their name shifted left by one position
        #  (is a convention in PDB, but not part of the standard).
        # isdigit() check on last two characters to avoid mis-assignment of
        # hydrogens atoms (GLN HE21 for example)
        # Hs may have digit in [0]
        putative_element = atom_name[1] if str(atom_name[0]).isdigit() else atom_name[0]
        if putative_element.capitalize() in common.atom_weights:
            atom_name = putative_element
        else:
            atom_name = ""
    return atom_name


def read(filename, assignCharge=False, **kwargs):
    """ Open pdb_file and read each line into pdb (a list of lines)

    :param filename:
    :return:
        numpy structured array containing the PDB info and VdW-radii and charges

    Examples
    --------

    >>> pdb_file = './sample_data/model/hgbp1/hGBP1_closed.pdb'
    >>> pdb = mfm.io.pdb_file.read(pdb_file, verbose=True)
    >>> pdb[:5]
    array([ (0, ' ', 7, 'MET', 1, 'N', 'N', [72.739, -17.501, 8.879], 0.0, 1.65, 0.0, 14.0067),
           (1, ' ', 7, 'MET', 2, 'CA', 'C', [73.841, -17.042, 9.747], 0.0, 1.76, 0.0, 12.0107),
           (2, ' ', 7, 'MET', 3, 'C', 'C', [74.361, -18.178, 10.643], 0.0, 1.76, 0.0, 12.0107),
           (3, ' ', 7, 'MET', 4, 'O', 'O', [73.642, -18.708, 11.489], 0.0, 1.4, 0.0, 15.9994),
           (4, ' ', 7, 'MET', 5, 'CB', 'C', [73.384, -15.89, 10.649], 0.0, 1.76, 0.0, 12.0107)],
          dtype=[('i', '<i4'), ('chain', 'S1'), ('res_id', '<i4'), ('res_name', 'S5'), ('atom_id', '<i4'), ('atom_name', 'S5
    '), ('element', 'S1'), ('coord', '<f8', (3,)), ('charge', '<f8'), ('radius', '<f8'), ('bfactor', '<f8'), ('mass', '<f8')
    ])
    """
    verbose = kwargs.get('verbose', True)
    with open(filename, 'r') as f:
        string = f.read()
        if verbose:
            path, baseName = os.path.split(filename)
            print("======================================")
            print("Filename: %s" % filename)
            print("Path: %s" % path)
        if filename.endswith('.pdb'):
            atoms = parse_string_pdb(string, assignCharge, **kwargs)
        elif filename.endswith('.pqr'):
            atoms = parse_string_pqr(string, **kwargs)
    return atoms


def write(filename, atoms=None, append=False):
    """
    Writes a structured numpy array containing the PDB-info to a
    PDB-file
    :param filename: target-filename
    :param atoms: structured numpy array
    """
    mode = 'a+' if append else 'w+'
    if atoms is not None:
        fp = open(filename, mode)
        # http://cupnet.net/pdb_format/
        al = ["%-6s%5d %4s%1s%3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s%2s\n" %
              ("ATOM ", at['atom_id'], at['atom_name'], " ", at['res_name'], at['chain'], at['res_id'], " ",
               at['coord'][0], at['coord'][1], at['coord'][2], 0.0, at['bfactor'], at['element'], "  ")
              for at in atoms
        ]
        if mode == 'a+':
            fp.write('MODEL')
        fp.write("".join(al))
        if mode == 'a+':
            fp.write('ENDMDL')
        fp.close()


class Pdb(object):
    """
    The Pdb-class is able to read and write PDB-coordinate files.
    For now only atoms with a ATOM-flag are reagarded models in
    PDB-Files are only suppoted partially. As in PDB-files no atom
    size is included the atomic-radii are taken from a VdW-dictionary
    located in :mod:`lib.common`.
    Naming scheme should follow the Pdb2Pqr internal naming scheme.

    grep -E '(\sCA\s)|(\sH\s)|(\sC\s)|(\sCB\s)|(\sN\s)|(\sCD\s)|(\sCG\s)|(\sO\s)' HM_1FN5_InternalNaming.pqr
    """

    ending = '.pdb'

    def __init__(self, verbose=False, assignCharge=None):
        self.assignCharge = assignCharge
        self.verbose = verbose
        locale.setlocale(locale.LC_NUMERIC, "")

    def sequence(self, use_atoms=False):
        """
        ParseModel the SEQRES entries in a pdb file.  If this fails, use the ATOM
        entries.  Return dictionary of sequences keyed to chain and type of
        sequence used.
        """
        # Try using SEQRES
        f = open(self.filename, 'r')
        pdb = f.readlines()
        f.close()
        # get residue-sequence
        seq = [l for l in pdb if l[0:6] == "SEQRES"]
        if len(seq) != 0 and not use_atoms:
            chain_dict = dict([(l[11], []) for l in seq])
            for c in list(chain_dict.keys()):
                chain_seq = [l[19:70].split() for l in seq if l[11] == c]
                for x in chain_seq:
                    chain_dict[c].extend(x)
        else:
            # Check to see if there are multiple models.  If there are, only look
            # at the first model.
            models = [i for i, l in enumerate(pdb) if l.startswith("MODEL")]
            if len(models) > 1:
                pdb = pdb[models[0]:models[1]]
            atoms = [l for l in pdb if l[0:6] == "ATOM  " and l[13:16] == "CA "]
            chain_dict = dict([(l[21], []) for l in atoms])
            for c in list(chain_dict.keys()):
                chain_dict[c] = [l[17:20] for l in atoms if l[21] == c]
        return chain_dict

    def writeData(self, filename, atoms=None, append=False):
        """
        :param filename:
        :param atoms:
        :return:
        """
        mode = 'a+' if append else 'w+'
        if atoms is not None:
            fp = open(filename, mode)
            # http://cupnet.net/pdb_format/
            al = ["%-6s%5d %4s%1s%3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s%2s\n" %
                  ("ATOM ", at['atom_id'], at['atom_name'], " ", at['res_name'], at['chain'], at['res_id'], " ",
                   at['coord'][0], at['coord'][1], at['coord'][2], 0.0, at['bfactor'], at['element'], "  ")
                  for at in atoms
            ]
            if mode == 'a+':
                fp.write('MODEL')
            fp.write("".join(al))
            if mode == 'a+':
                fp.write('ENDMDL')
            fp.close()

    def readData(self, filename, assignCharge=True):
        """
        Open pdb_file and read each line into pdb (a list of lines)
        :param filename:
        :return:
        """
        self.filename = filename
        assignCharge = int(self.assignCharge) if assignCharge is None else assignCharge
        if not os.path.isfile(filename):
            raise ValueError("")
        f = open(filename, 'r')
        rows = f.readlines()
        f.close()
        atoms = np.empty(len(rows), dtype={'names': keys, 'formats': formats})
        ni = 0
        for line in rows:
            if line[0:6] == "ATOM  ":
                atom_name = line[12:16].strip().upper()
                atoms['i'][ni] = ni
                atoms['chain'][ni] = line[21]
                atoms['res_name'][ni] = line[17:20].strip().upper()
                atoms['atom_name'][ni] = atom_name.strip().upper()
                atoms['res_id'][ni] = line[22:26]
                atoms['atom_id'][ni] = line[6:11]
                atoms['coord'][ni][0] = line[30:38]
                atoms['coord'][ni][1] = line[38:46]
                atoms['coord'][ni][2] = line[46:54]
                atoms['bfactor'][ni] = line[60:65]
                if assignCharge:
                    try:
                        if atoms['res_name'][ni] in common.CHARGE_DICT:
                            if atoms['atom_name'][ni] == common.TITR_ATOM_COARSE[atoms['res_name'][ni]]:
                                atoms['charge'][ni] = common.CHARGE_DICT[atoms['res_name'][ni]]
                        atoms['element'][ni] = assign_element(atoms['atom_name'][ni])
                        atoms['mass'][ni] = common.atom_weights[atoms['element'][ni]]
                        atoms['radius'][ni] = common.VDW_DICT[atoms['element'][ni]]
                    except KeyError:
                        print("Cloud not assign parameters to: %s" % line)
                ni += 1
        path, baseName = os.path.split(filename)
        if self.verbose:
            print("======================================")
            print("Filename: %s" % filename)
            print("Path: %s" % path)
            print("Number of atoms: %s" % (ni + 1))
            print("--------------------------------------")
        return atoms[:ni]
