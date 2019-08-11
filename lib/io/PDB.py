import os
import locale
import numpy as np
import lib.common as common


keys = ['i', 'chain', 'res_id', 'res_name',
             'atom_id', 'atom_name', 'element',
             'coord',
             'charge', 'radius', 'bfactor', 'mass']

formats = ['i4', '|S1', 'i4', '|S5',
           'i4', '|S5', '|S1',
           '3f8',
           'f8', 'f8', 'f8', 'f8']


def assign_element(atomName):
    """Tries to guess element from atom name if not recognised."""
    if not atomName or atomName.capitalize() not in common.atom_weights:
    # Inorganic elements have their name shifted left by one position
    #  (is a convention in PDB, but not part of the standard).
    # isdigit() check on last two characters to avoid mis-assignment of
    # hydrogens atoms (GLN HE21 for example)
        # Hs may have digit in [0]
        if atomName[0].isdigit():
            putative_element = atomName[1]
        else:
            putative_element = atomName[0]
        if putative_element.capitalize() in common.atom_weights:
            msg = "Used element %r for Atom (name=%s) with given element %r" % (putative_element, atomName, atomName)
            atomName = putative_element
        else:
            msg = "Could not assign element %r for Atom (name=%s) with given element %r" % \
                  (putative_element, atomName, atomName)
            atomName = ""
    return atomName


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


def read(filename, assignCharge=False, verbose=True):
    """
    Open pdb_file and read each line into pdb (a list of lines)
    :param filename:
    :return: numpy structured array containing the PDB info and VdW-radii and charges
    """
    #if verbose:
    print("Opening PDB-file: %s" % filename)
    if not os.path.isfile(filename):
        raise ValueError("PDB-Filename %s does not exist" % filename)
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
            atoms['res_name'][ni] = line[17:20]#.strip().upper()
            atoms['atom_name'][ni] = atom_name
            atoms['res_id'][ni] = line[22:26]
            atoms['atom_id'][ni] = line[6:11]
            atoms['coord'][ni][0] = line[30:38]
            atoms['coord'][ni][1] = line[38:46]
            atoms['coord'][ni][2] = line[46:54]
            atoms['bfactor'][ni] = line[60:65]
            try:
                if assignCharge:
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
    if verbose:
        print("======================================")
        print("Filename: %s" % filename)
        print("Path: %s" % path)
        print("Number of atoms: %s" % (ni + 1))
        print("--------------------------------------")
    return atoms[:ni]


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
                atoms['res_name'][ni] = line[17:20]#.strip().upper()
                atoms['atom_name'][ni] = atom_name
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
