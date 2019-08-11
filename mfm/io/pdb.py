import os

import numpy as np

import mfm
import mfm.common as common


keys = ['i', 'chain', 'res_id', 'res_name',
             'atom_id', 'atom_name', 'element',
             'coord',
             'charge', 'radius', 'bfactor', 'mass']

formats = ['i4', '|S1', 'i4', '|S5',
           'i4', '|S5', '|S1',
           '3f8',
           'f8', 'f8', 'f8', 'f8']



def assign_element_to_atom_name(atom_name):
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
        putative_element = atom_name[1] if atom_name[0].isdigit() else atom_name[0]
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

    >>> import mfm
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
    if isinstance(filename, str):
        verbose = kwargs.get('verbose', mfm.verbose)

        if verbose:
            print("Opening PDB-file: %s" % filename)
        if not os.path.isfile(filename):
            raise ValueError("PDB-Filename %s does not exist" % filename)
        f = open(filename, 'r')
        rows = f.readlines()
        f.close()
        atoms = np.empty(len(rows), dtype={'names': keys, 'formats': formats})
        ni = 0
        for line in rows:
            if line.startswith('ATOM'):
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
                #atoms['bfactor'][ni] = line[60:65]
                atoms['element'][ni] = assign_element_to_atom_name(atoms['atom_name'][ni])
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
        path, baseName = os.path.split(filename)
        if verbose:
            print("======================================")
            print("Filename: %s" % filename)
            print("Path: %s" % path)
            print("Number of atoms: %s" % (ni + 1))
            print("--------------------------------------")
        return atoms[:ni]
    else:
        return None


def write(filename, atoms=None, append_model=False, append_coordinates=False):
    """ Writes a structured numpy array containing the PDB-info to a PDB-file

    If append_model and append_coordinates are False the file is overwritten. Otherwise the atomic-coordinates
    are appended to the existing file.


    :param filename: target-filename
    :param atoms: structured numpy array
    :param append_model: bool
        If True the atoms are appended as a new model
    :param append_coordinates:
        If True the coordinates are appended to the file

    """
    mode = 'a+' if append_model or append_coordinates else 'w+'
    fp = open(filename, mode)

    # http://cupnet.net/pdb_format/
    al = ["%-6s%5d %4s%1s%3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2s%2s\n" %
          ("ATOM ", at['atom_id'], at['atom_name'], " ", at['res_name'], at['chain'], at['res_id'], " ",
           at['coord'][0], at['coord'][1], at['coord'][2], 0.0, at['bfactor'], at['element'], "  ")
          for at in atoms
    ]
    if append_model:
        fp.write('MODEL')
    fp.write("".join(al))
    if append_model:
        fp.write('ENDMDL')
    fp.close()


def write_xyz(filename, points, verbose=False):
    """
    Writes the points as xyz-format file. The xyz-format file can be opened and displayed for instance
    in PyMol

    :param filename: string
    :param points: array
    :param verbose: bool

    """
    if verbose:
        print "write_xyz\n"
        print "Filename: %s\n" % filename
    fp = open(filename, 'w')
    npoints = len(points)
    fp.write('%i\n' % npoints)
    fp.write('Name\n')
    for p in points:
        fp.write('D %.3f %.3f %.3f\n' % (p[0], p[1], p[2]))
    fp.close()


def get_attachment_atom_index(atoms, chain_identifier, residue_seq_number, atom_name, residue_name, **kwargs):
    """
    Get the attachment atom index by the the identifier

    :param atoms:
    :param chain_identifier:
    :param residue_seq_number:
    :param atom_name:
    :param residue_name:
    :param kwargs:
    :return:
    """
    # Determine Labeling position
    verbose = kwargs.get('verbose')
    if verbose:
        print("Labeling position")
        print("Chain ID: %s" % chain_identifier)
        print("Residue seq. number: %s" % residue_seq_number)
        print("Residue name: %s" % residue_name)
        print("Atom name: %s" % atom_name)

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
        print("Atom index: %s" % attachment_atom_index)
    return attachment_atom_index

