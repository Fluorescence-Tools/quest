import os
import locale
import threading
import numpy as np


keys = ['i', 'chain', 'res_id', 'res_name',
        'atom_id', 'atom_name', 'element',
        'coord',
        'charge', 'radius', 'bfactor', 'mass']

formats = ['i4', '|U1', 'i4', '|U5',
           'i4', '|U5', '|U1',
           '3f8',
           'f8', 'f8', 'f8', 'f8']




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


#: IMP builds decorator objects through SWIG, which is not safe to do from
#: several threads at once — a scan running sites in a pool hit
#: "Wrong number or type of arguments for overloaded function 'new_Atom'".
#: One lock around the parse, and a cache so a scan parses each structure once
#: instead of once per site.
_PARSE_LOCK = threading.Lock()
_PARSE_CACHE: dict[tuple, np.ndarray] = {}
_PARSE_CACHE_LIMIT = 8


def _cache_key(path: str):
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return (os.path.abspath(path), stat.st_mtime_ns, stat.st_size)


def _atom_record_selector():
    """IMP selector accepting exactly the records QuEst's parser accepted.

    QuEst has always read **ATOM records only** — HETATM waters, ions and
    ligands were never part of the steric environment the dye diffuses through.
    None of IMP's stock selectors expresses that (`AllPDBSelector` keeps
    everything, `NonWater*` still keeps ligands), and the difference is not
    cosmetic: on 148L it is 1322 atoms versus 1525, which would move every
    accessible volume.
    """
    import IMP.atom

    class _AtomRecords(IMP.atom.PDBSelector):
        def __init__(self):
            IMP.atom.PDBSelector.__init__(self, "ATOM records")

        def get_is_selected(self, record):
            # IMP hands this a raw line (2.17) or a PDBRecord (2.24); the
            # newer object answers the question directly.
            if hasattr(record, "get_is_atom"):
                return bool(record.get_is_atom())
            return str(record).startswith("ATOM")

    return _AtomRecords()


def read(filename, **kwargs):
    """Read a structure into QuEst's atom array, using IMP as the parser.

    Returns a NumPy structured array with the fields in :data:`keys`. PDB and
    mmCIF are both handled by IMP (``IMP.atom.read_pdb`` / ``read_mmcif``);
    QuEst no longer carries its own parser for either.

    Only ATOM records are read, which is what QuEst's own parser did; see
    :func:`_atom_record_selector` for why no stock IMP selector will do.

    Raises
    ------
    ImportError
        If IMP is unavailable. QuEst needs it for accessible volumes anyway.
    """
    verbose = kwargs.get("verbose", True)
    path = str(filename)
    if verbose:
        directory, base_name = os.path.split(path)
        print("======================================")
        print("Filename: %s" % path)
        print("Path: %s" % directory)

    key = _cache_key(path)
    if key is not None:
        cached = _PARSE_CACHE.get(key)
        if cached is not None:
            # A copy: callers treat the array as theirs to modify.
            return cached.copy()

    with _PARSE_LOCK:
        atoms = _read_with_imp(path)

    if key is not None:
        if len(_PARSE_CACHE) >= _PARSE_CACHE_LIMIT:
            _PARSE_CACHE.clear()
        _PARSE_CACHE[key] = atoms
    return atoms.copy()


def _read_with_imp(path):
    """Parse *path* into QuEst's atom array. Call under :data:`_PARSE_LOCK`."""
    import IMP
    import IMP.atom
    import IMP.core

    model = IMP.Model()
    selector = _atom_record_selector()
    if path.lower().endswith((".cif", ".mmcif")):
        hierarchy = IMP.atom.read_mmcif(path, model, selector)
    else:
        hierarchy = IMP.atom.read_pdb(path, model, selector)

    leaves = IMP.atom.get_leaves(hierarchy)
    atoms = np.zeros(len(leaves), dtype={"names": keys, "formats": formats})
    element_table = IMP.atom.get_element_table()

    for index, leaf in enumerate(leaves):
        particle = leaf.get_particle()
        atom = IMP.atom.Atom(particle)
        residue = IMP.atom.Residue(IMP.atom.get_residue(atom))
        chain = IMP.atom.Chain(IMP.atom.get_chain(residue))
        element = element_table.get_name(atom.get_element())
        radius = IMP.core.XYZR(particle).get_radius()
        mass = IMP.atom.Mass(particle).get_mass()

        atoms["i"][index] = index
        atoms["chain"][index] = chain.get_id()
        atoms["res_id"][index] = residue.get_index()
        atoms["res_name"][index] = residue.get_residue_type().get_string()
        atoms["atom_id"][index] = index
        atoms["atom_name"][index] = atom.get_atom_type().get_string().strip()
        atoms["element"][index] = element
        atoms["coord"][index] = list(IMP.core.XYZ(particle).get_coordinates())
        atoms["bfactor"][index] = atom.get_temperature_factor()
        atoms["radius"][index] = radius
        atoms["mass"][index] = mass

    return atoms

