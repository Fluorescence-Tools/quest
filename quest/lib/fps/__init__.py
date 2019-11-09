from __future__ import annotations

import numpy as np

import quest.lib.fps.fps
import quest.lib.io
import LabelLib as ll
import quest.lib.structure
import quest.lib.fps.fps


def calculate1R(l: float,
                w: float,
                r: float,
                atom_i: int,
                x: np.array,
                y: np.array,
                z: np.array,
                vdw: np.array,
                linkersphere: float = 0.5,
                linknodes: int = 3,
                vdwRMax: float = 1.8,
                dg: float = 0.5,
                verbose: bool = False
                ):
    """
    :param l: linker length
    :param w: linker width
    :param r: dye-radius
    :param atom_i: attachment-atom index
    :param x: Cartesian coordinates of atoms (x)
    :param y: Cartesian coordinates of atoms (y)
    :param z: Cartesian coordinates of atoms (z)
    :param vdw: Van der Waals radii (same length as number of atoms)
    :param linkersphere: Initial linker-sphere to start search of allowed dye positions (not used anymore)
    :param linknodes: By default 3 (not used anymore)
    :param vdwRMax: Maximal Van der Waals radius (not used anymore)
    :param dg: Resolution of accessible volume in Angstrom
    :param verbose: If true informative output is printed on std-out
    :return:
    """
    if verbose:
        print("AV: calculate1R")
    n_atoms = len(vdw)
    x0, y0, z0 = x[atom_i], y[atom_i], z[atom_i]
    r0 = np.array([x0, y0, z0])

    vdw_copy = np.copy(vdw)
    vdw_copy[atom_i] = 0.0
    atoms = np.vstack([x, y, z, vdw_copy])
    source = r0
    av1 = ll.dyeDensityAV1(
        atoms,
        source,
        l, w, r, dg
    )
    nx, ny, nz = av1.shape
    g = np.array(av1.grid).reshape([nx, ny, nz], order='F')
    points = av1.points()

    # print(nx, ny, nz, dg, dg, dg, x0, y0, z0)
    # points = density2points_ll(nx, ny, nz, g, dg, dg, dg, x0, y0, z0)

    g[g < 0] = 0
    g[g > 0] = 1
    density = g.astype(np.uint8)
    ng = nx

    if verbose:
        print("Number of atoms: %i" % n_atoms)
        print("Attachment atom coordinates: %s" % r0)

    return points, density, ng, r0


class AV(object):

    def __init__(
            self,
            structure: quest.lib.structure.Structure,
            residue_seq_number: int = None,
            atom_name: str = None,
            chain_identifier: str = None,
            linker_length: float = 20.0,
            linker_width: float = 0.5,
            radius1: float = 5.0,
            radius2: float = 4.5,
            radius3: float = 3.5,
            simulation_grid_resolution: float = 0.5,
            save_av: bool = False,
            output_file: str = 'out',
            attachment_atom_index: int = None,
            verbose: bool = False,
            allowed_sphere_radius: float = 0.5,
            simulation_type: str = 'AV1',
            position_name: str = None,
            residue_name: str = None,
            min_points: int = 400
    ):

        """
        Parameters
        ----------
        :param structure: Structure of str
            either string with path of pdb-file or pdb object
        :param residue_seq_number:
        :param atom_name:
        :param chain_identifier:
        :param linker_length:
        :param linker_width:
        :param radius1:
        :param radius2:
        :param radius3:
        :param simulation_grid_resolution:
        :param save_av:
        :param output_file:
        :param attachment_atom_index:
        :param verbose:
        :param allowed_sphere_radius:
        :param simulation_type:
        :param position_name:
        :param residue_name:
        :raise ValueError:

        Examples
        --------
        Calculating accessible volume using provided pdb-file
        >>> import quest.lib
        >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
        >>> residue_number = 18
        >>> atom_name = 'CB'
        >>> attachment_atom = 1
        >>> av = quest.lib.fps.AV(pdb_filename, attachment_atom=1, verbose=True)

        Calculating accessible volume using provided Structure object
        >>> import quest.lib
        >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
        >>> structure = lib.Structure(pdb_filename)
        >>> av = quest.lib.fps.AV(structure, attachment_atom=1, verbose=True)
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
        >>> import quest.lib
        >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
        >>> structure = quest.lib.Structure(pdb_filename)
        >>> av = quest.lib.fps.AV(structure, residue_seq_number=11, atom_name='CB', verbose=True)

        If save_av is True the calculated accessible volume is save to disk. The filename of the calculated
        accessible volume is determined by output_file
        >>> import quest.lib
        >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
        >>> structure = quest.lib.Structure(pdb_filename)
        >>> av = quest.lib.fps.AV(structure, residue_seq_number=11, atom_name='CB', verbose=True, save_av=True, output_file='test')
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
        self.min_points = min_points
        self.verbose = verbose
        self.points_slow = None
        self.position_name = position_name
        self.residue_name = residue_name
        if verbose:
            print("")
            print("Calculating accessible volume")
            print("-----------------------------")
        if isinstance(structure, quest.lib.structure.Structure):
            atoms = structure.atoms
        else:
            if verbose:
                print("Loading PDB")
            atoms = quest.lib.io.PDB.read(
                filename=structure,
                verbose=verbose
            )
        x = atoms['coord'][:, 0]
        y = atoms['coord'][:, 1]
        z = atoms['coord'][:, 2]
        vdw = atoms['radius']
        self.structure = structure
        self.dg = simulation_grid_resolution
        self.output_file = output_file

        # Determine Labeling position
        if verbose:
            print("Labeling position")
            print("Chain ID: %s" % chain_identifier)
            print("Residue seq. number: %s" % residue_seq_number)
            print("Residue name: %s" % residue_name)
            print("Atom name: %s" % atom_name)

        if attachment_atom_index is None:
            if residue_seq_number is None or atom_name is None:
                raise ValueError("either attachment_atom number or residue and atom_name have to be provided.")
            self.attachment_residue = residue_seq_number
            self.attachment_atom = atom_name
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
                print("Atom index: %s" %attachment_atom_index)

        if verbose:
            print("Calculating initial-AV")
            print("Linker-length  : %.2f" % linker_length)
            print("Linker-width   : %.2f" % linker_width)
            print("Linker-radius  : %.2f" % radius1)
            print("Attachment-atom: %i" % attachment_atom_index)
            print("AV-resolution  : %.2f" % simulation_grid_resolution)
        if simulation_type == 'AV3':
            points, density, ng, x0 = calculate3R(linker_length, linker_width, radius1, radius2, radius3,
                                                  attachment_atom_index, x, y, z, vdw, dg=simulation_grid_resolution, verbose=verbose,
                                                  linkersphere=allowed_sphere_radius)
        else:
            points, density, ng, x0 = calculate1R(linker_length, linker_width, radius1, attachment_atom_index, x, y, z, vdw, verbose=verbose,
                                                  linkersphere=allowed_sphere_radius, dg=simulation_grid_resolution)
        self.ng = ng
        self.points = points
        self.points_fast = points
        self.density_fast = density
        self.density_slow = None
        self.density = density
        self.x0 = x0
        if verbose:
            print("Points in total-AV: %i" % density.sum())
        if save_av:
            quest.lib.io.write_xyz(output_file + '.xyz', points, verbose=verbose)

    def calc_slow_av(
            self,
            slow_centers=None,
            slow_radius=None,
            save=True,
            verbose=True,
            replace_density=False
    ):
        """
        Calculates a subav given the accessible volume determined during class initiation.
        : list/numpy array of points dimension (npoints, 3)


        Parameters
        ----------
        :param slow_centers : array_like
             A list of Cartesian coordinated representing the slow centers. The shape should be (n,3)
             where n is the number of slow centers.
        :param slow_radius : array_like, or float
            A list of radii the shape
        :param replace_density : bool
            If replace_density is True the AV-density will be replaced by the density of the slow
            accessible volume
        :param verbose : bool
        :param save : bool
            If save is True the slow-accessible volume is saved to an xyz-file with the suffix '_slow'

        """
        verbose = verbose or self.verbose
        if verbose:
            print("Calculating slow-AV:")
            print(slow_centers)
        if isinstance(slow_radius, (int, float)):
            slow_radii = np.ones(slow_centers.shape[0])*slow_radius
        elif len(slow_radius) != slow_centers.shape[0]:
            raise ValueError("The size of the slow_radius doesnt match the number of slow_centers")

        if verbose or self.verbose:
            print("Calculating slow AV")
            print("slow centers: \n%s" % slow_centers)

        ng = self.ng
        npm = (ng - 1) / 2
        density = self.density_fast
        dg = self.dg
        x0 = self.x0
        slow_density = quest.lib.fps.fps.subav(
            density,
            ng,
            dg,
            slow_radii,
            slow_centers,
            x0
        )
        n = slow_density.sum()
        points = fps.density2points(n, npm, dg, slow_density.reshape(ng*ng*ng), x0, ng)
        if verbose or self.verbose:
            print("Points in slow-AV: %i" % n)
        if save:
            quest.lib.io.write_xyz(self.output_file + '_slow.xyz', points, verbose=verbose)
        self.density_slow = slow_density
        self.points_slow = points
        if replace_density:
            self.density = slow_density

    @property
    def Rmp(self):
        return self.points.mean(axis=0)

    def dRmp(self, av):
        """
        Calculate the distance between the mean positions with respect to the accessible volume `av`

        Parameters
        ----------
        av : accessible volume object

        Example
        -------
        >>> import quest.lib
        >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
        >>> structure = quest.lib.Structure(pdb_filename)
        >>> av1 = quest.lib.fps.AV(structure, residue_seq_number=72, atom_name='CB')
        >>> av2 = quest.lib.fps.AV(structure, residue_seq_number=134, atom_name='CB')
        >>> av1.dRmp(av2)
        """
        return quest.lib.fps.fps.dRmp(self, av)

    def dRDA(self, av):
        """
        Calculate the mean distance to the second accessible volume `av`

        Parameters
        ----------
        av : accessible volume object

        Example
        -------
        >>> import quest.lib
        >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
        >>> structure = quest.lib.Structure(pdb_filename)
        >>> av1 = quest.lib.fps.AV(structure, residue_seq_number=72, atom_name='CB')
        >>> av2 = quest.lib.fps.AV(structure, residue_seq_number=134, atom_name='CB')
        >>> av1.dRDA(av2)
        """
        return quest.lib.fps.fps.RDAMean(self, av)

    def dRDAE(self, av):
        """
        Calculate the FRET-averaged mean distance to the second accessible volume `av`

        Parameters
        ----------
        av : accessible volume object

        Example
        -------
        >>> import quest.lib
        >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
        >>> structure = quest.lib.Structure(pdb_filename)
        >>> av1 = quest.lib.fps.AV(structure, residue_seq_number=72, atom_name='CB')
        >>> av2 = quest.lib.fps.AV(structure, residue_seq_number=134, atom_name='CB')
        >>> av1.dRDAE(av2)
        """
        return quest.lib.fps.fps.RDAMeanE(self, av)

    def __repr__(self):
        s = '\n'
        s += 'Accessible Volume\n'
        s += '-----------------\n'
        s += 'Attachment residue: %s\n' % self.attachment_residue
        s += 'Attachment atom: %s\n' % self.attachment_atom
        s += '\n'
        s += 'Fast-AV:\n'
        s += 'n-points: %s\n' % self.points_fast.shape[0]
        if self.points_slow is not None:
            s += 'Slow-AV:\n'
            s += 'n-points: %s\n' % self.points_slow.shape[0]
        else:
            s += 'Slow-AV: not calculated'
        return s

