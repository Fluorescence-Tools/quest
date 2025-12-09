import os.path
import json
import os
from collections import OrderedDict

import numpy as np
from numba import njit, prange
from qtpy import QtGui, QtCore, QtWidgets, uic
from lib.io import PDB
from lib.structure import Structure
from lib.io import write_xyz

try:
    import LabelLib as ll
    HAS_LABELLIB = True
except Exception:  # pragma: no cover - optional dependency
    HAS_LABELLIB = False

from quest.lib.imp_av import (
    HAS_IMP_BFF,
    build_imp_accessible_volume,
)


# Numba-accelerated kernels (flattened from fps.py)
@njit(cache=True)
def _density2points(n, npm, dg, density, r0, ng):
    r = np.empty((n, 3), dtype=np.float64)
    gd = np.arange(-npm, npm, dtype=np.float64) * dg
    n_out = 0
    for ix in range(-npm, npm):
        offset = ng * (ng * (ix + npm)) + npm
        for iy in range(-npm, npm):
            for iz in range(-npm, npm):
                if density[iz + offset] > 0:
                    r[n_out, 0] = gd[ix + npm] + r0[0]
                    r[n_out, 1] = gd[iy + npm] + r0[1]
                    r[n_out, 2] = gd[iz + npm] + r0[2]
                    n_out += 1
            offset += ng
    return r[:n_out]


def density2points(n, npm, dg, density, r0, ng):
    return _density2points(
        int(n),
        int(npm),
        float(dg),
        np.asarray(density, dtype=np.float64),
        np.asarray(r0, dtype=np.float64),
        int(ng),
    )


@njit(cache=True, parallel=True)
def _ran_dist(av1_points, av2_points, distances, n_samples):
    lp1 = av1_points.shape[0]
    lp2 = av2_points.shape[0]
    for i in prange(n_samples):
        i1 = int(np.random.random() * lp1)
        i2 = int(np.random.random() * lp2)
        d0 = av1_points[i1, 0] - av2_points[i2, 0]
        d1 = av1_points[i1, 1] - av2_points[i2, 1]
        d2 = av1_points[i1, 2] - av2_points[i2, 2]
        distances[i] = np.sqrt(d0 * d0 + d1 * d1 + d2 * d2)


def RDAMeanE(av1, av2, R0: float = 52.0, nSamples: int = 50_000):
    distances = np.empty(nSamples, dtype=np.float64)
    _ran_dist(av1.points, av2.points, distances, nSamples)
    Esum = np.mean(1.0 / (1.0 + (distances / R0) ** 6.0))
    return (1.0 / Esum - 1.0) ** (1.0 / 6.0) * R0


def RDAMean(av1, av2, nSamples: int = 50_000):
    distances = np.empty(nSamples, dtype=np.float64)
    _ran_dist(av1.points, av2.points, distances, nSamples)
    return float(np.mean(distances))


def dRmp(av1, av2):
    return float(np.sqrt(((av1.Rmp - av2.Rmp) ** 2).sum()))


@njit(cache=True, parallel=True)
def _subav(density, ng, slow_radius, rs, r0, dg):
    subav_density = density.copy()
    n_slow_center = rs.shape[0]
    for ix in prange(ng):
        for iy in range(ng):
            for iz in range(ng):
                if subav_density[ix, iy, iz] == 0:
                    continue
                is_slow = 0
                for isa in range(n_slow_center):
                    ix0 = int(((rs[isa, 0] - r0[0]) / dg)) + (ng - 1) // 2
                    iy0 = int(((rs[isa, 1] - r0[1]) / dg)) + (ng - 1) // 2
                    iz0 = int(((rs[isa, 2] - r0[2]) / dg)) + (ng - 1) // 2
                    slow_radius_idx = int(slow_radius[isa] / dg)
                    dx = ix - ix0
                    dy = iy - iy0
                    dz = iz - iz0
                    if dx * dx + dy * dy + dz * dz < slow_radius_idx * slow_radius_idx:
                        is_slow = 1
                        break
                if is_slow == 0:
                    subav_density[ix, iy, iz] = 0
    return subav_density


def subav(density, ng, dg, slow_radius, rs, r0):
    return _subav(
        np.asarray(density, dtype=np.uint8),
        int(ng),
        np.asarray(slow_radius, dtype=np.float64),
        np.asarray(rs, dtype=np.float64),
        np.asarray(r0, dtype=np.float64),
        float(dg),
    )


@njit(cache=True, parallel=True)
def _simulate_traj(d, ds, dg, t_max, t_step, D, slow_fact):
    n_samples = int(t_max / t_step)
    ng = d.shape[0]
    sigma = np.sqrt(2 * D * 3 * t_step) / dg
    pos = np.zeros((n_samples, 3), dtype=np.float64)
    accepted = np.zeros(n_samples, dtype=np.uint8)
    r = np.random.normal(0.0, sigma, size=(n_samples, 3))
    slow_fact_sqrt = np.sqrt(slow_fact)

    # pick a starting point in AV
    found = False
    for _ in range(1000):
        x_idx = np.random.randint(0, ng)
        y_idx = np.random.randint(0, ng)
        z_idx = np.random.randint(0, ng)
        if d[x_idx, y_idx, z_idx] > 0:
            pos[0, 0] = x_idx
            pos[0, 1] = y_idx
            pos[0, 2] = z_idx
            found = True
            break
    if not found:
        return pos, accepted, 0, 0

    n_accepted = 1
    n_rejected = 0
    i_accepted = 0
    for i in range(n_samples):
        x_idx = int(pos[i_accepted, 0])
        y_idx = int(pos[i_accepted, 1])
        z_idx = int(pos[i_accepted, 2])
        if ds[x_idx, y_idx, z_idx] > 0:
            r[i, 0] *= slow_fact_sqrt
            r[i, 1] *= slow_fact_sqrt
            r[i, 2] *= slow_fact_sqrt

        pos[i, 0] = pos[i_accepted, 0] + r[i, 0]
        pos[i, 1] = pos[i_accepted, 1] + r[i, 1]
        pos[i, 2] = pos[i_accepted, 2] + r[i, 2]

        x_idx = int(pos[i, 0])
        y_idx = int(pos[i, 1])
        z_idx = int(pos[i, 2])
        if 0 <= x_idx < ng and 0 <= y_idx < ng and 0 <= z_idx < ng and d[x_idx, y_idx, z_idx] > 0:
            i_accepted = i
            n_accepted += 1
            accepted[i] = 1
        else:
            n_rejected += 1
            accepted[i] = 0
    return (pos - (ng - 1) / 2.0) * dg, accepted, n_accepted, n_rejected


def simulate_traj(d, ds, dg, t_max, t_step, D, slow_fact):
    return _simulate_traj(
        np.asarray(d, dtype=np.uint8),
        np.asarray(ds, dtype=np.uint8),
        float(dg),
        float(t_max),
        float(t_step),
        float(D),
        float(slow_fact),
    )


def spherePoints(nSphere: int):
    inc = np.pi * (3.0 - np.sqrt(5.0))
    offset = 2.0 / float(nSphere)
    k_vals = np.arange(nSphere, dtype=np.float32)
    y = k_vals * offset - 1.0 + (offset / 2.0)
    rd = np.sqrt(1.0 - y * y)
    phi = k_vals * inc
    x = np.cos(phi) * rd
    z = np.sin(phi) * rd
    return np.stack([x, y, z], axis=1).astype(np.float32)


@njit(cache=True)
def _asa(r, vdw, probe_atom_indices, sphere_points, probe, radius):
    nSphere = sphere_points.shape[0]
    n_probe_atoms = probe_atom_indices.shape[0]
    n_atoms = r.shape[0]
    if n_probe_atoms == 0 or nSphere == 0 or n_atoms == 0:
        return np.zeros(n_probe_atoms, dtype=np.float32)

    neighbor_indices = np.zeros(n_atoms, dtype=np.uint32)
    probe_atom_asa = np.zeros(n_probe_atoms, dtype=np.float32)
    c = 4.0 * np.pi / float(nSphere)
    for i in range(n_probe_atoms):
        probe_atom_index = probe_atom_indices[i]
        n_neighbor = 0
        aX, aY, aZ = r[probe_atom_index]
        for atom_index in range(n_atoms):
            if probe_atom_index != atom_index:
                bX, bY, bZ = r[atom_index]
                dist2 = (aX - bX) ** 2 + (aY - bY) ** 2 + (aZ - bZ) ** 2
                if dist2 < 2 * (vdw[probe_atom_index] + probe):
                    neighbor_indices[n_neighbor] = atom_index
                    n_neighbor += 1

        n_accessible_point = 0
        for j in range(nSphere):
            is_accessible = True
            aX = sphere_points[j, 0] * radius + r[probe_atom_index, 0]
            aY = sphere_points[j, 1] * radius + r[probe_atom_index, 1]
            aZ = sphere_points[j, 2] * radius + r[probe_atom_index, 2]
            for k in range(n_neighbor):
                bX, bY, bZ = r[neighbor_indices[k]]
                dist2 = (aX - bX) ** 2 + (aY - bY) ** 2 + (aZ - bZ) ** 2
                if dist2 < (radius + probe) * (radius + probe):
                    is_accessible = False
                    break
            if is_accessible:
                n_accessible_point += 1
        probe_atom_asa[i] = c * n_accessible_point * vdw[probe_atom_index] ** 2
    return probe_atom_asa


def asa(r, vdw, probe_atom_indices, sphere_points, probe=1.0, radius=2.5):
    r = np.asarray(r, dtype=np.float32)
    vdw = np.asarray(vdw, dtype=np.float32)
    probe_atom_indices = np.asarray(probe_atom_indices, dtype=np.uint32)
    if sphere_points is None or np.asarray(sphere_points).ndim != 2:
        sphere_points = spherePoints(590)
    sphere_points = np.asarray(sphere_points, dtype=np.float32)
    return _asa(r, vdw, probe_atom_indices, sphere_points, float(probe), float(radius))


def density2points_ll(nx, ny, nz, g, dx, dy, dz, ox, oy, oz):
    points = np.empty((nx*ny*nz, 3), dtype=np.float64)
    iat = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                val = g[ix, iy, iz]
                if val <= 0.0:
                    continue
                iat += 1
                points[iat][0] = ix * dx + ox
                points[iat][1] = iy * dy + oy
                points[iat][2] = iz * dz + oz
    return points[:iat]


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
    print(nx, ny, nz, dg, dg, dg, x0, y0, z0)
    points = density2points_ll(nx, ny, nz, g, dg, dg, dg, x0, y0, z0)
    g[g < 0] = 0
    g[g > 0] = 1
    density = g.astype(np.uint8)
    ng = nx

    if verbose:
        print("Number of atoms: %i" % n_atoms)
        print("Attachment atom coordinates: %s" % r0)

    return points, density, ng, r0


class AV(object):

    def __init__(self, structure, residue_seq_number=None, atom_name=None, chain_identifier=None,
                 linker_length=20.0, linker_width=0.5, radius1=5.0,
                 radius2=4.5, radius3=3.5, simulation_grid_resolution=0.5, save_av=False, output_file='out',
                 attachment_atom_index=None, verbose=False, allowed_sphere_radius=0.5,
                 simulation_type='AV1', position_name=None, residue_name=None):

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
        >>> import lib
        >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
        >>> residue_number = 18
        >>> atom_name = 'CB'
        >>> attachment_atom = 1
        >>> av = lib.fps.AV(pdb_filename, attachment_atom=1, verbose=True)

        Calculating accessible volume using provided Structure object
        >>> import lib
        >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
        >>> structure = lib.Structure(pdb_filename)
        >>> av = lib.fps.AV(structure, attachment_atom=1, verbose=True)
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
        >>> import lib
        >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
        >>> structure = lib.Structure(pdb_filename)
        >>> av = lib.fps.AV(structure, residue_seq_number=11, atom_name='CB', verbose=True)

        If save_av is True the calculated accessible volume is save to disk. The filename of the calculated
        accessible volume is determined by output_file
        >>> import lib
        >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
        >>> structure = lib.Structure(pdb_filename)
        >>> av = lib.fps.AV(structure, residue_seq_number=11, atom_name='CB', verbose=True, save_av=True, output_file='test')
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
        self.verbose = verbose
        self.points_slow = None
        self.position_name = position_name
        self.residue_name = residue_name
        if verbose:
            print("")
            print("Calculating accessible volume")
            print("-----------------------------")
        if isinstance(structure, Structure):
            atoms = structure.atoms
        else:
            if verbose:
                print("Loading PDB")
            atoms = PDB.read(structure, verbose=verbose)
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

        def _decode(value):
            if isinstance(value, bytes):
                return value.decode().strip()
            if isinstance(value, np.bytes_):
                return value.tobytes().decode().strip()
            if isinstance(value, np.str_):
                return str(value).strip()
            return str(value).strip()

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
        else:
            if residue_seq_number is None:
                residue_seq_number = int(atoms['res_id'][attachment_atom_index])
            if atom_name is None:
                atom_name = _decode(atoms['atom_name'][attachment_atom_index])
            if chain_identifier is None:
                chain_identifier = _decode(atoms['chain'][attachment_atom_index])
            self.attachment_residue = residue_seq_number
            self.attachment_atom = atom_name

        if verbose:
            print("Calculating initial-AV")
            print("Linker-length  : %.2f" % linker_length)
            print("Linker-width   : %.2f" % linker_width)
            print("Linker-radius  : %.2f" % radius1)
            print("Attachment-atom: %i" % attachment_atom_index)
            print("AV-resolution  : %.2f" % simulation_grid_resolution)
        if HAS_IMP_BFF:
            av_result = build_imp_accessible_volume(
                structure=structure,
                residue_seq_number=residue_seq_number,
                atom_name=atom_name,
                chain_identifier=chain_identifier,
                linker_length=linker_length,
                linker_width=linker_width,
                radii=(radius1, radius2, radius3),
                allowed_sphere_radius=allowed_sphere_radius,
                simulation_grid_resolution=simulation_grid_resolution,
            )
            points = av_result.points if av_result.points.size else np.zeros((0, 3), dtype=np.float64)
            density = av_result.density
            ng = av_result.grid_size
            x0 = av_result.attachment_point
        elif HAS_LABELLIB:
            if simulation_type == 'AV3':
                points, density, ng, x0 = calculate3R(
                    linker_length,
                    linker_width,
                    radius1,
                    radius2,
                    radius3,
                    attachment_atom_index,
                    x,
                    y,
                    z,
                    vdw,
                    dg=simulation_grid_resolution,
                    verbose=verbose,
                    linkersphere=allowed_sphere_radius,
                )
            else:
                points, density, ng, x0 = calculate1R(
                    linker_length,
                    linker_width,
                    radius1,
                    attachment_atom_index,
                    x,
                    y,
                    z,
                    vdw,
                    verbose=verbose,
                    linkersphere=allowed_sphere_radius,
                    dg=simulation_grid_resolution,
                )
        else:
            raise RuntimeError(
                "Neither IMP.bff nor LabelLib are available to compute accessible volumes. "
                "Install IMP.bff (preferred) or LabelLib."
            )

        self.ng = ng
        self.points = points
        self.points_fast = points.copy()
        self.density_fast = density
        self.density_slow = None
        self.density = density
        self.x0 = x0
        if verbose:
            print("Points in total-AV: %i" % density.sum())
        if save_av:
            write_xyz(output_file + '.xyz', points, verbose=verbose)

    def calc_slow_av(self, slow_centers=None, slow_radius=None, save=True, verbose=True, replace_density=False):
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

        ng = int(self.ng)
        npm = int((ng - 1) // 2)
        density = self.density_fast
        dg = self.dg
        x0 = self.x0
        slow_density = subav(density, ng, dg, slow_radii, slow_centers, x0)
        n = slow_density.sum()
        points = density2points(n, npm, dg, slow_density.reshape(ng*ng*ng), x0, ng)
        if verbose or self.verbose:
            print("Points in slow-AV: %i" % n)
        if save:
            write_xyz(self.output_file + '_slow.xyz', points, verbose=verbose)
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
        >>> import lib
        >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
        >>> structure = lib.Structure(pdb_filename)
        >>> av1 = lib.fps.AV(structure, residue_seq_number=72, atom_name='CB')
        >>> av2 = lib.fps.AV(structure, residue_seq_number=134, atom_name='CB')
        >>> av1.dRmp(av2)
        """
        return dRmp(self, av)

    def dRDA(self, av):
        """
        Calculate the mean distance to the second accessible volume `av`

        Parameters
        ----------
        av : accessible volume object

        Example
        -------
        >>> import lib
        >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
        >>> structure = lib.Structure(pdb_filename)
        >>> av1 = lib.fps.AV(structure, residue_seq_number=72, atom_name='CB')
        >>> av2 = lib.fps.AV(structure, residue_seq_number=134, atom_name='CB')
        >>> av1.dRDA(av2)
        """
        return RDAMean(self, av)


    def dRDAE(self, av):
        """
        Calculate the FRET-averaged mean distance to the second accessible volume `av`

        Parameters
        ----------
        av : accessible volume object

        Example
        -------
        >>> import lib
        >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
        >>> structure = lib.Structure(pdb_filename)
        >>> av1 = lib.fps.AV(structure, residue_seq_number=72, atom_name='CB')
        >>> av2 = lib.fps.AV(structure, residue_seq_number=134, atom_name='CB')
        >>> av1.dRDAE(av2)
        """
        return RDAMeanE(self, av)

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


class AvPotential(object):

    name = 'Av'

    def __init__(self, distances=None, positions=None, av_samples=10000, min_av=150, verbose=False):
        """
        parameters should be a dictionary determining the
         system (labeling positions etc.)
        >>> import json
        >>> labeling_file = './sample_data/model/labeling.json'
        >>> labeling = json.load(open(labeling_file, 'r'))
        >>> distances = labeling['Distances']
        >>> positions = labeling['Positions']
        >>> import lib
        >>> av_potential = lib.fps.AvPotential(distances=distances, positions=positions)
        >>> structure = lib.Structure('/sample_data/model/HM_1FN5_Naming.pdb')
        >>> av_potential.getChi2(structure)
        """
        self.verbose = verbose
        self.distances = distances
        self.positions = positions
        self.n_av_samples = av_samples
        self.min_av = min_av
        self.avs = OrderedDict()

    @property
    def structure(self):
        return self._structure

    @structure.setter
    def structure(self, structure):
        self._structure = structure
        self.calc_avs()

    def calc_avs(self):
        if self.positions is None:
            raise ValueError("Positions not set unable to calculate AVs")
        arguments = [
            dict(
                {'structure': self.structure,
                'verbose': self.verbose,
                },
                **self.positions[position_key]
            )
            for position_key in self.positions
        ]
        avs = map(lambda x: AV(**x), arguments)
        for i, position_key in enumerate(self.positions):
            self.avs[position_key] = avs[i]

    def calc_distances(self, structure=None, verbose=False):
        verbose = verbose or self.verbose
        if isinstance(structure, Structure):
            self.structure = structure
        for distance_key in self.distances:
            distance = self.distances[distance_key]
            av1 = self.avs[distance['position1_name']]
            av2 = self.avs[distance['position2_name']]
            distance_type = distance['distance_type']
            R0 = distance['Forster_radius']
            if distance_type == 'RDAMean':
                d12 = RDAMean(av1, av2, self.n_av_samples)
            elif distance_type == 'Rmp':
                d12 = dRmp(av1, av2)
            elif distance_type == 'RDAMeanE':
                d12 = RDAMeanE(av1, av2, R0, self.n_av_samples)
            distance['model_distance'] = d12
            if verbose:
                print("-------------")
                print("Distance: %s" % distance_key)
                print("Forster-Radius %.1f" % distance['Forster_radius'])
                print("Distance type: %s" % distance_type)
                print("Model distance: %.1f" % d12)
                print("Experimental distance: %.1f (-%.1f, +%.1f)" % (distance['distance'],
                                                                      distance['error_neg'], distance['error_pos']))

    def getChi2(self, structure=None, reduced=False, verbose=False):
        verbose = self.verbose or verbose
        if isinstance(structure, Structure):
            self.structure = structure

        chi2 = 0.0
        self.calc_distances(verbose=verbose)
        for distance in list(self.distances.values()):
            dm = distance['model_distance']
            de = distance['distance']
            error_neg = distance['error_neg']
            error_pos = distance['error_pos']
            d = dm - de
            chi2 += (d / error_neg)**2 if d < 0 else (d / error_pos)**2
        if reduced:
            return chi2 / (len(list(self.distances.keys())) - 1.0)
        else:
            return chi2

    def getEnergy(self, structure=None):
        if isinstance(structure, Structure):
            self.structure = structure
        return self.getChi2()


class AvWidget(AvPotential, QtWidgets.QWidget):

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent=parent)
        uic.loadUi('./lib/fps/avWidget.ui', self)
        AvPotential.__init__(self)
        self._filename = None
        self.connect(self.actionOpenLabeling, QtCore.SIGNAL("triggered()"), self.onLoadAvJSON)

    def onLoadAvJSON(self):
        self.filename = str(QtWidgets.QFileDialog.getOpenFileName(None, 'Open FRET-JSON', '', 'link file (*.fps.json)')[0])

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, v):
        if os.path.isfile(v):
            self._filename = v
            p = json.load(open(v))
            self.distances = p["Distances"]
            self.positions = p["Positions"]
        else:
            v = 'None'
        self.lineEdit_2.setText(v)

    @property
    def n_av_samples(self):
        return int(self.spinBox_2.value())

    @n_av_samples.setter
    def n_av_samples(self, v):
        return self.spinBox_2.setValue(int(v))

    @property
    def min_av(self):
        return int(self.spinBox_2.value())

    @min_av.setter
    def min_av(self, v):
        return self.spinBox.setValue(int(v))

