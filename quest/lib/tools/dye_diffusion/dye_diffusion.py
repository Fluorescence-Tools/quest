import time
import os
from collections import OrderedDict
import gc
import json
import tempfile
from pathlib import Path
import inspect

import numpy as np
import numexpr as ne
from qtpy import QtGui, QtCore, QtWidgets, uic
import pyqtgraph as pg

from . import photon

import lib.io.PDB as PDB
from lib import Structure
import lib.fps as fps
from lib.plots.MolView import MolQtWidget
from lib.math import autocorr
from lib.widgets import PDBSelector
from lib.fps import AV, write_xyz
import os
from pathlib import Path

# Optional Chimol viewer (fallback to PyMOL if unavailable or disabled).
# ChiSurf setting: gui.mol_viewer in settings_chisurf.yaml (pymol|chimol).
try:
    import chisurf.settings as _cs_settings  # type: ignore
except Exception:
    _cs_settings = None

_quest_viewer_pref = None
if _cs_settings is not None:
    try:
        _quest_viewer_pref = (
            str(_cs_settings.gui.get("mol_viewer", "pymol"))  # type: ignore[attr-defined]
            .lower()
            .strip()
        )
    except Exception:
        _quest_viewer_pref = None
if not _quest_viewer_pref or _quest_viewer_pref not in ("pymol", "chimol"):
    _quest_viewer_pref = "pymol"

try:
    if _quest_viewer_pref == "chimol":
        from chisurf.plugins.chimol.chimol.renderer.view import MolView as _ChimolMolView
        from chisurf.plugins.chimol.chimol.io import load_structure_payload as _chimol_load_structure_payload

        _HAS_CHIMOL = True
    else:
        _ChimolMolView = None
        _chimol_load_structure_payload = None
        _HAS_CHIMOL = False
except Exception:
    _ChimolMolView = None
    _chimol_load_structure_payload = None
    _HAS_CHIMOL = False


class _ChimolCmdShim:
    """Minimal shim to satisfy .pymol.cmd.do/orient calls."""

    def __init__(self, adapter):
        self._adapter = adapter

    def do(self, command: str):
        cmd = (command or "").strip().lower()
        if cmd == "orient":
            try:
                self._adapter.orient()
            except Exception:
                pass
        # Other PyMOL commands are ignored in Chimol mode.

    # Compatibility for pymol.cmd.set(...) used nowhere else here.
    def set(self, *_, **__):
        return None

    def delete(self, *_, **__):
        return None


class _ChimolAdapter(QtWidgets.QWidget):
    """Lightweight adapter exposing the subset of PyMOL-like API we need."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.viewer = _ChimolMolView(self)
        self._cmd = _ChimolCmdShim(self)
        self._objects: list[str] = []

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.viewer)

    @property
    def pymol(self):
        return self

    @property
    def cmd(self):
        return self._cmd

    def reset(self):
        for obj_id in list(getattr(self.viewer, "_objects", {}).keys()):
            try:
                self.viewer.remove_object(obj_id)
            except Exception:
                pass
        self._objects.clear()

    def orient(self):
        try:
            self.viewer.reset_view()
        except Exception:
            pass

    def openFile(self, path, frame=None, mode=None, object_name=None, verbose=False):
        try:
            structure, coords = _chimol_load_structure_payload(Path(path))
        except Exception:
            structure, coords = None, None
        name = object_name or Path(path).stem
        obj_id = None
        try:
            if structure is not None:
                obj_id = self.viewer.add_structure(structure, name=name, source_path=str(path))
            elif coords is not None:
                obj_id = self.viewer.add_coordinates(coords, name=name, source_path=str(path))
        except Exception:
            obj_id = None

        if obj_id:
            self._objects.append(obj_id)
        return obj_id

    # PyMOL API compatibility
    @property
    def cmd(self):
        return self.cmd



def selection2atom_idx(pdb, res_types, verbose=False):
    """
    Lookup atom-idx using residues:
     a = {'TRP': 'CA'}
     selection2atom_idx(pdb, a, all_atoms=False)
    returns all CA atoms of TRP in the PDB

    selection2atom_idx(pdb, a, all_atoms=True)
    returns all atoms of TRP in the PDB

    """
    if verbose:
        print("selection2atom_idx")
    atom_idx = OrderedDict()
    for residue_key in res_types:
        atoms = []
        for atom_name in res_types[residue_key]:
            atoms.append(np.where((pdb['res_name'] == residue_key) & (pdb['atom_name'] == atom_name))[0])
        if len(atoms) > 0:
            atom_idx[residue_key] = np.array(np.hstack(atoms), dtype=np.uint32)
        else:
            atom_idx[residue_key] = np.array([], dtype=np.uint32)
    return atom_idx


def save_hist(filename, x, y, verbose=False):
    """
    Saves data x, y to file in histogram-format (csv). x and y
    should have the same lenght.
    :param filename: string
        Target filename
    :param x: array
    :param y: array
    :param verbose: bool
    """
    if verbose:
        print("Writing histogram to file: %s" % filename)
    fp = open(filename, 'w')
    for p in zip(x, y):
        fp.write('%.3f\t%i\n' % (p[0], p[1]))
    fp.close()


def is_quenched(distance2, critical_distance, verbose=False):
    """
    square of distance to quencher
    """
    critical_distance2 = critical_distance**2
    collided = np.less(distance2, critical_distance2)
    re = collided.sum(axis=0).T
    if verbose:
        print("Determine if frame is quenched")
        nq = re.sum()
        nt = re.shape[0]
        print("Quenched frames: %s/%s, %.2f%%" % (nq, nt, float(nq) / nt * 100.0))
    return np.asarray(re, dtype=np.uint8)


def get_quencher_coordinates(pdb, quencher, verbose=True, all_atoms=True):
    """
    returns coordinates and indices of all atoms of certain
    residue type.
    pdb: is a numpy-pdb array as returned by PDB.read
    """
    if verbose:
        print("Finding quenchers")
    print(quencher)
    atom_idx = selection2atom_idx(pdb, quencher)
    coordinates = OrderedDict()
    for res_key in quencher:
        coordinates[res_key] = pdb['coord'][atom_idx[res_key]]
    if verbose:
        print("Quencher atom-indeces: \n %s" % atom_idx)
        print("Quencher coordinates: \n %s" % coordinates)
    return coordinates


class SimulateDiffusion(object):

    @property
    def mean(self):
        traj = self._traj
        mean = traj.mean(axis=0)
        return mean

    @property
    def distance_to_mean(self):
        mean = self.mean
        dm = np.linalg.norm(self._traj - mean, axis=1)
        return dm

    @property
    def traj(self):
        if self.quencher is None:
            raise ValueError("Set quenching residues")
        if self._traj is None:
            raise ValueError("Run simulation first.")
        if self._critical_distance is None:
            raise ValueError("Set critical quenching distance")
        return self._traj, self._is_quenched

    @property
    def critical_distance(self):
        return self._critical_distance

    @property
    def quencher_distances(self):
        if self._quencher_distances is None:
            raise ValueError("Set quencher first.")
        return self._quencher_distances

    @critical_distance.setter
    def critical_distance(self, v):
        self._critical_distance = v
        self._is_quenched = is_quenched(self.quencher_distances, self._critical_distance, self.verbose)

    @property
    def quenched(self):
        if self._is_quenched is None:
            raise ValueError("Run simulation first and set critical distance")
        return self._is_quenched

    def save_trajectory(self, filename, skip):
        write_xyz(filename, (self._traj)[::skip])

    def __init__(self, av, quencher, output_file='out', verbose=False,
                 critical_distance=None, asa_calc=False, asa_n_sphere_point=590, asa_probe_radius=0.5,
                 trajectory_suffix='_traj.xyz', all_quencher_atoms=True):
        """

        :param av:
        :param quencher:
        :param output_file:
        :param verbose:
        :param critical_distance:
        :param asa_calc: bool
            By default False. If True the ASA of the residues is calculated
        :param asa_n_sphere_point: int
            By default 590. The number of sphere-points around each atoms used for the calculationg
            of the ASA.
        :param asa_probe_radius:
        :param trajectory_suffix:
        :param all_quencher_atoms:
        pdb_file: pdb-file of the structure
        """
        if verbose:
            print("")
            print("Simulate Diffusion")
            print("-----------------")
        self.av = av
        self.all_quencher_atoms = all_quencher_atoms

        # ASA of residues
        self.asa_calc = asa_calc
        self.asa_n_sphere_point = asa_n_sphere_point
        self.asa_probe_radius = asa_probe_radius
        self.asa_n_sphere_point = asa_n_sphere_point
        self.asa_sphere_points = fps.spherePoints(asa_n_sphere_point) if asa_calc else None
        self._asa = None

        self.pdb = av.structure.atoms
        self.verbose = verbose
        av_fast, av_slow, x0, dg = av.density_fast, av.density_slow, av.x0, av.dg
        self.dg = dg
        self.x0 = x0
        self.output_file = output_file
        self.av_fast = av_fast
        self.av_slow = av_slow
        self.t_step = None
        self._traj = None
        self._critical_distance = critical_distance

        self._quencher = None
        self._quencher_coordinates = None
        self._is_quenched = None
        self._kQ = None
        self.quencher = quencher

        self.trajectory_suffix = trajectory_suffix

    def get_quencher_distance2(self, quencher_coordinates, verbose=False):
        """
        quencher_coordinates: a list or 2D array containing positions of the quenchers
        `critical_distance`: flurophoreq gets quenched below the critical distance
        """
        verbose = self.verbose or verbose
        m, n = self._traj.shape[0], quencher_coordinates.shape[0]
        dist = np.zeros((n, m), dtype=np.float16)
        traj = self._traj
        if verbose:
            print("")
            print("Calculating quencher distances")
            print("Total quenching atoms: %s" % n)
        for i, quencher_position in enumerate(quencher_coordinates):
            dist[i] = ne.evaluate("sum((traj - quencher_position)**2, axis=1)")
        return dist

    @property
    def quencher(self):
        return self._quencher

    @quencher.setter
    def quencher(self, quencher):
        if self.all_quencher_atoms:
            q_new = OrderedDict()
            pdb = self.pdb
            for residue_key in quencher:
                atoms_idx = np.where(pdb['res_name'] == residue_key)[0]
                q_new[residue_key] = list(set(pdb[atoms_idx]['atom_name']))
            quencher = q_new
        self._quencher = quencher
        self._quencher_atom_indices = selection2atom_idx(self.pdb, quencher)
        if self.verbose:
            print("_quencher_atom_indices")
            print(self._quencher_atom_indices)

        if self.verbose:
            print("Calculating ASA of quenchers.")
            print("radii:")
            print(self.av.atoms['radius'])
            print("coordinates:")
            print(self.av.atoms['coord'])
            print("Quencher atom-indices")
            print(self._quencher_atom_indices)

        quencher_atom_index_array = np.hstack([v.flatten() for v in list(self._quencher_atom_indices.values())])
        asa_v = fps.asa(self.av.structure.atoms['coord'], self.av.structure.atoms['radius'], quencher_atom_index_array,
                        self.asa_sphere_points, self.asa_probe_radius)

        asa_it = (x for x in asa_v)
        asa = OrderedDict()
        for res_key in self._quencher_atom_indices:
            asa[res_key] = [next(asa_it) for idx in self._quencher_atom_indices[res_key]]
        self._asa = asa
        if self.verbose:
            print("Accessible surface area:")
            print(self._asa)
        self._quencher_coordinates = get_quencher_coordinates(self.pdb, quencher, verbose=self.verbose)
        if self.verbose:
            print("Quencher coordinates:")
            print(self._quencher_coordinates)

    @property
    def kQ(self):
        return self._kQ

    @property
    def quencher_atom_indices(self):
        return self._quencher_atom_indices

    @property
    def quencher_coordinates(self):
        return np.vstack(list(self._quencher_coordinates.values()))

    @property
    def quencher_asa(self):
        if self._asa is None:
            raise ValueError("Set quencher property quencher_atom_indices first.")

    @property
    def quencher_atom_indices(self):
        return self._quencher_atom_indeces

    @quencher_atom_indices.setter
    def quencher_atom_indices(self, atom_indices):
        if not isinstance(atom_indices, list):
            raise ValueError("Quencher indeces have to be a list of integers")
        atom_indices = np.array(self._quencher_atom_indices, dtype=np.uint32)
        self._quencher_atom_indeces = atom_indices

    def run(self, D=40.0, slow_fact=0.01, t_step=0.002, t_max=10000, save_traj=False, save_j=50, verbose=True, **kwargs):
        """

        :param D: float
            diffusion coefficient of dye in Ang**2/ns
        :param slow_fact: float
            The diffusion coefficient of the dye within the slow-accessible volume is slower by a factor of slow_fact.
        :param t_step: float
            The time-step of the simulation in nano-seconds
        :param t_max: float
            The maximum time of the simulation in nano-seconds
        :param save_traj: bool
            If True the trajectory is saved to self.output_file+'_traj.xyz'. Default value True
        :param save_j: int
            Only every save_j frame of the trajectory is saved.
        :param verbose:
        """
        verbose = verbose or self.verbose
        self.t_step = t_step
        dg = self.dg
        density, slow_density = self.av_fast, self.av_slow
        if verbose:
            print("Simulate dye-trajectory")
            print("Simulation time [us]: %.2f" % (t_max / 1000))
            print("Time-step [ps]: %.2f" % (t_step * 1000))
            print("Number of steps: %.2f" % int((t_max / t_step)))
            print("AV-grid parameter [Ang]: %.2f" % dg)
            print("Diff coeff. [Ang/ns2]: %.2f" % D)
            print("Shape-AV slow (x,y,z): %s %s %s" % (slow_density.shape))
            print("Shape-AV fast (x,y,z): %s %s %s" % (density.shape))
            print("------")
        start = time.process_time()
        traj, a, n_accepted, n_rejected = fps.simulate_traj(
            density, slow_density, dg,
            slow_fact=slow_fact, t_step=t_step, t_max=t_max, D=D
        )
        if n_accepted < 0:
            self._traj = None
            return
        traj += self.x0
        self._traj = traj

        if verbose:
            print("Accepted steps: %i" % n_accepted)
            print("Rejected steps: %i" % n_rejected)
        end = time.process_time()
        if verbose:
            print("time spent: %.2gs" % (end-start))
            print("n_accepted: %s" % n_accepted)
            print("-----------------------")
            print("Attachment point: %s" % self.x0)
            print("Mean dye-position: %s" % self.mean)

        if save_traj:
            self.save_trajectory(self.output_file+self.trajectory_suffix, save_j)
        self._quencher_distances = self.get_quencher_distance2(self.quencher_coordinates, verbose)
        if verbose:
            print("Quencher distances:")
            print("Quencher distances shape: %s, %s" % self._quencher_distances.shape)
            print(self._quencher_distances)


class DonorDecay(object):

    def __init__(self, tau0=None, kQ=1.0, nph=0, verbose=False, auto_update=False, pdb=None,
                 attachment_residue=None, attachment_atom=None, attachment_chain=None, dg=0.5, sticky_mode='quencher',
                 save_avs=True, D=None, slow_fact=0.05, critical_distance=None, output_file='out',
                 av_parameter={'linker_length': 20.0, 'linker_width': 0.5, 'radius1': 5.0},
                 quencher={'TRP': ['CB'], 'TYR': ['CB'], 'HIS': ['CB']}, t_max=10000.0, t_step=0.004,
                 slow_radius=10.0, all_quencher_atoms=True):
        """
        quenching trajectory as calculated by SimulateDiffusion.quenched
        :param tau0: float
        :param kQ: float
        :param nph:
        :param verbose: bool
        :param auto_update:
        :param pdb:
        :param attachment_residue: string
        :param attachment_atom: int
        :param attachment_chain: string
        :param dg: float
        :param sticky_mode:
        :param save_avs: bool
        :param D: float
        :param slow_fact: float
        :param critical_distance: float
        :param output_file: string
        :param av_parameter:
        :param quencher:
        :param t_max: float
        :param t_step: float
        :param slow_radius: float
        :param all_quencher_atoms:
        """
        self.verbose = verbose
        self._av = None
        self._diffusion = None
        self.tau0 = tau0
        self.n_photons = nph
        self.kQ = kQ
        self.auto_update = auto_update
        self._photon_trace = None
        self._pdb = pdb
        self.attachment_residue = attachment_residue
        self.attachment_atom = attachment_atom
        self.attachment_chain = attachment_chain
        self.diffusion_coefficient = D
        self.slow_fact = slow_fact
        self.dg = dg
        self.save_avs = save_avs
        self.critical_distance = critical_distance
        self.output_file = output_file
        self.av_parameter = av_parameter
        self.quencher = quencher
        self.t_max = t_max
        self.t_step = t_step
        self.sticky_mode = sticky_mode
        self.slow_radius = slow_radius
        self.all_quencher_atoms = all_quencher_atoms
        self._structure = None
        self.fret_enabled = False
        self.fret_R0 = None
        self.acceptor_chain = None
        self.acceptor_residue = None
        self.acceptor_atom = None
        self.fret_av_parameter = None
        self._acceptor_av = None
        self._photon_trace_fret = None
        self._fret_efficiency = None

    def __str__(self):
        s = ''
        return s

    @property
    def quantum_yield(self):
        dt, n = self.photon_trace
        n_ph = self.n_photons
        n_f = n.sum()
        return float(n_f) / n_ph

    @property
    def diffusion(self):
        if self._diffusion is None:
            self.simulate_diffusion()
        return self._diffusion

    @property
    def slow_center(self):
        if self.sticky_mode == 'quencher':
            coordinates = get_quencher_coordinates(self.structure.atoms, self.quencher)
            s = [np.vstack(coordinates[res_key]) for res_key in coordinates if len(coordinates[res_key]) > 0]
            coordinates = np.vstack(s)
        elif self.sticky_mode == 'surface':
            slow_atoms = np.where(self.structure.atoms['atom_name'] == 'CB')[0]
            coordinates = self.structure.atoms['coord'][slow_atoms]
        return coordinates

    @property
    def av(self):
        if self._av is None:
            self.calc_av()
            self._av.calc_slow_av(save=self.save_avs, slow_centers=self.slow_center,
                                  slow_radius=self.slow_radius, verbose=self.verbose)
        return self._av

    @property
    def photon_trace(self):
        if self._photon_trace is None:
            self.calc_photons()
        return self._photon_trace

    @property
    def collisions(self):
        return float(self.diffusion.quenched.sum()) / self.diffusion.quenched.shape[0]

    @property
    def structure(self):
        return self._structure

    @structure.setter
    def structure(self, v):
        self._structure = Structure(v, make_coarse=False)
        tf = tempfile.NamedTemporaryFile(suffix=".pdb")
        self._tmp_file = tf.name
        tf.close()
        self._structure.write(self._tmp_file)

    def calc_av(self, verbose=False):
        verbose = verbose or self.verbose
        dg = self.dg
        residue = self.attachment_residue
        attachment_atom = self.attachment_atom
        chain = self.attachment_chain
        structure = self.structure
        av = AV(structure, residue_seq_number=residue, atom_name=attachment_atom, chain_identifier=chain,
                simulation_grid_resolution=dg, save_av=self.save_avs, verbose=verbose,
                output_file=self.output_file, **self.av_parameter)
        self._av = av

    def simulate_diffusion(self, verbose=False, plot_traj=False):
        all_quencher_atoms = self.all_quencher_atoms
        verbose = verbose or self.verbose
        av = self.av
        D = self.diffusion_coefficient
        slow_fact = self.slow_fact
        diffusion = SimulateDiffusion(av, quencher=self.quencher, verbose=verbose, all_quencher_atoms=all_quencher_atoms)
        diffusion.run(D=D, slow_fact=slow_fact, t_max=self.t_max, t_step=self.t_step)
        self._diffusion = diffusion
        success = diffusion._traj is not None
        return success

    def calc_photons(self, verbose=False, donor_traj=None, kQ=None):
        """
        `n_ph`: number of generated photons
        `pq`: quenching probability if distance of fluorphore is below the `critical_distance`
        """
        verbose = verbose or self.verbose
        n_photons = self.n_photons
        if kQ is None:
            kQ = self.kQ
        tau0 = self.tau0
        if donor_traj is None:
            donor_traj = self.diffusion
        is_collided = donor_traj.quenched
        if verbose:
            print("")
            print("Simulating decay:")
            print("----------------")
            print("Number of excitation photons: %s" % n_photons)
            print("Number of frames    : %s" % is_collided.shape[0])
            print("Number of collisions: %s" % is_collided.sum())
            print("Quenching constant kQ[1/ns]: %s" % kQ)

        t_step = donor_traj.t_step
        #dts, phs = photon.simulate_photon_trace(n_ph=n_photons, collided=is_collided,
        #                                        quenching_prob=kQ, t_step=t_step, tau0=tau0)
        dts, phs = photon.simulate_photon_trace_kQ(n_ph=n_photons, collided=is_collided, kQ=kQ, t_step=t_step, tau0=tau0)

        #kQ = donor_traj.kQ
        #dts, phs = photon.simulate_photon_trace(n_ph=n_photons, kQ=kQ, t_step=t_step, tau0=tau0)

        n_photons = phs.shape[0]
        n_f = phs.sum()
        if verbose or self.verbose:
            print("Number of absorbed photons: %s" % (n_photons))
            print("Number of fluorescent photons: %s" % (n_f))
            print("Quantum yield: %.2f" % (float(n_f) / n_photons))
        self._photon_trace = dts, phs

    def get_histogram(self, nbins=4096, range=(0, 50)):
        dts, nph = self.photon_trace
        y, x = np.histogram(dts, range=range, bins=nbins)
        return x, y

    def save_histogram(self, filename='hist.txt', verbose=False, nbins=4096, range=(0, 50)):
        verbose = verbose or self.verbose
        x, y, = self.get_histogram(nbins, range)
        save_hist(filename, x, y, verbose)

    def calc_acceptor_av(self, verbose=False):
        verbose = verbose or self.verbose
        if self._structure is None:
            return None
        residue = self.acceptor_residue
        atom = self.acceptor_atom
        chain = self.acceptor_chain
        if residue is None or atom is None:
            return None
        params = self.fret_av_parameter if self.fret_av_parameter is not None else self.av_parameter
        av = AV(self.structure, residue_seq_number=residue, atom_name=atom, chain_identifier=chain,
                simulation_grid_resolution=self.dg, save_av=self.save_avs, verbose=verbose,
                output_file=self.output_file + '_A', **params)
        av.calc_slow_av(save=self.save_avs, slow_centers=self.slow_center,
                        slow_radius=self.slow_radius, verbose=verbose)
        self._acceptor_av = av
        return av

    def calc_fret_rate(self, verbose=False, n_samples=50000):
        if not getattr(self, "fret_enabled", False):
            return None
        verbose = verbose or self.verbose
        donor_av = self.av
        acceptor_av = self._acceptor_av if self._acceptor_av is not None else self.calc_acceptor_av(verbose=verbose)
        if acceptor_av is None:
            return None
        R0 = self.fret_R0 if self.fret_R0 is not None else 52.0
        try:
            r_eff = float(fps.RDAMeanE(donor_av, acceptor_av, R0, n_samples))
        except Exception:
            return None
        if r_eff <= 0:
            return None
        k_fret = (1.0 / self.tau0) * (R0 / r_eff) ** 6
        self._fret_efficiency = 1.0 / (1.0 + (r_eff / R0) ** 6)
        return k_fret

    def calc_photons_fret(self, verbose=False):
        verbose = verbose or self.verbose
        if not getattr(self, "fret_enabled", False):
            self._photon_trace_fret = None
            return
        k_fret = self.calc_fret_rate(verbose=verbose)
        if k_fret is None:
            self._photon_trace_fret = None
            return
        donor_traj = self.diffusion
        collided = donor_traj.quenched
        t_step = donor_traj.t_step
        n_photons = self.n_photons
        k_total = np.where(collided > 0, self.kQ + k_fret, k_fret).astype(np.float32)
        dts, phs = photon.simulate_photon_trace_rate(n_ph=n_photons, kQ=k_total, t_step=t_step, tau0=self.tau0)
        self._photon_trace_fret = dts, phs

    @property
    def photon_trace_fret(self):
        if self._photon_trace_fret is None:
            self.calc_photons_fret()
        return self._photon_trace_fret

    @property
    def quantum_yield_fret(self):
        if self._photon_trace_fret is None:
            self.calc_photons_fret()
        if self._photon_trace_fret is None:
            return 0.0
        dt, n = self._photon_trace_fret
        n_ph = self.n_photons
        n_f = n.sum()
        return float(n_f) / n_ph

    @property
    def fret_efficiency(self):
        if self._fret_efficiency is not None:
            return self._fret_efficiency
        qy_d = self.quantum_yield
        qy_da = self.quantum_yield_fret
        if qy_d <= 0:
            return 0.0
        return 1.0 - qy_da / qy_d

    def get_histogram_fret(self, nbins=4096, range=(0, 50)):
        if self._photon_trace_fret is None:
            self.calc_photons_fret()
        if self._photon_trace_fret is None:
            return None, None
        dts, nph = self._photon_trace_fret
        y, x = np.histogram(dts, range=range, bins=nbins)
        return x, y

    def update_all(self, verbose=False):
        verbose = verbose or self.verbose
        gc.collect()
        if verbose:
            print("Updating simulation")
            print("Calculating normal-AV")
        self.calc_av(verbose=verbose)
        if verbose:
            print("Calculating slow-AV")
        self._av.calc_slow_av(save=self.save_avs, slow_radius=self.slow_radius, slow_centers=self.slow_center)
        if verbose:
            print("Simulating diffusion")
        success = self.simulate_diffusion(verbose=verbose)
        print("success:", success)
        if success:
            if verbose:
                print("Determining quencher distances.")
            self.diffusion.critical_distance = self.critical_distance
            if verbose:
                print("Simulating decay")
            self.calc_photons(verbose=verbose)
            return True
        else:
            print("Trajectory calculation failed.")
            return False


class TransientDecayGenerator(QtWidgets.QWidget, DonorDecay):

    name = "Lifetime-Calculator"

    def __init__(
            self,
            verbose=False,
            settings_file=None
    ):
        super(TransientDecayGenerator, self).__init__()
        uic.loadUi(
            os.path.join(
                os.path.dirname(
                    __file__
                ),
                'dye_diffusion2.ui'
            ),
            self
        )
        if settings_file is None:
            settings_file = os.path.join(
                os.path.dirname(
                    __file__
                ),
                'dye_diffusion.json'
            )

        # Donor and acceptor label controls side-by-side
        self.donor_groupbox = QtWidgets.QGroupBox("Donor", self.groupBox)
        donor_layout = QtWidgets.QGridLayout(self.donor_groupbox)

        # Donor attachment selector (chain / residue / atom)
        self.pdb_selector = PDBSelector()
        donor_layout.addWidget(self.pdb_selector, 0, 0, 1, 2)

        # Move donor AV parameter widgets (Length/Width/Radius/dg) into the Donor box.
        # These widgets are created by dye_diffusion2.ui and originally live in
        # the right-hand column of the Label group; reparenting them here keeps
        # behavior identical but with a cleaner layout.
        try:
            donor_layout.addWidget(self.label_4, 1, 0)
            donor_layout.addWidget(self.doubleSpinBox, 1, 1)

            donor_layout.addWidget(self.label_5, 2, 0)
            donor_layout.addWidget(self.doubleSpinBox_2, 2, 1)

            donor_layout.addWidget(self.label_6, 3, 0)
            donor_layout.addWidget(self.doubleSpinBox_3, 3, 1)

            donor_layout.addWidget(self.label_23, 4, 0)
            donor_layout.addWidget(self.doubleSpinBox_17, 4, 1)

            # Donor photophysics: tau0 and diffusion coefficient D
            donor_layout.addWidget(self.label_12, 5, 0)
            donor_layout.addWidget(self.doubleSpinBox_8, 5, 1)

            donor_layout.addWidget(self.label_7, 6, 0)
            donor_layout.addWidget(self.doubleSpinBox_4, 6, 1)
        except AttributeError:
            # If any of these widgets are missing, leave the original layout.
            pass

        # FRET acceptor: checkable group box with its own PDB selector and AV params
        self.fret_groupbox = QtWidgets.QGroupBox("FRET acceptor", self.groupBox)
        self.fret_groupbox.setCheckable(True)
        self.fret_groupbox.setChecked(False)

        fret_layout = QtWidgets.QGridLayout(self.fret_groupbox)

        # Acceptor attachment selector (chain / residue / atom)
        self.acceptor_selector = PDBSelector(show_labels=False)
        fret_layout.addWidget(self.acceptor_selector, 0, 0, 1, 2)

        # Acceptor AV parameters: length / width / radius, mirroring donor UI
        label_len = QtWidgets.QLabel("Length", self.fret_groupbox)
        self.fret_len_spin = QtWidgets.QDoubleSpinBox(self.fret_groupbox)
        self.fret_len_spin.setMinimum(1.0)
        self.fret_len_spin.setMaximum(100.0)
        self.fret_len_spin.setSingleStep(0.5)
        self.fret_len_spin.setValue(float(self.doubleSpinBox.value()))
        fret_layout.addWidget(label_len, 1, 0)
        fret_layout.addWidget(self.fret_len_spin, 1, 1)

        label_wid = QtWidgets.QLabel("Width", self.fret_groupbox)
        self.fret_wid_spin = QtWidgets.QDoubleSpinBox(self.fret_groupbox)
        self.fret_wid_spin.setMinimum(0.1)
        self.fret_wid_spin.setMaximum(10.0)
        self.fret_wid_spin.setSingleStep(0.1)
        self.fret_wid_spin.setValue(float(self.doubleSpinBox_2.value()))
        fret_layout.addWidget(label_wid, 2, 0)
        fret_layout.addWidget(self.fret_wid_spin, 2, 1)

        label_rad = QtWidgets.QLabel("Radius", self.fret_groupbox)
        self.fret_rad_spin = QtWidgets.QDoubleSpinBox(self.fret_groupbox)
        self.fret_rad_spin.setMinimum(0.1)
        self.fret_rad_spin.setMaximum(50.0)
        self.fret_rad_spin.setSingleStep(0.25)
        self.fret_rad_spin.setValue(float(self.doubleSpinBox_3.value()))
        fret_layout.addWidget(label_rad, 3, 0)
        fret_layout.addWidget(self.fret_rad_spin, 3, 1)

        # Acceptor dg: share the same grid parameter as the donor (doubleSpinBox_17)
        label_dg = QtWidgets.QLabel("dg", self.fret_groupbox)
        self.fret_dg_spin = QtWidgets.QDoubleSpinBox(self.fret_groupbox)
        self.fret_dg_spin.setMinimum(self.doubleSpinBox_17.minimum())
        self.fret_dg_spin.setMaximum(self.doubleSpinBox_17.maximum())
        self.fret_dg_spin.setSingleStep(self.doubleSpinBox_17.singleStep())
        self.fret_dg_spin.setValue(float(self.doubleSpinBox_17.value()))
        fret_layout.addWidget(label_dg, 4, 0)
        fret_layout.addWidget(self.fret_dg_spin, 4, 1)
        try:
            self.fret_dg_spin.valueChanged.connect(self.onFretDgChanged)
        except Exception:
            pass

        # Acceptor diffusion coefficient D_A (independent of donor D)
        label_D = QtWidgets.QLabel("D [A2/ns]", self.fret_groupbox)
        self.fret_D_spin = QtWidgets.QDoubleSpinBox(self.fret_groupbox)
        try:
            self.fret_D_spin.setDecimals(self.doubleSpinBox_4.decimals())
            self.fret_D_spin.setMinimum(self.doubleSpinBox_4.minimum())
            self.fret_D_spin.setMaximum(self.doubleSpinBox_4.maximum())
            self.fret_D_spin.setSingleStep(self.doubleSpinBox_4.singleStep())
            self.fret_D_spin.setValue(float(self.doubleSpinBox_4.value()))
        except AttributeError:
            pass
        fret_layout.addWidget(label_D, 5, 0)
        fret_layout.addWidget(self.fret_D_spin, 5, 1)

        # Place donor and acceptor group boxes next to each other
        donor_acceptor_layout = QtWidgets.QHBoxLayout()
        donor_acceptor_layout.addWidget(self.donor_groupbox)
        donor_acceptor_layout.addWidget(self.fret_groupbox)
        self.verticalLayout_10.addLayout(donor_acceptor_layout)

        # The original "Dye" group box only contained tau0 and donor D, which
        # are now moved into the Donor group box; hide the empty container.
        try:
            self.groupBox_3.hide()
        except AttributeError:
            pass

        # Backwards-compatible alias: existing code checks self.fret_checkbox.isChecked()
        self.fret_checkbox = self.fret_groupbox
        try:
            self.fret_groupbox.toggled.connect(self.onFretToggled)
        except AttributeError:
            pass
        self._settings_file = None
        self.settings_file = settings_file
        fp = open(settings_file)
        settings = json.load(fp)
        fp.close()

        DonorDecay.__init__(self, **settings)
        ## User-interface
        self.actionLoad_PDB.triggered.connect(self.onLoadPDB)
        self.actionLoad_settings.triggered.connect(self.onLoadSettings)

        self.doubleSpinBox_6.valueChanged.connect(self.onSimulationTimeChanged)
        self.doubleSpinBox_7.valueChanged.connect(self.onSimulationDtChanged)
        self.pushButton_3.clicked.connect(self.update_all)
        self.pushButton_4.clicked.connect(self.onSaveHist)
        self.pushButton_5.clicked.connect(self.onSaveAV)

        self.tmp_dir = tempfile.gettempdir()
        print("Temporary Directory: %s" % self.tmp_dir)

        ## Decay-Curve
        decay_plot = pg.PlotWidget(title="Decay")
        decay_plot.setLabel('bottom', 'time [ns]')
        decay_plot.setLabel('left', 'counts')
        decay_plot.setLogMode(False, True)
        # Histogram (measured decay) in red, donor-only in cyan, FRET in magenta
        self.hist_curve = decay_plot.plot([], [], pen=pg.mkPen(color='r', width=1))
        self.unquenched_curve = decay_plot.plot([], [], pen=pg.mkPen(color='c', width=1))
        self.fret_curve = decay_plot.plot([], [], pen=pg.mkPen(color='m', width=1))
        self.plot_decay = decay_plot
        self.verticalLayout_2.addWidget(decay_plot)

        ## Diffusion-Trajectory-Curve
        traj_plot = pg.PlotWidget(title="Trajectory")
        traj_plot.setLabel('bottom', 'time [ns]')
        traj_plot.setLabel('left', '|R-<R>|')
        self.diffusion_curve = traj_plot.plot([], [], pen=pg.mkPen(color='b', width=1))
        self.plot_diffusion = traj_plot
        self.verticalLayout_6.addWidget(traj_plot)

        auto_plot = pg.PlotWidget(title="Auto-correlation")
        auto_plot.setLabel('bottom', 'corr. time [ns]')
        auto_plot.setLabel('left', 'A.Corr.(|R-<R>|)')
        auto_plot.setLogMode(True, False)
        self.diffusion_autocorrelation = auto_plot.plot([], [], pen=pg.mkPen(color='r', width=1))
        self.plot_autocorr = auto_plot
        self.verticalLayout_6.addWidget(auto_plot)

        ## Protein Structure
        self.viewer_backend = (_quest_viewer_pref if _HAS_CHIMOL else "pymol")
        self.molview = self._build_molview_widget()
        self.verticalLayout_4.addWidget(self.molview)
        self._chimol_object_id = None

        self.diff_file = None
        self.av_slow_file = None
        self.av_fast_file = None
        self.acceptor_av_slow_file = None
        self.acceptor_av_fast_file = None

        try:
            self.label_fret = QtWidgets.QLabel("E(FRET)", self.groupBox_7)
            self.doubleSpinBox_fret = QtWidgets.QDoubleSpinBox(self.groupBox_7)
            self.doubleSpinBox_fret.setReadOnly(True)
            self.doubleSpinBox_fret.setMaximum(1.0)
            self.gridLayout_7.addWidget(self.label_fret, 3, 0)
            self.gridLayout_7.addWidget(self.doubleSpinBox_fret, 3, 1)
        except AttributeError:
            pass

        try:
            self.actionSave_project.triggered.connect(self.onSaveProject)
            self.actionLoad_project.triggered.connect(self.onLoadProject)
        except AttributeError:
            pass

    def _build_molview_widget(self):
        """Construct 3D viewer based on ChiSurf setting (pymol|chimol)."""
        if self.viewer_backend == "chimol" and _HAS_CHIMOL:
            return _ChimolAdapter(self)
        return MolQtWidget(self, enableUi=False)

    def onSaveHist(self, verbose=False):
        verbose = self.verbose or verbose
        filename = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File', '.')
        self.save_histogram(filename=filename, verbose=verbose)

    def molview_highlight_quencher(self):
        quencher = self.quencher
        pymol = self.molview.pymol
        for res_name in quencher:
            pymol.cmd.do("hide lines, resn %s" % res_name)
            pymol.cmd.do("show sticks, resn %s" % res_name)
            pymol.cmd.do("color red, resn %s" % res_name)

    def onLoadSettings(self):
        print("onLoadSettings")
        print("Setting File: %s" % self.settings_file)

    @property
    def settings_file(self):
        return self._settings_file

    @settings_file.setter
    def settings_file(self, v):
        self.lineEdit_2.setText(v)

    @property
    def load_3d_av(self):
        return bool(self.checkBox_3.isChecked())

    def update_3D(self, load_av=False):
        load_av = load_av or self.load_3d_av
        self.molview.reset()
        if self.pdb_filename is not None:
            self.molview.openFile(self._tmp_file, frame=1, mode='cartoon', object_name='p')
        self.molview.pymol.cmd.do("hide all")
        self.molview.pymol.cmd.do("show spheres, p")
        self.molview.pymol.cmd.do("color gray, p")

        # Remove previously drawn AV objects to prevent stale visuals
        try:
            self.molview.pymol.cmd.delete("av")
            self.molview.pymol.cmd.delete("sticky_av")
            self.molview.pymol.cmd.delete("acceptor_av")
            self.molview.pymol.cmd.delete("acceptor_av_slow")
        except Exception:
            pass

        if self.diff_file is not None:
            self.molview.openFile(self.diff_file, frame=1, object_name='trajectory')
            self.molview.pymol.cmd.do("color green, trajectory")
            self.molview.pymol.cmd.do("set sphere_scale, 0.2, trajectory")
        if load_av:
            if self.av_slow_file is not None:
                self.molview.openFile(self.av_slow_file, frame=1, object_name='sticky_av')
                self.molview.pymol.cmd.do("color cyan, sticky_av")
                self.molview.pymol.cmd.do("show spheres, sticky_av")

            if self.av_fast_file is not None:
                self.molview.openFile(self.av_fast_file, frame=1, object_name='av')
                self.molview.pymol.cmd.do("color cyan, av")
                self.molview.pymol.cmd.do("show spheres, av")

            if self.acceptor_av_slow_file is not None:
                self.molview.openFile(self.acceptor_av_slow_file, frame=1, object_name='acceptor_av_slow')
                self.molview.pymol.cmd.do("color magenta, acceptor_av_slow")
                self.molview.pymol.cmd.do("show spheres, acceptor_av_slow")

            if self.acceptor_av_fast_file is not None:
                self.molview.openFile(self.acceptor_av_fast_file, frame=1, object_name='acceptor_av')
                self.molview.pymol.cmd.do("color magenta, acceptor_av")
                self.molview.pymol.cmd.do("show spheres, acceptor_av")
        self.molview.pymol.cmd.orient()

        self.molview_highlight_quencher()

    def update_trajectory_curve(self):
        y = self.diffusion.distance_to_mean
        x = np.linspace(0, self.t_max, y.shape[0])
        self.diffusion_curve.setData(x, y)
        auto_corr = autocorr(y)
        self.diffusion_autocorrelation.setData(x, auto_corr)
        self.plot_autocorr.enableAutoRange(True, True)
        self.plot_autocorr.autoRange()
        self.plot_diffusion.enableAutoRange(True, True)
        self.plot_diffusion.autoRange()

    def update_decay_histogram(self):
        x, y = self.get_histogram(nbins=self.nBins)
        y[y < 0.001] = 0.001
        self.hist_curve.setData(x[1:], y + 1.0)
        yu = np.exp(-x/self.tau0) * y[1]
        self.unquenched_curve.setData(x, yu + 1.0)
        if getattr(self, "fret_checkbox", None) is not None and self.fret_checkbox.isChecked():
            xf, yf = self.get_histogram_fret(nbins=self.nBins)
            if xf is not None:
                yf[yf < 0.001] = 0.001
                self.fret_curve.setData(xf[1:], yf + 1.0)
        else:
            self.fret_curve.setData([], [])
        self.plot_decay.enableAutoRange(True, True)
        self.plot_decay.autoRange()

    def update_all(self, verbose=False):
        success = DonorDecay.update_all(self)
        if success:
            self.doubleSpinBox_16.setValue(self.quantum_yield)
            self.doubleSpinBox_15.setValue(self.collisions * 100.0)
            if getattr(self, "fret_checkbox", None) is not None and self.fret_checkbox.isChecked():
                self.fret_enabled = True
                self.acceptor_chain = getattr(self.acceptor_selector, "chain_id", None)
                self.acceptor_residue = getattr(self.acceptor_selector, "residue_id", None)
                self.acceptor_atom = getattr(self.acceptor_selector, "atom_name", None)
                # Sync acceptor AV parameters from the FRET group box controls
                try:
                    self.fret_av_parameter = {
                        'linker_length': float(self.fret_len_spin.value()),
                        'linker_width': float(self.fret_wid_spin.value()),
                        'radius1': float(self.fret_rad_spin.value()),
                    }
                except AttributeError:
                    self.fret_av_parameter = None
                self.calc_acceptor_av(verbose=verbose)
                self.acceptor_av_slow_file, self.acceptor_av_fast_file = self.save_av(
                    'acceptor',
                    self._acceptor_av,
                    directory=self.tmp_dir,
                    verbose=verbose,
                )
                self.calc_photons_fret(verbose=verbose)
                try:
                    if hasattr(self, "doubleSpinBox_fret") and self.fret_efficiency is not None:
                        self.doubleSpinBox_fret.setValue(self.fret_efficiency)
                except AttributeError:
                    pass
            else:
                self.fret_enabled = False
                self._photon_trace_fret = None
                self.acceptor_av_slow_file = None
                self.acceptor_av_fast_file = None
            diff_file, av_slow_file, av_fast_file = self.onSaveAV(directory=self.tmp_dir)
            self.diff_file = diff_file
            self.av_slow_file = av_slow_file
            self.av_fast_file = av_fast_file
            self.update_3D()
            self.update_decay_histogram()
            self.update_trajectory_curve()
            # After a successful simulation, show the Decay tab (index 1)
            try:
                self.tabWidget.setCurrentIndex(1)
            except Exception:
                pass
        else:
            # Display a message box in case of failure
            msg_box = QtWidgets.QMessageBox()
            msg_box.setIcon(QtWidgets.QMessageBox.Warning)
            msg_box.setWindowTitle("Trajectory Calculation Failed")
            msg_box.setText("The trajectory calculation failed. Possible reason: Inaccessible labeling location.")
            msg_box.setStandardButtons(QtWidgets.QMessageBox.Ok)
            msg_box.exec_()

    def save_av(self, dye_name, av_obj, verbose=False, directory=None):
        if av_obj is None:
            return None, None
        verbose = self.verbose or verbose
        if directory is None:
            directory = self.tmp_dir if hasattr(self, "tmp_dir") and self.tmp_dir else tempfile.gettempdir()
        safe_name = (dye_name or "").strip().lower()
        suffix = f"_{safe_name}" if safe_name else ""
        base = os.path.join(directory, f"{self.filename_prefix}{suffix}")
        slow_file = base + '_av_slow.xyz'
        fast_file = base + '_av_fast.xyz'
        if verbose:
            print(f"Saving {dye_name or 'AV'} files to {directory}")
            print("  slow AV ->", slow_file)
            print("  fast AV ->", fast_file)
        write_xyz(slow_file, av_obj.points_slow)
        write_xyz(fast_file, av_obj.points_fast)
        return slow_file, fast_file

    def onSaveAV(self, verbose=False, directory=None):
        verbose = self.verbose or verbose
        if verbose:
            print("\nWriting AVs to directory")
            print("-------------------------")
        if directory is None:
            directory = str(QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose directory', '.'))
        if verbose:
            print("Directory: %s" % directory)
            print("Filename-Prefix: %s" % str(self.filename_prefix))
        diff_file = os.path.join(directory, self.filename_prefix+self.diffusion.trajectory_suffix)
        if verbose:
            print("Saving trajectory...")
            print("Trajectory filename: %s" % diff_file)
            print("Skipping every %i frame." % self.skip_frame)
        self.diffusion.save_trajectory(diff_file, self.skip_frame)

        donor_slow, donor_fast = self.save_av('donor', self.av, verbose=verbose, directory=directory)
        return diff_file, donor_slow, donor_fast

    def to_project_dict(self):
        project = OrderedDict()

        # Input data
        project['pdb'] = self.pdb_filename or None
        project['attachment'] = {
            'chain': self.attachment_chain,
            'residue': self.attachment_residue,
            'atom': self.attachment_atom,
        }

        # Accessible volume / dye parameters
        project['av_parameter'] = self.av_parameter
        project['dg'] = self.dg
        project['slow_radius'] = self.slow_radius

        # Simulation settings
        project['t_max'] = self.t_max
        project['t_step'] = self.t_step
        project['n_photons'] = self.n_photons
        project['n_bins'] = self.nBins
        project['skip_frame'] = self.skip_frame

        # Photophysics / quenching
        project['tau0'] = self.tau0
        project['kQ'] = self.kQ
        project['slow_fact'] = self.slow_fact
        project['critical_distance'] = self.critical_distance
        project['quencher'] = self.quencher

        # Output / bookkeeping
        project['output_file'] = self.filename_prefix
        project['save_avs'] = self.save_avs
        project['all_quencher_atoms'] = self.all_quencher_atoms
        project['sticky_mode'] = self.sticky_mode

        fret = OrderedDict()
        fret['enabled'] = bool(getattr(self, "fret_checkbox", None) is not None and self.fret_checkbox.isChecked())

        dyes = []

        donor = OrderedDict()
        donor_attachment = OrderedDict()
        donor_attachment['chain'] = self.attachment_chain
        donor_attachment['residue'] = self.attachment_residue
        donor_attachment['atom'] = self.attachment_atom
        donor['name'] = 'D0'
        donor['role'] = 'donor'
        donor['attachment'] = donor_attachment
        donor['av_parameter'] = self.av_parameter
        # Store donor diffusion coefficient explicitly with the donor dye
        donor['D'] = float(self.diffusion_coefficient)
        dyes.append(donor)

        acceptor = OrderedDict()
        acceptor_attachment = OrderedDict()
        if hasattr(self, "acceptor_selector"):
            acceptor_attachment['chain'] = self.acceptor_selector.chain_id
            acceptor_attachment['residue'] = self.acceptor_selector.residue_id
            acceptor_attachment['atom'] = self.acceptor_selector.atom_name
        else:
            acceptor_attachment['chain'] = None
            acceptor_attachment['residue'] = None
            acceptor_attachment['atom'] = None
        acceptor['name'] = 'A0'
        acceptor['role'] = 'acceptor'
        acceptor['attachment'] = acceptor_attachment

        # Ensure FRET AV parameters reflect the current UI when FRET is enabled
        if getattr(self, "fret_checkbox", None) is not None and self.fret_checkbox.isChecked():
            try:
                self.fret_av_parameter = {
                    'linker_length': float(self.fret_len_spin.value()),
                    'linker_width': float(self.fret_wid_spin.value()),
                    'radius1': float(self.fret_rad_spin.value()),
                }
            except AttributeError:
                self.fret_av_parameter = None
        else:
            self.fret_av_parameter = None

        acceptor['av_parameter'] = self.fret_av_parameter if self.fret_av_parameter is not None else self.av_parameter

        # Store acceptor diffusion coefficient; default to donor D if no
        # separate value is provided.
        acc_D = None
        try:
            if hasattr(self, "fret_D_spin"):
                acc_D = float(self.fret_D_spin.value())
        except Exception:
            acc_D = None
        if acc_D is None:
            acc_D = float(self.diffusion_coefficient)
        acceptor['D'] = acc_D
        dyes.append(acceptor)

        fret['dyes'] = dyes

        if self.fret_R0 is not None:
            R0_matrix = [
                [None, self.fret_R0],
                [self.fret_R0, None]
            ]
        else:
            R0_matrix = [
                [None, None],
                [None, None]
            ]
        fret['R0_matrix'] = R0_matrix

        project['fret'] = fret

        return project

    def load_project_dict(self, settings):
        if settings is None:
            return

        # Input data
        pdb_path = settings.get('pdb')
        if pdb_path:
            self.pdb_filename = pdb_path
            try:
                self.structure = pdb_path
                self.pdb_selector.atoms = self.structure.atoms
                if hasattr(self, "acceptor_selector"):
                    try:
                        self.acceptor_selector.atoms = self.structure.atoms
                    except Exception:
                        pass
                self.tmp_dir = tempfile.gettempdir()
                self.update_3D()
            except Exception as e:
                print("Failed to load PDB from project:", e)

        attachment = settings.get('attachment', {})
        try:
            self.attachment_chain = attachment.get('chain', self.attachment_chain)
            self.attachment_residue = attachment.get('residue', self.attachment_residue)
            self.attachment_atom = attachment.get('atom', self.attachment_atom)
        except Exception as e:
            print("Failed to restore attachment from project:", e)

        # Accessible volume / dye parameters
        if 'av_parameter' in settings:
            try:
                self.av_parameter = settings['av_parameter']
            except Exception as e:
                print("Failed to restore av_parameter from project:", e)

        self.dg = settings.get('dg', self.dg)
        self.slow_radius = settings.get('slow_radius', self.slow_radius)

        # Simulation settings
        self.t_max = settings.get('t_max', self.t_max)
        self.t_step = settings.get('t_step', self.t_step)
        self.n_photons = settings.get('n_photons', self.n_photons)
        try:
            if 'n_bins' in settings:
                self.spinBox_2.setValue(int(settings['n_bins']))
        except AttributeError:
            pass
        try:
            if 'skip_frame' in settings:
                self.spinBox_3.setValue(int(settings['skip_frame']))
        except AttributeError:
            pass

        # Photophysics / quenching
        self.tau0 = settings.get('tau0', self.tau0)

        # Donor diffusion coefficient: prefer per-dye value in fret.dyes[0]['D']
        donor_D = None
        fret_settings_for_D = settings.get('fret')
        if isinstance(fret_settings_for_D, dict):
            dyes_for_D = fret_settings_for_D.get('dyes')
            if isinstance(dyes_for_D, (list, tuple)) and len(dyes_for_D) > 0:
                donor_D = dyes_for_D[0].get('D', None)
        if donor_D is None:
            # Legacy projects may still store a top-level 'D'
            donor_D = settings.get('D', self.diffusion_coefficient)
        self.diffusion_coefficient = donor_D

        self.kQ = settings.get('kQ', self.kQ)
        self.slow_fact = settings.get('slow_fact', self.slow_fact)
        self.critical_distance = settings.get('critical_distance', self.critical_distance)
        if 'quencher' in settings:
            self.quencher = settings['quencher']

        # Output / bookkeeping
        if 'output_file' in settings:
            try:
                self.lineEdit_5.setText(str(settings['output_file']))
            except AttributeError:
                pass
        self.save_avs = settings.get('save_avs', self.save_avs)
        self.all_quencher_atoms = settings.get('all_quencher_atoms', self.all_quencher_atoms)
        self.sticky_mode = settings.get('sticky_mode', self.sticky_mode)

        fret_settings = settings.get('fret')
        if fret_settings:
            try:
                if getattr(self, "fret_checkbox", None) is not None:
                    self.fret_checkbox.setChecked(bool(fret_settings.get('enabled', self.fret_checkbox.isChecked())))
            except AttributeError:
                pass

            dyes = fret_settings.get('dyes')
            R0_matrix = fret_settings.get('R0_matrix')

            if isinstance(dyes, (list, tuple)) and len(dyes) >= 2:
                acceptor = dyes[1]
                acc_att = acceptor.get('attachment', {})
                try:
                    if hasattr(self, "acceptor_selector"):
                        if 'chain' in acc_att:
                            self.acceptor_selector.chain_id = acc_att.get('chain')
                        if 'residue' in acc_att:
                            self.acceptor_selector.residue_id = acc_att.get('residue')
                        if 'atom' in acc_att:
                            self.acceptor_selector.atom_name = acc_att.get('atom')
                except Exception:
                    pass
                av_param_acc = acceptor.get('av_parameter')
                if av_param_acc is not None:
                    self.fret_av_parameter = av_param_acc
                    try:
                        self.fret_len_spin.setValue(float(av_param_acc.get('linker_length', self.doubleSpinBox.value())))
                        self.fret_wid_spin.setValue(float(av_param_acc.get('linker_width', self.doubleSpinBox_2.value())))
                        self.fret_rad_spin.setValue(float(av_param_acc.get('radius1', self.doubleSpinBox_3.value())))
                    except AttributeError:
                        pass

                # Restore acceptor diffusion coefficient if present
                try:
                    acc_D = acceptor.get('D', None)
                    if acc_D is not None and hasattr(self, "fret_D_spin"):
                        self.fret_D_spin.setValue(float(acc_D))
                except Exception:
                    pass

            if isinstance(R0_matrix, (list, tuple)) and len(R0_matrix) > 0:
                row0 = R0_matrix[0]
                if isinstance(row0, (list, tuple)) and len(row0) > 1:
                    self.fret_R0 = row0[1]

    def onSaveProject(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            'Save project',
            '.',
            'QuEst project (*.quest.json);;JSON (*.json);;All files (*)'
        )
        if not filename:
            return
        project = self.to_project_dict()
        try:
            with open(filename, 'w') as fp:
                json.dump(project, fp, indent=2, sort_keys=True)
            print("Saved project to", filename)
        except IOError as e:
            msg_box = QtWidgets.QMessageBox(self)
            msg_box.setIcon(QtWidgets.QMessageBox.Warning)
            msg_box.setWindowTitle("Save project failed")
            msg_box.setText("Could not write project file:\n%s" % e)
            msg_box.exec_()

    def onLoadProject(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            'Load project',
            '.',
            'QuEst project (*.quest.json);;JSON (*.json);;All files (*)'
        )
        if not filename:
            return
        try:
            with open(filename) as fp:
                settings = json.load(fp, object_pairs_hook=OrderedDict)
        except (IOError, ValueError) as e:
            msg_box = QtWidgets.QMessageBox(self)
            msg_box.setIcon(QtWidgets.QMessageBox.Warning)
            msg_box.setWindowTitle("Load project failed")
            msg_box.setText("Could not read project file:\n%s" % e)
            msg_box.exec_()
            return

        self.load_project_dict(settings)
        # Optionally, update all outputs based on loaded settings
        # but do not run a full simulation automatically to keep CLI parity

    @property
    def all_quencher_atoms(self):
        return not bool(self.groupBox_5.isChecked())

    @all_quencher_atoms.setter
    def all_quencher_atoms(self, v):
        try:
            self.groupBox_5.setChecked(not v)
        except AttributeError:
            pass

    @property
    def filename_prefix(self):
        return str(self.lineEdit_5.text())

    @property
    def skip_frame(self):
        return int(self.spinBox_3.value())

    @property
    def n_frames(self):
        return int(self.spinBox.value())

    def onSimulationDtChanged(self):
        time_steps = self.t_max / self.t_step
        self.spinBox.setValue(int(time_steps))

    def onSimulationTimeChanged(self):
        time_steps = self.t_max / self.t_step
        self.spinBox.setValue(int(time_steps))

    def onLoadPDB(self):
        self.pdb_filename = str(QtWidgets.QFileDialog.getOpenFileName(None, 'Open PDB-File', '', 'PDB-files (*.pdb)')[0])
        self.structure = self.pdb_filename
        self.pdb_selector.atoms = self.structure.atoms
        # Keep acceptor selector in sync with the donor when available
        if hasattr(self, "acceptor_selector"):
            try:
                self.acceptor_selector.atoms = self.structure.atoms
            except Exception:
                pass
        self.tmp_dir = tempfile.gettempdir()
        self.update_3D()
        # When a new PDB is opened, switch to the 3D view tab (index 2)
        try:
            self.tabWidget.setCurrentIndex(2)
        except Exception:
            pass

    def onFretToggled(self, checked):
        """Handle enabling/disabling of the FRET acceptor controls.

        When enabled, ensure the acceptor selector is populated with the
        current structure atoms so that chain/residue/atom choices match
        the donor selector.
        """

        if not hasattr(self, "acceptor_selector"):
            return

        # Enable/disable the selector and AV parameter widgets
        for w in (
            getattr(self, "acceptor_selector", None),
            getattr(self, "fret_len_spin", None),
            getattr(self, "fret_wid_spin", None),
            getattr(self, "fret_rad_spin", None),
            getattr(self, "fret_dg_spin", None),
            getattr(self, "fret_D_spin", None),
        ):
            if w is not None:
                try:
                    w.setEnabled(checked)
                except Exception:
                    pass

        # When enabling, ensure the selector is populated once a structure
        # is available so that the user can pick residues/atoms immediately.
        if checked and getattr(self, "structure", None) is not None:
            try:
                if getattr(self.acceptor_selector, "atoms", None) is None:
                    self.acceptor_selector.atoms = self.structure.atoms
            except Exception:
                pass

    def onFretDgChanged(self, value):
        """Keep donor and acceptor dg parameters in sync."""

        try:
            self.doubleSpinBox_17.setValue(float(value))
        except Exception:
            pass

    @property
    def dg(self):
        return float(self.doubleSpinBox_17.value())

    @dg.setter
    def dg(self, v):
        try:
            v_f = float(v)
            self.doubleSpinBox_17.setValue(v_f)
            if hasattr(self, "fret_dg_spin"):
                self.fret_dg_spin.setValue(v_f)
        except AttributeError:
            pass

    @property
    def slow_radius(self):
        return float(self.doubleSpinBox_10.value())

    @slow_radius.setter
    def slow_radius(self, v):
        try:
            self.doubleSpinBox_10.setValue(v)
        except AttributeError:
            pass

    @property
    def nBins(self):
        return int(self.spinBox_2.value())

    @property
    def n_photons(self):
        return int(self.doubleSpinBox_11.value() * 1e6)

    @n_photons.setter
    def n_photons(self, v):
        try:
            self.doubleSpinBox_11.setValue(v / 1e6)
        except AttributeError:
            pass

    @property
    def quencher(self):
        """
        dict of atom names, dict keys are residue types
        """
        p = OrderedDict()
        for qn in str(self.lineEdit_3.text()).split():
            p[qn] = ['CB']
        return p

    @quencher.setter
    def quencher(self, v):
        s = ""
        for resID in list(v.keys()):
            s += " "+resID
        try:
            self.lineEdit_3.setText(s)
        except AttributeError:
            pass

    @property
    def sticky_mode(self):
        if self.radioButton.isChecked():
            return 'surface'
        elif self.radioButton_2.isChecked():
            return 'quencher'

    @sticky_mode.setter
    def sticky_mode(self, v):
        try:
            print("set sticky: ")
            if v == 'surface':
                print("surface")
                self.radioButton.setChecked(True)
                self.radioButton_2.setChecked(False)
            elif v == 'quencher':
                print("quencher")
                self.radioButton.setChecked(False)
                self.radioButton_2.setChecked(True)
        except AttributeError:
            pass

    @property
    def av_parameter(self):
        p = OrderedDict()
        p['linker_length'] = float(self.doubleSpinBox.value())
        p['linker_width'] = float(self.doubleSpinBox_2.value())
        p['radius1'] = float(self.doubleSpinBox_3.value())
        return p

    @av_parameter.setter
    def av_parameter(self, d):
        try:
            self.doubleSpinBox.setValue(d['linker_length'])
            self.doubleSpinBox_2.setValue(d['linker_width'])
            self.doubleSpinBox_3.setValue(d['radius1'])
        except AttributeError:
            pass

    @property
    def critical_distance(self):
        return float(self.doubleSpinBox_12.value())

    @critical_distance.setter
    def critical_distance(self, v):
        try:
            self.doubleSpinBox_12.setValue(float(v))
        except AttributeError:
            pass

    @property
    def slow_fact(self):
        return float(self.doubleSpinBox_5.value())

    @slow_fact.setter
    def slow_fact(self, v):
        try:
            self.doubleSpinBox_5.setValue(float(v))
        except AttributeError:
            pass

    @property
    def t_max(self):
        """
        simulation time in nano-seconds
        """
        return float(self.doubleSpinBox_6.value()) * 1000.0

    @t_max.setter
    def t_max(self, v):
        try:
            self.onSimulationTimeChanged()
            self.doubleSpinBox_6.setValue(float(v / 1000.0))
        except AttributeError:
            pass

    @property
    def t_step(self):
        """
        time-step in nano-seconds
        """
        return float(self.doubleSpinBox_7.value()) / 1000.0

    @t_step.setter
    def t_step(self, v):
        try:
            self.onSimulationTimeChanged()
            self.doubleSpinBox_7.setValue(float(v * 1000.0))
        except AttributeError:
            pass

    @property
    def diffusion_coefficient(self):
        return float(self.doubleSpinBox_4.value())

    @diffusion_coefficient.setter
    def diffusion_coefficient(self, v):
        try:
            self.doubleSpinBox_4.setValue(v)
        except AttributeError:
            pass

    @property
    def kQ(self):
        return float(self.doubleSpinBox_9.value())

    @kQ.setter
    def kQ(self, v):
        try:
            self.doubleSpinBox_9.setValue(float(v))
        except AttributeError:
            pass

    @property
    def tau0(self):
        return float(self.doubleSpinBox_8.value())

    @tau0.setter
    def tau0(self, v):
        try:
            self.doubleSpinBox_8.setValue(float(v))
        except AttributeError:
            pass

    @property
    def attachment_chain(self):
        return self.pdb_selector.chain_id

    @attachment_chain.setter
    def attachment_chain(self, v):
        try:
            self.pdb_selector.chain_id = v
        except AttributeError:
            pass

    @property
    def attachment_residue(self):
        return self.pdb_selector.residue_id

    @attachment_residue.setter
    def attachment_residue(self, v):
        try:
            self.pdb_selector.residue_id = v
        except AttributeError:
            pass

    @property
    def attachment_atom(self):
        return self.pdb_selector.atom_name

    @attachment_atom.setter
    def attachment_atom(self, v):
        try:
            self.pdb_selector.atom_name = v
        except AttributeError:
            pass

    @property
    def pdb_filename(self):
        return str(self.lineEdit.text())

    @pdb_filename.setter
    def pdb_filename(self, value):
        try:
            self.lineEdit.setText(value)
        except AttributeError:
            pass


