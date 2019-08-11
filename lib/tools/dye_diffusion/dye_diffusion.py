import time
import os
from collections import OrderedDict
import gc
import json
import tempfile

import numpy as np
import numexpr as ne
from PyQt4 import QtCore, QtGui, uic
from guiqwt.plot import CurveDialog
from guiqwt.builder import make

from . import photon
import lib.io.PDB as PDB
from lib import Structure
import lib.fps as fps
from lib.plots.MolView import MolQtWidget
from lib.math.functions import autocorr
from lib.widgets import PDBSelector
from lib.fps import AV, write_xyz


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
        start = time.clock()
        traj, a, n_accepted, n_rejected = fps.simulate_traj(density, slow_density, dg,
                                                          slow_fact=slow_fact, t_step=t_step, t_max=t_max, D=D)
        traj += self.x0
        self._traj = traj

        if verbose:
            print("Accepted steps: %i" % n_accepted)
            print("Rejected steps: %i" % n_rejected)
        end = time.clock()
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
        print self._tmp_file
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
        self.simulate_diffusion(verbose=verbose)
        if verbose:
            print("Determining quencher distances.")
        self.diffusion.critical_distance = self.critical_distance
        if verbose:
            print("Simulating decay")
        self.calc_photons(verbose=verbose)


class TransientDecayGenerator(QtGui.QWidget, DonorDecay):

    name = "Lifetime-Calculator"

    def __init__(self, verbose=False, settings_file='./settings/dye_diffusion.json'):
        QtGui.QWidget.__init__(self)
        uic.loadUi('./lib/ui/dye_diffusion2.ui', self)
        self.pdb_selector = PDBSelector()
        self.verticalLayout_10.addWidget(self.pdb_selector)
        self._settings_file = None
        self.settings_file = settings_file
        fp = open(settings_file)
        settings = json.load(fp)
        fp.close()

        DonorDecay.__init__(self, **settings)
        ## User-interface
        self.connect(self.actionLoad_PDB, QtCore.SIGNAL('triggered()'), self.onLoadPDB)
        self.connect(self.actionLoad_settings, QtCore.SIGNAL('triggered()'), self.onLoadSettings)

        self.connect(self.doubleSpinBox_6, QtCore.SIGNAL("valueChanged(double)"), self.onSimulationTimeChanged)
        self.connect(self.doubleSpinBox_7, QtCore.SIGNAL("valueChanged(double)"), self.onSimulationDtChanged)
        self.connect(self.pushButton_3, QtCore.SIGNAL("clicked()"), self.update_all)
        self.connect(self.pushButton_4, QtCore.SIGNAL("clicked()"), self.onSaveHist)
        self.connect(self.pushButton_5, QtCore.SIGNAL("clicked()"), self.onSaveAV)

        self.tmp_dir = tempfile.gettempdir()
        print("Temporary Directory: %s" % self.tmp_dir)

        ## Decay-Curve
        fd = CurveDialog(edit=False, toolbar=True)
        self.plot_decay = fd.get_plot()
        self.hist_curve = make.curve([1],  [1], color="r", linewidth=1)
        self.unquenched_curve = make.curve([1],  [1], color="b", linewidth=1)
        self.plot_decay.add_item(self.hist_curve)
        self.plot_decay.add_item(self.unquenched_curve)
        self.plot_decay.set_scales('lin', 'log')
        self.verticalLayout_2.addWidget(fd)

        ## Diffusion-Trajectory-Curve
        options = dict(title="Trajectory", xlabel="time [ns]", ylabel=("|R-<R>|"))
        fd = CurveDialog(edit=False, toolbar=True, options=options)
        self.plot_diffusion = fd.get_plot()
        self.diffusion_curve = make.curve([1],  [1], color="b", linewidth=1)
        self.plot_diffusion.add_item(self.diffusion_curve)
        self.plot_diffusion.set_scales('lin', 'lin')
        self.verticalLayout_6.addWidget(fd)

        options = dict(xlabel="corr. time [ns]", ylabel=("A.Corr.(|R-<R>|)"))
        fd = CurveDialog(edit=False, toolbar=True, options=options)
        self.plot_autocorr = fd.get_plot()
        self.diffusion_autocorrelation = make.curve([1],  [1], color="r", linewidth=1)
        self.plot_autocorr.add_item(self.diffusion_autocorrelation)
        self.plot_autocorr.set_scales('log', 'lin')
        self.verticalLayout_6.addWidget(fd)

        ## Protein Structure
        self.molview = MolQtWidget(self, enableUi=False)
        self.verticalLayout_4.addWidget(self.molview)

        self.diff_file = None
        self.av_slow_file = None
        self.av_fast_file = None

    def onSaveHist(self, verbose=False):
        verbose = self.verbose or verbose
        filename = QtGui.QFileDialog.getSaveFileName(self, 'Save File', '.')
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
        return self.lineEdit_2.setText(v)

    @property
    def load_3d_av(self):
        return bool(self.checkBox_3.isChecked())

    def update_3D(self, load_av=False):
        load_av = load_av or self.load_3d_av
        self.molview.reset()
        if self.pdb_filename is not None:
            self.molview.openFile(self._tmp_file, frame=1, mode='cartoon', object_name='protein')
        self.molview.pymol.cmd.do("hide all")
        self.molview.pymol.cmd.do("show spheres, protein")
        self.molview.pymol.cmd.do("color gray, protein")

        if self.diff_file is not None:
            self.molview.openFile(self.diff_file, frame=1, object_name='trajectory')
            self.molview.pymol.cmd.do("color green, trajectory")
        if load_av:
            if self.av_slow_file is not None:
                self.molview.openFile(self.av_slow_file, frame=1, object_name='sticky_av')

            if self.av_fast_file is not None:
                self.molview.openFile(self.av_fast_file, frame=1, object_name='av')
        self.molview.pymol.cmd.orient()

        self.molview_highlight_quencher()

    def update_trajectory_curve(self):
        y = self.diffusion.distance_to_mean
        x = np.linspace(0, self.t_max, y.shape[0])
        self.diffusion_curve.set_data(x, y)
        auto_corr = autocorr(y)
        self.diffusion_autocorrelation.set_data(x, auto_corr)
        self.plot_autocorr.do_autoscale()
        self.plot_diffusion.do_autoscale()

    def update_decay_histogram(self):
        x, y = self.get_histogram(nbins=self.nBins)
        y[y < 0.001] = 0.001
        self.hist_curve.set_data(x[1:], y + 1.0)
        yu = np.exp(-x/self.tau0) * y[1]
        self.unquenched_curve.set_data(x, yu + 1.0)
        self.plot_decay.do_autoscale()

    def update_all(self, verbose=False):
        DonorDecay.update_all(self)
        self.doubleSpinBox_16.setValue(self.quantum_yield)
        self.doubleSpinBox_15.setValue(self.collisions * 100.0)
        diff_file, av_slow_file, av_fast_file = self.onSaveAV(directory=self.tmp_dir)
        self.diff_file = diff_file
        self.av_slow_file = av_slow_file
        self.av_fast_file = av_fast_file
        self.update_3D()
        self.update_decay_histogram()
        self.update_trajectory_curve()

    def onSaveAV(self, verbose=False, directory=None):
        verbose = self.verbose or verbose
        if verbose:
            print("\nWriting AVs to directory")
            print("-------------------------")
        if directory is None:
            directory = str(QtGui.QFileDialog.getExistingDirectory(self, 'Choose directory', '.'))
        if verbose:
            print("Directory: %s" % directory)
            print("Filename-Prefix: %s" % str(self.filename_prefix))
        diff_file = os.path.join(directory, self.filename_prefix+self.diffusion.trajectory_suffix)
        if verbose:
            print("Saving trajectory...")
            print("Trajectory filename: %s" % diff_file)
            print("Skipping every %i frame." % self.skip_frame)
        self.diffusion.save_trajectory(diff_file, self.skip_frame)

        av_slow_file = os.path.join(directory, self.filename_prefix + '_av_slow.xyz')
        if verbose:
            print("\nSaving slow AV...")
            print("Trajectory filename: %s" % av_slow_file)
        write_xyz(av_slow_file, self.av.points_slow)

        av_fast_file = os.path.join(directory, self.filename_prefix + '_av_fast.xyz')
        if verbose:
            print("\nSaving slow AV...")
            print("Trajectory filename: %s" % av_fast_file)
        write_xyz(av_fast_file, self.av.points_fast)
        return diff_file, av_slow_file, av_fast_file

    @property
    def all_quencher_atoms(self):
        return not bool(self.groupBox_5.isChecked())

    @all_quencher_atoms.setter
    def all_quencher_atoms(self, v):
        self.groupBox_5.setChecked(not v)

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
        self.pdb_filename = str(QtGui.QFileDialog.getOpenFileName(None, 'Open PDB-File', '', 'PDB-files (*.pdb)'))
        self.structure = self.pdb_filename
        self.pdb_selector.atoms = self.structure.atoms
        self.tmp_dir = tempfile.gettempdir()
        self.update_3D()

    @property
    def dg(self):
        return float(self.doubleSpinBox_17.value())

    @dg.setter
    def dg(self, v):
        self.doubleSpinBox_17.setValue(float(v))

    @property
    def slow_radius(self):
        return float(self.doubleSpinBox_10.value())

    @slow_radius.setter
    def slow_radius(self, v):
        self.doubleSpinBox_10.setValue(v)

    @property
    def nBins(self):
        return int(self.spinBox_2.value())

    @property
    def n_photons(self):
        return int(self.doubleSpinBox_11.value() * 1e6)

    @n_photons.setter
    def n_photons(self, v):
        return self.doubleSpinBox_11.setValue(v / 1e6)

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
        self.lineEdit_3.setText(s)

    @property
    def sticky_mode(self):
        if self.radioButton.isChecked():
            return 'surface'
        elif self.radioButton_2.isChecked():
            return 'quencher'

    @sticky_mode.setter
    def sticky_mode(self, v):
        print("set sticky: ")
        if v == 'surface':
            print("surface")
            self.radioButton.setChecked(True)
            self.radioButton_2.setChecked(False)
        elif v == 'quencher':
            print("quencher")
            self.radioButton.setChecked(False)
            self.radioButton_2.setChecked(True)

    @property
    def av_parameter(self):
        p = OrderedDict()
        p['linker_length'] = float(self.doubleSpinBox.value())
        p['linker_width'] = float(self.doubleSpinBox_2.value())
        p['radius1'] = float(self.doubleSpinBox_3.value())
        return p

    @av_parameter.setter
    def av_parameter(self, d):
        self.doubleSpinBox.setValue(d['linker_length'])
        self.doubleSpinBox_2.setValue(d['linker_width'])
        self.doubleSpinBox_3.setValue(d['radius1'])

    @property
    def critical_distance(self):
        return float(self.doubleSpinBox_12.value())

    @critical_distance.setter
    def critical_distance(self, v):
        self.doubleSpinBox_12.setValue(float(v))

    @property
    def slow_fact(self):
        return float(self.doubleSpinBox_5.value())

    @slow_fact.setter
    def slow_fact(self, v):
        self.doubleSpinBox_5.setValue(float(v))

    @property
    def t_max(self):
        """
        simulation time in nano-seconds
        """
        return float(self.doubleSpinBox_6.value()) * 1000.0

    @t_max.setter
    def t_max(self, v):
        self.onSimulationTimeChanged()
        self.doubleSpinBox_6.setValue(float(v / 1000.0))

    @property
    def t_step(self):
        """
        time-step in nano-seconds
        """
        return float(self.doubleSpinBox_7.value()) / 1000.0

    @t_step.setter
    def t_step(self, v):
        self.onSimulationTimeChanged()
        return self.doubleSpinBox_7.setValue(float(v * 1000.0))

    @property
    def diffusion_coefficient(self):
        return float(self.doubleSpinBox_4.value())

    @diffusion_coefficient.setter
    def diffusion_coefficient(self, v):
        self.doubleSpinBox_4.setValue(v)

    @property
    def kQ(self):
        return float(self.doubleSpinBox_9.value())

    @kQ.setter
    def kQ(self, v):
        return self.doubleSpinBox_9.setValue(float(v))

    @property
    def tau0(self):
        return float(self.doubleSpinBox_8.value())

    @tau0.setter
    def tau0(self, v):
        return self.doubleSpinBox_8.setValue(float(v))

    @property
    def attachment_chain(self):
        return self.pdb_selector.chain_id

    @attachment_chain.setter
    def attachment_chain(self, v):
        self.pdb_selector.chain_id = v

    @property
    def attachment_residue(self):
        return self.pdb_selector.residue_id

    @attachment_residue.setter
    def attachment_residue(self, v):
        self.pdb_selector.residue_id = v

    @property
    def attachment_atom(self):
        return self.pdb_selector.atom_name

    @attachment_atom.setter
    def attachment_atom(self, v):
        self.pdb_selector.atom_name = v

    @property
    def pdb_filename(self):
        return str(self.lineEdit.text())

    @pdb_filename.setter
    def pdb_filename(self, value):
        self.lineEdit.setText(value)


