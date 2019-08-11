import time
import os
import copy
import tempfile
import json

from PyQt4 import QtCore, QtGui, uic
import numpy as np

import mfm
from mfm import Structure
from mfm.math.rand import mc
from mfm.fitting.models import Model
from mfm.structure.potential import potentials
from mfm.structure.trajectory import TrajectoryFile, Universe
import mfm.structure
import mfm.math.rand as mrand
from mfm import plots


class X(QtCore.QObject):

    def emitMySignal(self, xyz, energy, fret_energy, elapsed):
        self.emit(QtCore.SIGNAL('newStructure'), xyz, energy, fret_energy, elapsed)


class ProteinMCWorker(QtCore.QThread):

    daemonIsRunning = False
    daemonStopSignal = False
    x = X()

    def __init__(self, parent, **kwargs):
        QtCore.QThread.__init__(self, parent)
        self.parent = parent
        self.exiting = False
        self.verbose = kwargs.get('verbose', mfm.verbose)

    def monteCarlo2U(self, verbose=True):
        verbose = verbose or self.verbose

        p = self.p
        p.structure.auto_update =  False
        p.structure.update()
        moveMap = np.array(p.movemap)
        nRes = int(p.structure.n_residues)

        if verbose:
            print("monteCarlo2U")
            print("moveMap: %s" % moveMap)
            print("Universe(1)-Energy: %s" % p.u1.getEnergy(p.structure))
            print("Universe(2)-Energy: %s" % p.u2.getEnergy(p.structure))
            print("nRes: %s" % nRes)
            print("nOut: %s" % p.pdb_nOut)

        # Get paramters from parent object
        scale = float(p.scale)
        ns = int(p.number_of_moving_aa)
        s10 = p.structure
        s10.auto_update = False
        av_number_protein_mc = int(p.av_number_protein_mc)

        start = time.time()
        elapsed = 0.0
        nAccepted = 0
        do_av_steepest_descent = p.do_av_steepest_descent

        cPhi = np.empty_like(s10.phi)
        cPsi = np.empty_like(s10.psi)
        cOmega = np.empty_like(s10.omega)
        cChi = np.empty_like(s10.chi)
        nChi = cChi.shape[0]

        e20 = p.u2.getEnergy(s10)
        e10 = p.u1.getEnergy(s10)

        coord_back_1 = np.empty_like(s10.internal_coordinates)
        coord_back_2 = np.empty_like(s10.internal_coordinates)

        while not self.daemonStopSignal:
            elapsed = (time.time() - start)
            # save configuration before inner MC-loop
            np.copyto(coord_back_2, s10.internal_coordinates)

            # inner Monte-Carlo loop
            # run av_number_protein_mc monte carlo-steps
            for trialRes in range(av_number_protein_mc):
                # decide which aa to move
                moving_aa = mrand.weighted_choice(moveMap, n=ns)
                # decide which angle to move
                move_phi, move_psi, move_ome, move_chi = np.random.ranf(4) < [p.pPhi, p.pPsi, p.pOmega, p.pChi]
                # save coordinates
                np.copyto(coord_back_1, s10.internal_coordinates)
                # move aa
                if move_phi:
                    cPhi *= 0.0
                    cPhi[moving_aa] += np.random.normal(0.0, scale, ns) * s10.phi[moving_aa]
                    s10.phi = (s10.phi + cPhi)
                if move_psi:
                    cPsi *= 0.0
                    cPsi[moving_aa] += np.random.normal(0.0, scale, ns) * s10.psi[moving_aa]
                    s10.psi = (s10.psi + cPsi)
                if move_ome:
                    cOmega *= 0.0
                    cOmega[moving_aa] += np.random.normal(0.0, scale, ns) * s10.omega[moving_aa]
                    s10.omega = (s10.omega + cOmega)
                if move_chi:
                    cChi *= 0.0
                    cChi[moving_aa % nChi] += np.random.normal(0.0, scale, ns) * s10.chi[moving_aa % nChi]
                    s10.chi = (s10.chi + cChi)

                # Monte-Carlo step
                s10.update(min(moving_aa))
                e11 = p.u1.getEnergy(s10)
                if mc(e10, e11, p.kt):
                    e10 = e11
                    np.copyto(coord_back_1, s10.internal_coordinates)
                else:
                    np.copyto(s10.internal_coordinates, coord_back_1)

            s10.update()
            e21 = p.u2.getEnergy(s10)
            accept = e21 < e20 if do_av_steepest_descent else mc(e20, e21, p.ktAv)

            if accept:
                # AV-MC accepted
                e20 = e21
                nAccepted += 1
                if nAccepted % p.pdb_nOut == 0:
                    self.x.emitMySignal(s10.xyz, e10, e20, elapsed)
            else:
                # AV-MC not accepted return to stored coordinates
                np.copyto(s10.internal_coordinates, coord_back_2)

    def monteCarlo1U(self, verbose=True):
        verbose = verbose or self.verbose
        p = self.p
        p.auto_update =  False

        p.structure.update()
        moveMap = np.array(p.movemap)
        nRes = int(self.p.structure.n_residues)
        if verbose:
            print("monteCarlo1U")
            print("moveMap: %s" % moveMap)
            print("Universe(1)-Energy: %s" % self.p.u1.getEnergy(self.p.structure))
            print("nRes: %s" % nRes)
            print("nOut: %s" % p.pdb_nOut)

        # Get paramters from parent object
        scale = float(p.scale)
        ns = int(p.number_of_moving_aa)
        s10 = p.structure
        s10.auto_update = False

        start = time.time()
        nAccepted = 0

        cPhi = np.empty_like(s10.phi)
        cPsi = np.empty_like(s10.psi)
        cOmega = np.empty_like(s10.omega)
        cChi = np.empty_like(s10.chi)
        nChi = cChi.shape[0]

        coord_back = np.empty_like(s10.internal_coordinates)
        np.copyto(coord_back, s10.internal_coordinates)

        e10 = self.p.u1.getEnergy(s10)
        self.x.emitMySignal(s10.xyz, e10, 0.0, 0.0)

        while not self.daemonStopSignal:
            elapsed = (time.time() - start)
            # decide which angle to move
            move_phi, move_psi, move_ome, move_chi = np.random.ranf(4) < [p.pPhi, p.pPsi, p.pOmega, p.pChi]
            # decide which aa to move
            moving_aa = mrand.weighted_choice(moveMap, n=ns)
            if move_phi:
                cPhi *= 0.0
                cPhi[moving_aa] += np.random.normal(0.0, scale, ns) * s10.phi[moving_aa]
                s10.phi = (s10.phi + cPhi)
            if move_psi:
                cPsi *= 0.0
                cPsi[moving_aa] += np.random.normal(0.0, scale, ns) * s10.psi[moving_aa]
                s10.psi = (s10.psi + cPsi)
            if move_ome:
                cOmega *= 0.0
                cOmega[moving_aa] += np.random.normal(0.0, scale, ns) * s10.omega[moving_aa]
                s10.omega = (s10.omega + cOmega)
            if move_chi:
                cChi *= 0.0
                cChi[moving_aa % nChi] += np.random.normal(0.0, scale, ns) * s10.chi[moving_aa % nChi]
                s10.chi = (s10.chi + cChi)

            # Monte-Carlo step
            s10.update(min(moving_aa))
            e11 = p.u1.getEnergy(s10)
            if mc(e10, e11, p.kt):
                e10 = e11
                nAccepted += 1
                if nAccepted % p.pdb_nOut == 0:
                    self.x.emitMySignal(s10.xyz, e11, 0.0, elapsed)
                np.copyto(coord_back, s10.internal_coordinates)
            else:
                np.copyto(s10.internal_coordinates, coord_back)

    def setDaemonStopSignal(self, bool):
        print("setDaemonStopSignal: %s" % bool)
        self.daemonStopSignal = bool

    def run(self):
        self.daemonIsRunning = True
        self.daemonStopSignal = False

        if self.parent.mc_mode == 'simple':
            self.monteCarlo1U()
        elif self.parent.mc_mode == 'av_mc':
            self.monteCarlo2U()


class ProteinMCWidget(QtGui.QWidget, TrajectoryFile, Model):

    name = "ProteinMC"

    plot_classes = [(plots.ProteinMCPlot, {}),
                    (plots.SurfacePlot, {}),
                    (plots.MolView, {'enableUi': False,
                                                 'mode': 'coarse',
                                                 'sequence': False})
    ]

    def get_config(self):
        parameter = dict()
        parameter['number_of_moving_aa'] = self.number_of_moving_aa
        parameter['save_filename'] = self.filename
        parameter['pPsi'] = self.pPsi
        parameter['pPhi'] = self.pPhi
        parameter['pOmega'] = self.pOmega
        parameter['pChi'] = self.pChi
        parameter['pdb_nOut'] = self.pdb_nOut
        parameter['av_number_protein_mc'] = self.av_number_protein_mc
        parameter['ktAv'] = self.ktAv
        parameter['kt'] = self.kt
        parameter['scale'] = self.scale
        parameter['mc_mode'] = self.mc_mode
        parameter['fps_file'] = self.fps_file
        parameter['movemap'] = list(self.movemap)
        parameter['do_av_steepest_descent'] = self.do_av_steepest_descent
        return parameter

    def set_config(self, **kwargs):
        self.number_of_moving_aa = kwargs.get('number_of_moving_aa', 1)
        self.filename = kwargs.get('save_filename', 'out.h5')
        self.pPsi = kwargs.get('pPsi', 0.3)
        self.pPhi = kwargs.get('pPhi', 0.7)
        self.pOmega = kwargs.get('pOmega', 0.00)
        self.pChi = kwargs.get('pChi', 0.01)
        self.pdb_nOut = kwargs.get('pdb_nOut', 5)
        self.av_number_protein_mc = kwargs.get('av_number_protein_mc', 50)
        self.ktAv = kwargs.get('ktAv', 1.0)
        self.kt = kwargs.get('kt', 1.5)
        self.scale = kwargs.get('scale', 0.0025)
        self.mc_mode = kwargs.get('mc_mode', 'av_mc')
        self.fps_file = kwargs.get('fps_file', None)
        self.movemap =  kwargs.get('movemap', None)
        self.do_av_steepest_descent = kwargs.get('do_av_steepest_descent', True)

    def __init__(self, fit, number_of_moving_aa=1, save_filename=None,
                 pPsi=0.3, pPhi=0.7, pOmega=0.00, pChi=0.01, pdbOut=5, av_number_protein_mc=50,
                 ktAv=1.0, kt=1.5, scale=0.0025, config_file=None, fps_file=None,
                 movemap=None, do_av_steepest_descent=False):
        """
        Parameters
        ----------
        :param config_file: string, optional
            Filename containing the parameters for the simulation. If this is specified all other parameters
            are taken from the configuration file and the passed parameters are overwritten.
        :param fps_file: string, optional
            Filename of the fps-labeling file (JSON-File).
        :param movemap: array of floats
            If specified should have length of residues. Specifies probability that dihedral of a certain amino-acid
            is changed.
        :param fit:
        :param number_of_moving_aa: int
            The number of amino-acids move in one Monte-Carlo step
        :param save_filename: str, optional
            Filename the trajectory is saved to. The filename-ending should be '.h5'. If no filename is provided
            a temporary filename is generated and used.
        :param pPsi: float
            Probability of moving the Psi-angle
        :param pPhi: float
            Probability of moving the Phi-angle
        :param pOmega: float
            Probability of moving the Omega-angle
        :param pChi: float
            Probability of moving the Chi-angle
        :param pdbOut: int
            Only every pdbOut frame is written to the tajectory
        :param av_number_protein_mc: int
            Number of MC-steps (total number of accepted number) performed before an AV-Monte-Carlo-step (AV-MC)
            is performed.
        :param ktAv: float
            Specifies the 'temperature' of the AV-MC. Here Chi2 is taken as the energy of the AV.
        :param kt: float
            Specifies the temperature of the MC-step
        :param maxTime: number, optional
            Maximum time (in real-time/laboratory-time) the simulation is performed in seconds. If no value
            is provided the simulations runs for one hour.
        :param scale: float
            Magnitude of change of the Monte-Carlo step. The actual change in each MC-step is determined
            by taking a random number out of a normal-distribution of a width of 'scale'.
        """
        Model.__init__(self, fit=fit)
        QtGui.QWidget.__init__(self)
        self.icon = QtGui.QIcon(":/icons/icons/Peptide_FRET.ico")

        uic.loadUi('./mfm/ui/proteinMC.ui', self)
        filename = tempfile.mktemp(".h5") if save_filename is None else save_filename
        ouputDir = os.path.dirname(self.filename)
        self.trajectory = None
        self.surface = self
        self.fit = fit
        TrajectoryFile.__init__(self, filename, fit.data, mode='w')
        self.structure = fit.data
        self.proteinWorker = ProteinMCWorker(parent=self)

        # initialize variables
        self.u1 = Universe()
        self.u2 = Universe()
        self.av = mfm.fluorescence.fps.AvWidget(self)
        self.u2.addPotential(self.av)

        self.number_of_moving_aa = number_of_moving_aa
        self._structure = fit.data
        self.ktAv = ktAv
        self.kt = kt
        self.scale = scale
        self.pPsi = pPsi
        self.pPhi = pPhi
        self.pChi = pChi
        self.pOmega = pOmega
        self.pdb_nOut = pdbOut
        self.fps_file = fps_file
        self.outputDir = ouputDir
        self.potential_weight = 1.0
        self.do_av_steepest_descent = True
        self.av_number_protein_mc = av_number_protein_mc  # number of protein mc-steps before av-mc step

        if config_file is not None:
            self.config_filename = config_file
        self.movemap = movemap

        p = potentials.potentialDict['H-Potential'](structure=self.structure, parent=self)
        p.hide()
        self.onAddPotential(p)

        p = potentials.potentialDict['Iso-UNRES'](structure=self.structure, parent=self)
        p.hide()
        self.onAddPotential(p)

        ## User-interface
        self.verticalLayout.addWidget(self.av)
        self.connect(self.actionAdd_potential, QtCore.SIGNAL('triggered()'), self.onAddPotential)
        self.connect(self.actionSet_output_filename, QtCore.SIGNAL('triggered()'), self.onSetOutputFilename)
        self.connect(self.actionRun_Simulation, QtCore.SIGNAL('triggered()'), self.onRunSimulation)
        self.connect(self.actionLoad_configuration_file, QtCore.SIGNAL('triggered()'), self.onLoadConfigurationFile)
        self.connect(self.actionSave_configuration_file, QtCore.SIGNAL('triggered()'), self.onSave_configuration_file)
        self.connect(self.actionStop_simulation, QtCore.SIGNAL('triggered()'), self.onMTStop)
        self.connect(self.actionSave_RMSD_table, QtCore.SIGNAL('triggered()'), self.onSaveSimulationTable)
        self.connect(self.actionCurrent_potential_changed, QtCore.SIGNAL('triggered()'), self.onSelectedPotentialChanged)
        self.connect(self.tableWidget, QtCore.SIGNAL("cellDoubleClicked (int, int)"), self.onRemovePotential)
        self.comboBox_2.addItems(list(potentials.potentialDict.keys()))
        self.comboBox_2.setCurrentIndex(1)
        self.comboBox_2.setCurrentIndex(0)
        ## QtThread
        self.connect(self.proteinWorker, QtCore.SIGNAL("finished()"), self.onSimulationStop)
        self.connect(self.proteinWorker, QtCore.SIGNAL("stopped()"), self.onSimulationStop)
        self.connect(self.proteinWorker.x, QtCore.SIGNAL("newStructure"), self.append)

    def onSetOutputFilename(self):
        filename = str(QtGui.QFileDialog.getSaveFileName(None, 'Save File', '', 'MC-trajectory output (*.h5)'))
        self.filename = filename

    def onSave_configuration_file(self):
        filename = str(QtGui.QFileDialog.getSaveFileName(None, 'Save File', '', 'MC-config files (*.mc.json)'))
        p = self.get_config()
        json.dump(p, open(filename, 'w'), sort_keys=True, indent=4, separators=(',', ': '))
        self.config_filename = filename

    def onLoadConfigurationFile(self):
        filename = str(QtGui.QFileDialog.getOpenFileName(None, 'Open File', '', 'MC-config files (*.mc.json)'))
        self.config_filename = filename

    @property
    def update_rmsd(self):
        return bool(self.checkBox.isChecked())

    @property
    def config_filename(self):
        return str(self.lineEdit_2.text())

    @config_filename.setter
    def config_filename(self, v, verbose=False):
        verbose = self.verbose or verbose
        if verbose:
            print("Opening config-file: %s" % v)
        if os.path.isfile(v):
            self.lineEdit_2.setText(v)
            p = json.load(open(v))
            self.set_config(**p)
        else:
            self.lineEdit_2.setText('default')

    @property
    def do_av_steepest_descent(self):
        return bool(self.radioButton_2.isChecked())

    @do_av_steepest_descent.setter
    def do_av_steepest_descent(self, v):
        if v is True:
            self.radioButton_2.setChecked(True)
        else:
            self.radioButton.setChecked(True)

    @property
    def av_number_protein_mc(self):
        return int(self.spinBox_3.value())

    @av_number_protein_mc.setter
    def av_number_protein_mc(self, v):
        return self.spinBox_3.setValue(v)

    @property
    def fps_file(self):
        return self.av.filename

    @fps_file.setter
    def fps_file(self, v):
        if v is None:
            self.mc_mode = 'simple'
        elif os.path.isfile(v):
            self.av.filename = v

    @property
    def mc_mode(self):
        if self.groupBox_7.isChecked():
            return 'av_mc'
        else:
            return 'simple'

    @mc_mode.setter
    def mc_mode(self, v):
        if v == 'av_mc':
            self.groupBox_7.setChecked(True)

    @property
    def movemap(self):
        return self._movemap

    @movemap.setter
    def movemap(self, v):
        if v is None:
            mm = copy.copy(self.structure.b_factors)
            mm += 1.0 if max(mm) == 0.0 else 0.0
            self._movemap = mm / max(mm)
        else:
            self._movemap = v
        # onUpdateMoveMapTable"
        table = self.tableWidget_2
        table.setRowCount(0)
        for i, p in enumerate(self.movemap):
            table.setRowCount(i + 1)
            tmp = QtGui.QTableWidgetItem("%.3f" % p)
            tmp.setFlags(QtCore.Qt.ItemIsEnabled)
            table.setItem(i, 0, tmp)
        table.resizeRowsToContents()

    def append(self, xyz, energy, fret_energy, elapsed_time):
        TrajectoryFile.append(self, xyz, energy=energy, energy_fret=fret_energy, verbose=self.verbose,
                              update_rmsd=self.update_rmsd)

        table = self.tableWidget_3
        rows = table.rowCount()
        table.insertRow(rows)
        tmp = QtGui.QTableWidgetItem()
        tmp.setData(0, float(self.drmsd[-1]))
        table.setItem(rows, 0, tmp)
        tmp = QtGui.QTableWidgetItem()
        tmp.setData(0, float(energy))
        table.setItem(rows, 1, tmp)
        tmp = QtGui.QTableWidgetItem()
        tmp.setData(0, float(fret_energy))
        table.setItem(rows, 2, tmp)
        tmp = QtGui.QTableWidgetItem()
        tmp.setData(0, float(self.rmsd[-1]))
        table.setItem(rows, 3, tmp)

        # update plots
        if self.append_new_structures:
            self.pymolPlot.append_structure(self.structure)
        self.plot.update()

    @property
    def append_new_structures(self):
        return self.checkBox_4.isChecked()

    def onSelectedPotentialChanged(self):
        layout = self.verticalLayout_2
        for i in range(layout.count()):
            layout.itemAt(i).widget().close()
        layout.addWidget(self.potential)

    @property
    def potential(self):
        return potentials.potentialDict[self.potential_name](structure=self.structure, parent=self)

    def onAddPotential(self, potential=None, potential_weight=None):
        potential_weight = self.potential_weight if potential_weight is None else potential_weight
        potential = self.potential if potential is None else potential
        self.u1.addPotential(potential, potential_weight)

        table = self.tableWidget
        rc = table.rowCount()
        table.insertRow(rc)
        tmp = QtGui.QTableWidgetItem(str(potential.name))
        tmp.setFlags(QtCore.Qt.ItemIsEnabled)
        table.setItem(rc, 0, tmp)
        tmp = QtGui.QTableWidgetItem(str(potential_weight))
        tmp.setFlags(QtCore.Qt.ItemIsEnabled)
        table.setItem(rc, 1, tmp)
        table.resizeRowsToContents()

    def onRemovePotential(self):
        print("onRemovePotential")
        table = self.tableWidget
        rc = table.rowCount()
        idx = int(table.currentIndex().row())
        if rc >= 0:
            if idx < 0:
                idx = 0
            table.removeRow(idx)
            self.u1.removePotential(idx)

    def onSaveSimulationTable(self):
        print("onSaveSimulationTable")
        filename = str(QtGui.QFileDialog.getSaveFileName(self, 'Save structure table', '.txt'))
        fp = open(filename, 'w')
        s = "dRMSD\tEnergy\tFRET(chi2)\tRMSD(vsFirst)\n"
        fp.write(s)
        table = self.tableWidget_3
        for r in range(table.rowCount()):
            drmsd = table.item(r, 0).text()
            e = table.item(r, 1).text()
            chi2 = table.item(r, 2).text()
            rmsd = table.item(r, 3).text()
            s = "%s\t%s\t%s\t%s\n" % (drmsd, e, chi2, rmsd)
            fp.write(s)
        fp.close()

    def onSimulationStop(self):
        mfm.widgets.MyMessageBox('Simulation finished')

    def onRunSimulation(self):
        print("ProteinMC:onRunSimulation")
        self.proteinWorker.setDaemonStopSignal(False)
        if self.movemap is None:
           raise ValueError("Movemap not set not possible to perform MC-integration")
        self.proteinWorker.p = self
        self.proteinWorker.start()

    def onMTStop(self):
        print("onMTStop")
        self.proteinWorker.setDaemonStopSignal(True)

    @property
    def structure(self):
        return self._structure

    @structure.setter
    def structure(self, v):
        if isinstance(v, Structure):
            self._structure = v
        else:
            raise ValueError("Only Structure objects!")

    @property
    def update_rmsd(self):
        return bool(self.checkBox.isChecked())

    @property
    def plot(self):
        return [p for p in self.plots if isinstance(p, mfm.plots.ProteinMCPlot)][0]

    @property
    def pymolPlot(self):
        return [p for p in self.plots if isinstance(p, mfm.plots.MolView)][0]

    @property
    def cluster_structures(self):
        return self.groupBox_6.isChecked()

    @property
    def calculate_av(self):
        return self.groupBox_7.isChecked()

    @property
    def potential_number(self):
        return int(self.comboBox_2.currentIndex())

    @property
    def potential_name(self):
        return list(potentials.potentialDict.keys())[self.potential_number]

    @property
    def filename(self):
        return str(self.lineEdit_13.text())

    @filename.setter
    def filename(self, v):
        v = str(v)
        TrajectoryFile.filename.fset(self, v)
        self.lineEdit_13.setText(v)

    @property
    def number_of_moving_aa(self):
        return int(self.spinBox.value())

    @number_of_moving_aa.setter
    def number_of_moving_aa(self, v):
        self.spinBox.setValue(int(v))

    @property
    def av_filename(self):
        return str(self.lineEdit.text())

    @av_filename.setter
    def av_filename(self, v):
        self.lineEdit.setText(str(v))

    @property
    def ktAv(self):
        return float(self.doubleSpinBox_8.value())

    @ktAv.setter
    def ktAv(self, v):
        return self.doubleSpinBox_8.setValue(float(v))

    @property
    def scale(self):
        return float(self.doubleSpinBox.value()) / 1000.0

    @scale.setter
    def scale(self, v):
        return self.doubleSpinBox.setValue(float(v) * 1000.0)

    @property
    def pdb_nOut(self):
        return int(self.spinBox_2.value())

    @pdb_nOut.setter
    def pdb_nOut(self, v):
        return self.spinBox_2.setValue(int(v))

    @property
    def kt(self):
        return float(self.doubleSpinBox_2.value())

    @kt.setter
    def kt(self, v):
        self.doubleSpinBox_2.setValue(float(v))

    @property
    def pPsi(self):
        return float(self.doubleSpinBox_3.value())

    @pPsi.setter
    def pPsi(self, v):
        self.doubleSpinBox_3.setValue(float(v))

    @property
    def pPhi(self):
        return float(self.doubleSpinBox_4.value())

    @pPhi.setter
    def pPhi(self, v):
        self.doubleSpinBox_4.setValue(float(v))

    @property
    def pOmega(self):
        return float(self.doubleSpinBox_5.value())

    @pOmega.setter
    def pOmega(self, v):
        self.doubleSpinBox_5.setValue(float(v))

    @property
    def pChi(self):
        return float(self.doubleSpinBox_6.value())

    @pChi.setter
    def pChi(self, v):
        self.doubleSpinBox_6.setValue(float(v))

