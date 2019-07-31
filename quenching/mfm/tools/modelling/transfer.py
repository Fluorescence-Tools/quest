import tempfile

import numpy as np
from PyQt4 import QtCore, QtGui, uic
import mdtraj as md
import mfm
from mfm.fluorescence import distance2fretrate
from mfm.structure.cStructure import calculate_kappa2_distance
from mfm.io import pdb
from mfm.widgets import PDBSelector


class CalculateTransfer(object):

    def __init__(self, trajectory_file=None, dipoles=True, **kwargs):
        """

        :param trajectory: TrajectoryFile
            A list of PDB-filenames
        :param dipoles: bool
            If dipoles is True the transfer-efficiency is calculated using the distance and
            the orientation factor kappa2. If dipoles is False only the first atoms defining the
            Donor and the Acceptor are used to calculate the transfer-efficiency. If dipoles is True
            the first and the second atom of donor and acceptor are used.
        :param kappa2: float
            This parameter defines kappa2 if dipoles is False.
        :param verbose: bool
            If verbose is True -> output to std-out.
        """
        self.trajectory_file = trajectory_file
        self.__donorAtomID = None
        self.__acceptorAtomID = None
        self.dipoles = dipoles
        self.__kappa2s = None
        self.__distances = None

        self.stride = kwargs.get('stride', 1)
        self.verbose = kwargs.get('verbose', mfm.verbose)
        self.kappa2 = kwargs.get('kappa2', 0.66666666)
        self.tau0 = kwargs.get('tau0', 2.6)
        self.R0 = kwargs.get('forster_radius', 52.0)

    @property
    def donor(self):
        return self.__donorAtomID

    @donor.setter
    def donor(self, v):
        self.__donorAtomID = v

    @property
    def acceptor(self):
        return self.__acceptorAtomID

    @acceptor.setter
    def acceptor(self, v):
        self.__acceptorAtomID = v

    @property
    def kappa2(self):
        if self.dipoles:
            return self.__kappa2s
        else:
            return self.__kappa2

    @kappa2.setter
    def kappa2(self, v):
        if isinstance(v, np.ndarray):
            self.__kappa2s = v
        else:
            self.__kappa2 = float(v)

    @property
    def distances(self):
        return self.__distances

    @property
    def tau0(self):
        return self.__tau0

    @tau0.setter
    def tau0(self, v):
        self._tau0 = float(v)

    @property
    def R0(self):
        return self.__R0

    @R0.setter
    def R0(self, v):
        self.__R0 = float(v)

    def calc(self, filename, **kwargs):
        verbose = kwargs.get('verbose', self.verbose)
        trajectory_file = self.trajectory_file
        donor = self.donor
        acceptor = self.acceptor

        if verbose:
            print("Calculating distances and kappa2")
            print("Donor-Dipole atoms: %s" % donor)
            print("Acceptor-Dipole atoms: %s" % acceptor)

        # Write header
        open(filename, 'w').write(b'RDA[Ang]\tkappa2\tFRETrate[1/ns]\n')
        for chunk in md.iterload(trajectory_file, stride=self.stride):

            if self.dipoles:
                d, k2 = calculate_kappa2_distance(chunk.xyz, self.donor[0], self.donor[1],
                                                  self.acceptor[0], self.acceptor[1])
            else:
                k2 = np.zeros(chunk.n_frames, dtype=np.float32)

                d1 = chunk.xyz[:, self.donor[0], :]
                a1 = chunk.xyz[:, self.acceptor[0], :]
                d = np.sqrt(np.sum((a1 - d1)**2, axis=2))

            with open(filename, 'a') as f_handle:
                r = np.array([d * 10.0, k2, distance2fretrate(d * 10.0, self.R0, self.tau0, k2)]).T
                np.savetxt(f_handle, r,
                           delimiter='\t')


class Structure2Transfer(QtGui.QWidget, CalculateTransfer):

    name = "Kappa2Dist"

    def __init__(self, verbose=True):
        QtGui.QWidget.__init__(self)
        uic.loadUi('./mfm/ui/structure2transfer.ui', self)
        CalculateTransfer.__init__(self)
        self._trajectory_file = ''

        self.verbose = verbose
        self.d1 = PDBSelector()
        self.d2 = PDBSelector(show_labels=False)

        self.a1 = PDBSelector()
        self.a2 = PDBSelector(show_labels=False)

        self.horizontalLayout_2.addWidget(self.d1)
        self.horizontalLayout_3.addWidget(self.a1)
        self.horizontalLayout_2.addWidget(self.d2)
        self.horizontalLayout_3.addWidget(self.a2)

        self.connect(self.actionOpen_trajectory, QtCore.SIGNAL('triggered()'), self.onLoadTrajectory)
        self.connect(self.actionProcess_trajectory, QtCore.SIGNAL('triggered()'), self.calc)
        self.hide()

    def calc(self):
        filename = str(QtGui.QFileDialog.getSaveFileName(self, 'Select filename', '.'))
        CalculateTransfer.calc(self, verbose=True, filename=filename)

    @property
    def stride(self):
        return int(self.spinBox.value())

    @stride.setter
    def stride(self, v):
        self.spinBox.setValue(v)

    @property
    def donor(self):
        d = [self.d1.atom_number, self.d2.atom_number]
        return d

    @property
    def acceptor(self):
        d = [self.a1.atom_number, self.a2.atom_number]
        return d

    @property
    def R0(self):
        return float(self.doubleSpinBox.value())

    @R0.setter
    def R0(self, v):
        self.doubleSpinBox.setValue(float(v))

    @property
    def tau0(self):
        return self.doubleSpinBox_2.value()

    @tau0.setter
    def tau0(self, v):
        self.doubleSpinBox_2.setValue(float(v))

    @property
    def dipoles(self):
        return self.checkBox.isChecked()

    @dipoles.setter
    def dipoles(self, v):
        self.checkBox.setChecked(bool(v))

    @property
    def pdb(self):
        if self._pdb is None:
            raise ValueError("No pdb file set yet.")
        return self._pdb

    @pdb.setter
    def pdb(self, v):
        if isinstance(v, str):
            v = pdb.read(v, verbose=self.verbose)
        self._pdb = v

    @property
    def trajectory_file(self):
        return str(self.lineEdit_3.text())

    @trajectory_file.setter
    def trajectory_file(self, v):
        self.lineEdit_3.setText(str(v))

    @property
    def topology_file(self):
        return str(self.lineEdit.text())

    @topology_file.setter
    def topology_file(self, value):
        self.pdb = str(value)

    def onLoadTrajectory(self):
        self.trajectory_file = str(QtGui.QFileDialog.getOpenFileName(self, 'Open Trajectory-File', '.h5', 'H5-Trajectory-Files (*.h5)'))

        frame0 = md.load_frame(self.trajectory_file, 0)

        tmp = tempfile.mktemp(".pdb")
        frame0.save(tmp)

        self.topology_file = tmp

        self.d1.atoms = self.pdb
        self.d2.atoms = self.pdb

        self.a1.atoms = self.pdb
        self.a2.atoms = self.pdb
