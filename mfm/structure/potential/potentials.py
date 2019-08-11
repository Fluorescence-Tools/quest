__author__ = 'thomas'
from collections import OrderedDict
import copy

import numpy as np
from PyQt4 import QtCore, QtGui, uic

from mfm.structure.potential import cPotentials


try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s
try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)


class Ramachandran(object):

    def __init__(self, structure, filename='./mfm/structure/potential/database/rama_ala_pro_gly.npy'):
        """
        :param filename:
        :return:
        """
        self.structure = structure
        self.name = 'rama'
        self.filename = filename
        self.ramaPot = np.load(self.filename)

    def getEnergy(self):
        c = self.structure
        Erama = cPotentials.ramaEnergy(c.residue_lookup_i, c.iAtoms, self.ramaPot)
        self.E = Erama
        return Erama


class Electrostatics(object):

    def __init__(self, structure, type='gb'):
        """
        :param type:
        :return:
        """
        self.structure = structure
        self.name = 'ele'
        if type == 'gb':
            self.p = cPotentials.gb

    def getEnergy(self):
        structure = self.structure
        Eel = cPotentials.gb(structure.rAtoms)
        self.E = Eel
        return Eel


class HPotential(object):

    name = 'H-Bond'

    def __init__(self, structure, cutoff_ca=8.0, cutoff_hbond=3.0):
        self.structure = structure
        self.cutoffH = cutoff_hbond
        self.cutoffCA = cutoff_ca

    def getEnergy(self):
        s1 = self.structure
        cca2 = self.cutoffCA ** 2
        ch2 = self.cutoffH ** 2
        nHbond, Ehbond = cPotentials.hBondLookUpAll(s1.l_res, s1.dist_ca, s1.xyz, self._hPot, cca2, ch2)
        self.E = Ehbond
        self.nHbond = nHbond
        return self.E

    def getNbrBonds(self):
        """
        :return:
        """
        if self.nHbond is None:
            return 0
        return self.nHbond

    @property
    def potential(self):
        return self.hPot

    @potential.setter
    def potential(self, v):
        self._hPot = np.loadtxt(v, skiprows=1, dtype=np.float64).T[1:, :]
        self.hPot = self._hPot
        self.lineEdit_3.setText(str(v))


class HPotentialWidget(HPotential, QtGui.QWidget):

    def __init__(self, structure, parent, cutoff_ca=8.0, cutoff_hbond=3.0):
        QtGui.QWidget.__init__(self, parent=parent)
        uic.loadUi('mfm/ui/Potential_Hbond_2.ui', self)
        self.potential = './mfm/structure/potential/database/hb.csv'
        HPotential.__init__(self, structure)
        self.connect(self.checkBox, QtCore.SIGNAL("stateChanged (int)"), self.updateParameter)
        self.connect(self.checkBox_2, QtCore.SIGNAL("stateChanged (int)"), self.updateParameter)
        self.connect(self.checkBox_3, QtCore.SIGNAL("stateChanged (int)"), self.updateParameter)
        self.connect(self.checkBox_4, QtCore.SIGNAL("stateChanged (int)"), self.updateParameter)
        self.connect(self.actionLoad_potential, QtCore.SIGNAL('triggered()'), self.onOpenFile)
        self.cutoffCA = cutoff_ca
        self.cutoffH = cutoff_hbond

    def onOpenFile(self):
        filename = str(QtGui.QFileDialog.getOpenFileName(None, 'Open File', '', 'CSV data files (*.csv)'))
        self.potential = filename

    def updateParameter(self):
        print("updateParameter")
        hPot = copy.deepcopy(self._hPot)
        if not self.oh:
            hPot[2, :] *= 0.0
        if not self.on:
            hPot[1, :] *= 0.0
        if not self.cn:
            hPot[3, :] *= 0.0
        if not self.ch:
            hPot[0, :] *= 0.0
        self.hPot = hPot

    @property
    def oh(self):
        return int(self.checkBox.isChecked())

    @property
    def cn(self):
        return int(self.checkBox_2.isChecked())

    @property
    def ch(self):
        return int(self.checkBox_3.isChecked())

    @property
    def on(self):
        return int(self.checkBox_4.isChecked())

    @property
    def cutoffH(self):
        return float(self.doubleSpinBox.value())

    @cutoffH.setter
    def cutoffH(self, v):
        self.doubleSpinBox.setValue(float(v))

    @property
    def cutoffCA(self):
        return float(self.doubleSpinBox_2.value())

    @cutoffCA.setter
    def cutoffCA(self, v):
        self.doubleSpinBox_2.setValue(float(v))


class GoPotential(object):

    def __init__(self, structure):
        self.structure = structure
        self.name = 'go'

    def setGo(self):
        c = self.structure
        nnEFactor = self.nnEFactor if self.non_native_contact_on else 0.0
        cutoff = self.cutoff if self.native_cutoff_on else 1e6
        self.eMatrix, self.sMatrix = cPotentials.go_init(c.residue_lookup_r, c.dist_ca,
                                                         self.epsilon, nnEFactor, cutoff)

    def getEnergy(self):
        c = self.structure
        Etot, nNa, Ena, nNN, Enn = cPotentials.go(c.residue_lookup_r, c.dist_ca, self.eMatrix, self.sMatrix)
        self.E = Etot
        self.Ena = Ena
        self.Enn = Enn
        self.nNa = nNa
        self. nNN = nNN
        return Etot

    def getNbrNonNative(self):
        return self.nNN

    def getNbrNative(self):
        return self.nNa

    def set_sMatrix(self, sMatrix):
        self.sMatrix = sMatrix

    def set_eMatrix(self, eMatrix):
        self.eMatrix = eMatrix

    def set_nMatrix(self, nMatrix):
        self.nMatrix = nMatrix


class GoPotentialWidget(GoPotential, QtGui.QWidget):
    def __init__(self, structure, parent):
        GoPotential.__init__(self, structure)
        QtGui.QWidget.__init__(self, parent=None)
        uic.loadUi('mfm/ui/Potential-CaLJ.ui', self)
        self.connect(self.lineEdit, QtCore.SIGNAL("textChanged(QString)"), self.setGo)
        self.connect(self.lineEdit_2, QtCore.SIGNAL("textChanged(QString)"), self.setGo)
        self.connect(self.lineEdit_3, QtCore.SIGNAL("textChanged(QString)"), self.setGo)

    @property
    def native_cutoff_on(self):
        return bool(self.checkBox.isChecked())

    @property
    def non_native_contact_on(self):
        return bool(self.checkBox_2.isChecked())

    @property
    def epsilon(self):
        return float(self.lineEdit.text())

    @property
    def nnEFactor(self):
        return float(self.lineEdit_2.text())

    @property
    def cutoff(self):
        return float(self.lineEdit_3.text())


class MJPotential(object):

    name = 'Miyazawa-Jernigan'

    def __init__(self, structure, filename='./mfm/structure/potential/database/mj.csv', ca_cutcoff=6.5):
        self.filename = filename
        self.structure = structure
        self.potential = filename
        self.name = 'mj'
        self.ca_cutoff = ca_cutcoff

    @property
    def potential(self):
        return self.mjPot

    @potential.setter
    def potential(self, v):
        self.mjPot = np.loadtxt(v)

    def getEnergy(self):
        c = self.structure
        nCont, Emj = cPotentials.mj(c.l_res, c.residue_types, c.dist_ca, c.xyz, self.mjPot, cutoff=self.ca_cutoff)
        self.E = Emj
        self.nCont = nCont
        return Emj

    def getNbrContacts(self):
        return self.nCont


class MJPotentialWidget(MJPotential, QtGui.QWidget):

    def __init__(self, structure, parent, filename='./mfm/structure/potential/database/mj.csv', ca_cutoff=6.5):
        QtGui.QWidget.__init__(self, parent=None)
        uic.loadUi('mfm/ui/MJ-resource.ui', self)
        MJPotential.__init__(self, structure)
        self.connect(self.pushButton, QtCore.SIGNAL("clicked()"), self.onOpenFile)
        self.potential = filename
        self.ca_cutoff = ca_cutoff

    def onOpenFile(self):
        filename = str(QtGui.QFileDialog.getOpenFileName(None, 'Open MJ-Potential', '', 'CSV data files (*.csv)'))
        self.potential = filename

    @property
    def potential(self):
        return self.mjPot

    @potential.setter
    def potential(self, v):
        self.mjPot = np.loadtxt(v)
        self.lineEdit.setText(v)

    @property
    def ca_cutoff(self):
        return float(self.lineEdit_2.text())

    @ca_cutoff.setter
    def ca_cutoff(self, v):
        self.lineEdit_2.setText(str(v))


class ASA(object):
    def __init__(self, structure, probe=1.0, n_sphere_point=590, radius=2.5):
        self.structure = structure
        self.probe = probe
        self.n_sphere_point = n_sphere_point
        self.sphere_points = cPotentials.spherePoints(n_sphere_point)
        self.radius = radius

    def getEnergy(self):
        c = self.structure
        asa = cPotentials.asa(c.rAtoms['coord'], c.residue_lookup_r, c.dist_ca,
                              self.sphere_points, self.probe, self.radius)
        return asa


class CEPotential(object):
    """
    Examples
    --------

    >>> import mfm.structure
    >>> import mfm.structure.potential

    >>> s = mfm.Structure('./sample_data/model/hgbp1/hGBP1_closed.pdb', verbose=True, make_coarse=True)
    >>> pce = mfm.structure.potential.potentials.CEPotential(s, ca_cutoff=64.0)
    >>> pce.getEnergy()
    -0.15896629131635745
    """

    name = 'Iso-UNRES'

    def __init__(self, structure, potential='./mfm/structure/potential/database/unres.npy', ca_cutoff=15.0,
                 scaling_factor=0.593):
        """
        scaling_factor : factor to scale energies from kCal/mol to kT=1.0 at 298K
        """
        self.structure = structure
        self.ca_cutoff = ca_cutoff
        self._potential = None
        self.potential = potential
        self.scaling_factor = scaling_factor

    @property
    def potential(self):
        return self._potential

    @potential.setter
    def potential(self, v):
        self._potential = np.load(v)

    def getEnergy(self, cutoff=None):
        cutoff = cutoff if cutoff is not None else self.ca_cutoff
        c = self.structure
        coord = np.ascontiguousarray(c.xyz)
        dist_ca = np.ascontiguousarray(c.dist_ca)
        residue_types = np.ascontiguousarray(c.residue_types)
        l_res = np.ascontiguousarray(c.l_res)

        nCont, E = cPotentials.centroid2(l_res, residue_types, dist_ca, coord, self.potential,
                                         cutoff=cutoff)
        self.nCont = nCont
        return float(E * self.scaling_factor)

    def getNbrContacts(self):
        return self.nCont


class CEPotentialWidget(CEPotential, QtGui.QWidget):

    def __init__(self, structure, parent, potential='./mfm/structure/potential/database/unres.npy',
                 ca_cutoff=25.0):
        QtGui.QWidget.__init__(self, parent=None)
        uic.loadUi('mfm/ui/unres-cb-resource.ui', self)
        CEPotential.__init__(self, structure, potential, ca_cutoff=ca_cutoff)
        self.connect(self.actionOpen_potential_file, QtCore.SIGNAL('triggered()'), self.onOpenPotentialFile)
        self.ca_cutoff = ca_cutoff

    @property
    def potential(self):
        return self._potential

    @potential.setter
    def potential(self, v):
        self.lineEdit.setText(str(v))
        self._potential = np.load(v)

    @property
    def ca_cutoff(self):
        return float(self.doubleSpinBox.value())

    @ca_cutoff.setter
    def ca_cutoff(self, v):
        self.doubleSpinBox.setValue(float(v))

    def onOpenPotentialFile(self):
        filename = str(QtGui.QFileDialog.getOpenFileName(None, 'Open CE-Potential', '', 'Numpy file (*.npy)'))
        self.potential = filename


class ASA(object):

    name = 'Asa-Ca'

    def __init__(self, structure, probe=1.0, n_sphere_point=590, radius=2.5):
        self.structure = structure
        self.probe = probe
        self.n_sphere_point = n_sphere_point
        self.sphere_points = cPotentials.spherePoints(n_sphere_point)
        self.radius = radius

    def getEnergy(self):
        c = self.structure
        #def asa(double[:, :] xyz, int[:, :] resLookUp, double[:, :] caDist, double[:, :] sphere_points,
        #double probe=1.0, double radius = 2.5, char sum=1)
        asa = cPotentials.asa(c.xyz, c.l_res, c.dist_ca, self.sphere_points, self.probe, self.radius)
        return asa


class AsaWidget(ASA, QtGui.QWidget):

    def __init__(self, structure, parent):
        ASA.__init__(self, structure)
        QtGui.QWidget.__init__(self, parent=None)
        uic.loadUi('mfm/ui/Potential_Asa.ui', self)

        self.connect(self.lineEdit, QtCore.SIGNAL("textChanged(QString)"), self.setParameterSphere)
        self.connect(self.lineEdit_2, QtCore.SIGNAL("textChanged(QString)"), self.setParameterProbe)

        self.lineEdit.setText('590')
        self.lineEdit_2.setText('3.5')

    def setParameterSphere(self):
        self.n_sphere_point = int(self.lineEdit.text())

    def setParameterProbe(self):
        self.probe = float(self.lineEdit_2.text())


class RadiusGyration(QtGui.QWidget):

    name = 'Radius-Gyration'

    def __init__(self, structure, parent=None):
        QtGui.QWidget.__init__(self, parent=parent)
        self.structure = structure
        self.parent = parent

    def getEnergy(self, c=None):
        if c is None:
            c = self.structure
        return c.radius_gyration


class ClashPotential(object):

    name = 'Clash-Potential'

    def __init__(self, **kwargs):
        """
        :param kwargs:
        :return:

        Examples
        --------

        >>> import mfm.structure
        >>> import mfm.structure.potential

        >>> s = mfm.Structure('./sample_data/model/hgbp1/hGBP1_closed.pdb', verbose=True, make_coarse=True)
        >>> pce = mfm.structure.potential.potentials.ClashPotential(structure=s, clash_tolerance=6.0)
        >>> pce.getEnergy()

        """
        self.structure = kwargs.get('structure', None)
        self.clash_tolerance = kwargs.get('clash_tolerance', 2.0)
        self.covalent_radius = kwargs.get('covalent_radius', 1.5)

    def getEnergy(self):
        c = self.structure
        return cPotentials.clash_potential(c.xyz, c.vdw, self.clash_tolerance, self.covalent_radius)


class ClashPotentialWidget(ClashPotential, QtGui.QWidget):

    def __init__(self, **kwargs):
        QtGui.QWidget.__init__(self)
        uic.loadUi('mfm/ui/potential-clash.ui', self)
        ClashPotential.__init__(self, **kwargs)

    @property
    def clash_tolerance(self):
        return float(self.doubleSpinBox.value())

    @clash_tolerance.setter
    def clash_tolerance(self, v):
        self.doubleSpinBox.setValue(v)

    @property
    def covalent_radius(self):
        return float(self.doubleSpinBox_2.value())

    @covalent_radius.setter
    def covalent_radius(self, v):
        self.doubleSpinBox_2.setValue(v)



potentialDict = OrderedDict()
potentialDict['H-Potential'] = HPotentialWidget
potentialDict['Iso-UNRES'] = CEPotentialWidget
potentialDict['Miyazawa-Jernigan'] = MJPotentialWidget
#potentialDict['Go-Potential'] = GoPotentialWidget
potentialDict['ASA-Calpha'] = AsaWidget
potentialDict['Radius of Gyration'] = RadiusGyration
potentialDict['Clash potential'] = ClashPotentialWidget
