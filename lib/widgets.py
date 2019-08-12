import os
import pickle
import random
from PyQt5 import QtGui, QtCore, uic, QtWidgets
from lib.structure import Structure, TrajectoryFile
import numpy as np
import lib


def clearLayout(layout):
    if layout != None:
        while layout.count():
            child = layout.takeAt(0)
            if child.widget() is not None:
                child.widget().deleteLater()
            elif child.layout() is not None:
                clearLayout(child.layout())


class AVProperties(QtWidgets.QWidget):

    def __init__(self, av_type="AV1"):
        QtWidgets.QWidget.__init__(self)
        uic.loadUi('./lib/ui/av_property.ui', self)
        self._av_type = av_type
        self.av_type = av_type
        self.groupBox.hide()

    @property
    def av_type(self):
        return self._av_type

    @av_type.setter
    def av_type(self, v):
        self._av_type = v
        if v == 'AV1':
            self.label_4.setEnabled(False)
            self.label_5.setEnabled(False)
            self.doubleSpinBox_4.setEnabled(False)
            self.doubleSpinBox_5.setEnabled(False)
        if v == 'AV0':
            self.doubleSpinBox_4.setEnabled(False)
            self.doubleSpinBox_5.setEnabled(False)
        elif v == 'AV3':
            self.label_4.setEnabled(True)
            self.label_5.setEnabled(True)
            self.doubleSpinBox_4.setEnabled(True)
            self.doubleSpinBox_5.setEnabled(True)

    @property
    def linker_length(self):
        return float(self.doubleSpinBox.value())

    @linker_length.setter
    def linker_length(self, v):
        self.doubleSpinBox.setValue(v)

    @property
    def linker_width(self):
        return float(self.doubleSpinBox_2.value())

    @linker_width.setter
    def linker_width(self, v):
        self.doubleSpinBox_2.setValue(v)

    @property
    def radius_1(self):
        return float(self.doubleSpinBox_3.value())

    @radius_1.setter
    def radius_1(self, v):
        self.doubleSpinBox_3.setValue(v)

    @property
    def radius_2(self):
        return float(self.doubleSpinBox_4.value())

    @radius_2.setter
    def radius_2(self, v):
        self.doubleSpinBox_4.setValue(v)

    @property
    def radius_3(self):
        return float(self.doubleSpinBox_5.value())

    @radius_3.setter
    def radius_3(self, v):
        self.doubleSpinBox_5.setValue(v)

    @property
    def resolution(self):
        return float(self.doubleSpinBox_6.value())

    @resolution.setter
    def resolution(self, v):
        self.doubleSpinBox_6.setValue(v)

    @property
    def initial_linker_sphere(self):
        return float(self.doubleSpinBox_7.value())

    @initial_linker_sphere.setter
    def initial_linker_sphere(self, v):
        self.doubleSpinBox_7.setValue(v)

    @property
    def initial_linker_sphere_min(self):
        return float(self.doubleSpinBox_8.value())

    @initial_linker_sphere_min.setter
    def initial_linker_sphere_min(self, v):
        self.doubleSpinBox_8.setValue(v)

    @property
    def initial_linker_sphere_max(self):
        return float(self.doubleSpinBox_9.value())

    @initial_linker_sphere_max.setter
    def initial_linker_sphere_max(self, v):
        self.doubleSpinBox_9.setValue(v)


class PDBSelector(QtWidgets.QWidget):

    def __init__(self, show_labels=True, update=None):
        QtWidgets.QWidget.__init__(self)
        uic.loadUi('./lib/tools/dye_diffusion/ui/pdb_widget.ui', self)
        self._pdb = None
        self.comboBox.currentIndexChanged.connect(self.onChainChanged)
        self.comboBox_2.currentIndexChanged.connect(self.onResidueChanged)
        if not show_labels:
            self.label.hide()
            self.label_2.hide()
            self.label_3.hide()

    @property
    def atoms(self):
        return self._pdb

    @atoms.setter
    def atoms(self, v):
        self._pdb = v
        self.update_chain()

    @property
    def chain_id(self):
        return str(self.comboBox.currentText())

    @chain_id.setter
    def chain_id(self, v):
        pass

    @property
    def residue_name(self):
        try:
            return str(self.atoms[self.atom_number]['res_name'])
        except ValueError:
            return 0

    @property
    def residue_id(self):
        try:
            return int(self.comboBox_2.currentText())
        except ValueError:
            return 0

    @residue_id.setter
    def residue_id(self, v):
        pass

    @property
    def atom_name(self):
        return str(self.comboBox_3.currentText())

    @atom_name.setter
    def atom_name(self, v):
        pass

    @property
    def atom_number(self):
        pdb = self.atoms
        residue_key = self.residue_id
        atom_name = self.atom_name
        chain = self.chain_id
        w = np.where((pdb['res_id'] == residue_key) & (pdb['atom_name'] == atom_name) &
                     (pdb['chain'] == chain))
        return w[0][0]

    def onChainChanged(self):
        print("PDBSelector:onChainChanged")
        self.comboBox_2.clear()
        pdb = self._pdb
        chain = str(self.comboBox.currentText())
        atom_ids = np.where(pdb['chain'] == chain)[0]
        residue_ids = list(set(self.atoms['res_id'][atom_ids]))
        residue_ids_str = [str(x) for x in residue_ids]
        self.comboBox_2.addItems(residue_ids_str)

    def onResidueChanged(self):
        self.comboBox_3.clear()
        pdb = self.atoms
        chain = self.chain_id
        residue = self.residue_id
        print("onResidueChanged: %s" % residue)
        atom_ids = np.where((pdb['res_id'] == residue) & (pdb['chain'] == chain))[0]
        atom_names = [atom['atom_name'] for atom in pdb[atom_ids]]
        self.comboBox_3.addItems(atom_names)

    def update_chain(self):
        self.comboBox.clear()
        chain_ids = list(set(self.atoms['chain'][:]))
        chain_ids = [str(c) for c in chain_ids]
        self.comboBox.addItems(chain_ids)

