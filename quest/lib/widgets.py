import os
import pickle
import random
from PyQt5 import QtGui, QtCore, uic, QtWidgets
from lib.structure import Structure
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


class PDBSelector(QtWidgets.QWidget):

    def __init__(self, show_labels=True, update=None):
        QtWidgets.QWidget.__init__(self)
        uic.loadUi(
            os.path.join(
                os.path.dirname(
                    __file__
                ),
                'pdb_widget.ui'
            ),
            self
        )
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

