from __future__ import annotations

import os
import pickle
import random
from qtpy import QtGui, QtCore, uic, QtWidgets
from quest.lib.structure import Structure
import numpy as np
import quest.lib


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


class MyMessageBox(
    QtWidgets.QMessageBox
):

    def __init__(
            self,
            label: str = None,
            info: str = None,
            details: str = None
    ):
        """This Widget can be used to provide an output for warnings
        and exceptions. It can also display fortune cookies.

        :param label:
        :param info:
        :param show_fortune: if True than a fortune cookie is displayed.
        """
        super().__init__()
        self.Icon = 1
        self.setSizeGripEnabled(True)
        self.setIcon(
            QtWidgets.QMessageBox.Information
        )
        if label is not None:
            self.setWindowTitle(label)
        if details is not None:
            self.setDetailedText(details)
        else:
            self.close()

    def event(self, e):
        result = QtWidgets.QMessageBox.event(self, e)

        self.setMinimumHeight(0)
        self.setMaximumHeight(16777215)
        self.setMaximumWidth(16777215)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )
        text_edit = self.findChild(QtWidgets.QTextEdit)
        if text_edit is not None:
            text_edit.setMinimumHeight(0)
            text_edit.setMaximumHeight(16777215)
            text_edit.setMinimumWidth(0)
            text_edit.setMaximumWidth(16777215)
            text_edit.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Expanding
            )

        return result
