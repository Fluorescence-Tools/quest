import os
import pickle
import random
from PyQt4 import QtGui, QtCore, uic
#from PyQt4.QtCore import pyqtSignal
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


def get_fortune(fortunepath='./lib/ui/fortune/', min_length=0, max_length=100, attempts=1000, **kwargs):
    fortune_files = [os.path.splitext(pdat)[0] for pdat in os.listdir(fortunepath) if pdat.endswith(".pdat")]
    attempt = 0
    while True:
        fortune_file = os.path.join(fortunepath, random.choice(fortune_files))
        data = pickle.load(open(fortune_file+".pdat", "rb"))
        (start, length) = random.choice(data)
        print(random.choice(data))
        if length < min_length or (max_length is not None and length > max_length):
            attempt += 1
            if attempt > attempts:
                return ""
            continue
        ffh = open(fortune_file, 'rU')
        ffh.seek(start)
        fortunecookie = ffh.read(length)
        ffh.close()
        return fortunecookie


class AVProperties(QtGui.QWidget):

    def __init__(self, av_type="AV1"):
        QtGui.QWidget.__init__(self)
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


class PDBSelector(QtGui.QWidget):

    def __init__(self, show_labels=True, update=None):
        QtGui.QWidget.__init__(self)
        uic.loadUi('./lib/tools/dye_diffusion/ui/pdb_widget.ui', self)
        self._pdb = None
        self.connect(self.comboBox, QtCore.SIGNAL("currentIndexChanged(int)"), self.onChainChanged)
        self.connect(self.comboBox_2, QtCore.SIGNAL("currentIndexChanged(int)"), self.onResidueChanged)
        if not show_labels:
            self.label.hide()
            self.label_2.hide()
            self.label_3.hide()
        if update is not None:
            self.connect(self.comboBox_2, QtCore.SIGNAL("currentIndexChanged(int)"), update)
            self.connect(self.comboBox_2, QtCore.SIGNAL("currentIndexChanged(int)"), update)

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
        self.comboBox.addItems(chain_ids)


class MyMessageBox(QtGui.QMessageBox):

    def __init__(self, label=None, info=None):
        QtGui.QMessageBox.__init__(self)
        self.Icon = 1
        self.setSizeGripEnabled(True)
        self.setIcon(QtGui.QMessageBox.Information)
        if label is not None:
            self.setWindowTitle(label)
        if info is not None:
            self.setDetailedText(info)
        if lib.fortune_properties['enabled']:
            fortune = get_fortune(**lib.fortune_properties)
            self.setInformativeText(fortune)
            self.exec_()
            self.setMinimumWidth(450)
            self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        else:
            self.close()

    def event(self, e):
        result = QtGui.QMessageBox.event(self, e)

        self.setMinimumHeight(0)
        self.setMaximumHeight(16777215)
        self.setMaximumWidth(16777215)
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

        textEdit = self.findChild(QtGui.QTextEdit)
        if textEdit != None :
            textEdit.setMinimumHeight(0)
            textEdit.setMaximumHeight(16777215)
            textEdit.setMinimumWidth(0)
            textEdit.setMaximumWidth(16777215)
            textEdit.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

        return result


class CurveSelector(QtGui.QListWidget):

    @property
    def curve_name(self):
        try:
            return self.selected_curve.name
        except ValueError:
            return '-'

    @property
    def datasets(self):
        return lib.getDataCurves()

    @property
    def selected_index(self):
        return int(self.currentRow())

    @property
    def selected_curve(self):
        datasets = self.datasets
        if self.selected_index < len(datasets) and len(datasets) > 0:
            return datasets[self.selected_index]
        else:
            return None

    def show(self):
        QtGui.QListWidget.show(self)
        if len(self.datasets) == 0:
            self.hide()
        self.clear()
        self.addItems([d.name for d in self.datasets])

    def __init__(self, **kwargs):
        QtGui.QListWidget.__init__(self, **kwargs)
        self.resize(300, 150)
        self.doubleClicked.connect(self.hide)
        self.clicked.connect(self.hide)
        self.hide()

class PDBLoad(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self)
        uic.loadUi("lib/ui/proteinMCLoad.ui", self)
        self._data = None
        self._filename = ''

        #self.connect(self.pushButton_12, QtCore.SIGNAL("clicked()"), self.onLoadStructure)

    def onLoadStructure(self):
        self.filename = str(QtGui.QFileDialog.getOpenFileName(None, 'Open Structure', '', 'PDB-file (*.pdb)'))
        self.structure = self.filename
        self.lineEdit.setText(str(self.structure.n_atoms))
        self.lineEdit_2.setText(str(self.structure.n_residues))

    @property
    def filename(self):
        return str(self.lineEdit_7.text())

    @filename.setter
    def filename(self, v):
        self.lineEdit_7.setText(v)

    @property
    def calcLookUp(self):
        return self.checkBox.isChecked()

    @property
    def structure(self):
        return self._data

    @structure.setter
    def structure(self, v):
        self._data = Structure(v, make_coarse=self.calcLookUp)


class LoadThread(QtCore.QThread):

    #procDone = pyqtSignal(bool)
    #partDone = pyqtSignal(int)

    def run(self):
        nFiles = len(self.filenames)
        print('File loading started')
        print('#Files: %s' % nFiles)
        for i, fn in enumerate(self.filenames):
            f = self.read(fn, *self.read_parameter)
            self.target.append(f, *self.append_parameter)
            self.partDone.emit(float(i + 1) / nFiles * 100)
        #self.procDone.emit(True)
        print('reading finished')


class PDBFolderLoad(QtGui.QWidget):

    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self)
        uic.loadUi("lib/ui/proteinFolderLoad.ui", self)
        self.connect(self.pushButton_12, QtCore.SIGNAL("clicked()"), self.onLoadStructure)
        self.updatePBar(0)
        self.load_thread = LoadThread()
        self.load_thread.partDone.connect(self.updatePBar)
        self.load_thread.procDone.connect(self.fin)
        self.trajectory = TrajectoryFile()

    def fin(self):
        print("Loading of structures finished")
        self.lineEdit.setText(str(self.nAtoms))
        self.lineEdit_2.setText(str(self.nResidues))

    def updatePBar(self, val):
        self.progressBar.setValue(val)

    def onLoadStructure(self):
        directory = str(QtGui.QFileDialog.getExistingDirectory(self, "Select Directory"))

        self.folder = directory
        filenames = [os.path.join(directory, f) for f in os.listdir(directory)
                     if os.path.isfile(os.path.join(directory, f))]
        filenames.sort()

        pdb_filenames = []
        for i, filename in enumerate(filenames):
            extension = os.path.splitext(filename)[1][1:]
            if filename.lower().endswith('.pdb') or extension.isdigit():
                pdb_filenames.append(filename)

        self.n_files = len(pdb_filenames)

        self.load_thread.read = Structure
        self.load_thread.read_parameter = [self.calc_internal, self.verbose]
        self.load_thread.append_parameter = [self.calc_rmsd]
        self.trajectory = TrajectoryFile(use_objects=self.use_objects, calc_internal=self.calc_internal,
                                     verbose=self.verbose)
        self.load_thread.filenames = pdb_filenames
        self.load_thread.target = self.trajectory
        self.load_thread.start()

    @property
    def calc_rmsd(self):
        return self.checkBox_4.isChecked()

    @property
    def n_files(self):
        return int(self.lineEdit_3.text())

    @n_files.setter
    def n_files(self, v):
        self.lineEdit_3.setText(str(v))

    @property
    def verbose(self):
        return bool(self.checkBox_3.isChecked())

    @property
    def nAtoms(self):
        return self.trajectory[0].n_atoms

    @property
    def nResidues(self):
        return self.trajectory[0].n_residues

    @property
    def folder(self):
        return str(self.lineEdit_7.text())

    @folder.setter
    def folder(self, v):
        self.lineEdit_7.setText(v)

    @property
    def use_objects(self):
        return bool(self.checkBox_2.isChecked())

    @property
    def calc_internal(self):
        return self.checkBox.isChecked()


