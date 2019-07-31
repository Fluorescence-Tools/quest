import os

from PyQt4 import QtGui, uic, QtCore

from mfm.structure import Structure
from mfm.io.photons import filetypes, Photons


class SpcFileWidget(QtGui.QWidget):

    def __init__(self, parent):
        QtGui.QWidget.__init__(self)
        uic.loadUi('mfm/ui/io/spcSampleSelectWidget.ui', self)
        self.parent = parent
        self.filetypes = filetypes

        self.connect(self.actionSample_changed, QtCore.SIGNAL('triggered()'), self.onSampleChanged)
        self.connect(self.actionLoad_sample, QtCore.SIGNAL('triggered()'), self.onLoadSample)
        self.connect(self.comboBox_2, QtCore.SIGNAL("currentIndexChanged(int)"), self.onFileTypeChanged)

    @property
    def sampleName(self):
        try:
            return self.filenames[0] + "_" + self.comboBox.currentText()
        except AttributeError:
            return "--"

    @property
    def dt(self):
        return float(self.doubleSpinBox.value())

    @dt.setter
    def dt(self, v):
        self.doubleSpinBox.setValue(v)

    def onSampleChanged(self):
        index = self.comboBox.currentIndex()
        self._photons.sample = self.samples[index]
        self.dt = float(self._photons.MTCLK / self._photons.nTAC) * 1e6
        self.nTAC = self._photons.nTAC
        self.nROUT = self._photons.nROUT
        self.number_of_photons = self._photons.nPh
        self.measurement_time = self._photons.measTime
        self.lineEdit_7.setText("%.2f" % self.count_rate)

    @property
    def measurement_time(self):
        return float(self._photons.measTime)

    @measurement_time.setter
    def measurement_time(self, v):
        self.lineEdit_6.setText("%.1f" % v)

    @property
    def number_of_photons(self):
        return int(self.lineEdit_5.value())

    @number_of_photons.setter
    def number_of_photons(self, v):
        self.lineEdit_5.setText(str(v))

    @property
    def rep_rate(self):
        return float(self.doubleSpinBox_2.value())

    @rep_rate.setter
    def rep_rate(self, v):
        self.doubleSpinBox_2.setValue(v)

    @property
    def nROUT(self):
        return int(self.lineEdit_3.text())

    @nROUT.setter
    def nROUT(self, v):
        self.lineEdit_3.setText(str(v))

    @property
    def nTAC(self):
        return int(self.lineEdit.text())

    @nTAC.setter
    def nTAC(self, v):
        self.lineEdit.setText(str(v))

    @property
    def filetypes(self):
        return self._file_types

    @filetypes.setter
    def filetypes(self, v):
        self._file_types = v
        self.comboBox_2.addItems(list(v.keys()))

    @property
    def count_rate(self):
        return self._photons.nPh / float(self._photons.measTime) / 1000.0

    def onFileTypeChanged(self):
        self._photons = None
        self.comboBox.clear()
        if self.fileType == "hdf":
            self.comboBox.setDisabled(False)
        else:
            self.comboBox.setDisabled(True)

    @property
    def fileType(self):
        return str(self.comboBox_2.currentText())

    def onLoadSample(self):
        if self.fileType in ("hdf"):
            filenames = [str(QtGui.QFileDialog.getOpenFileName(None, 'Open Photon-HDF', '', 'link file (*.h5)'))]
        else:
            directory = str(QtGui.QFileDialog.getExistingDirectory(self, "Select Directory"))
            filenames = [directory + '/' + s for s in os.listdir(directory)]
        self.filenames = filenames
        self._photons = Photons(filenames, self.fileType)
        self.samples = self._photons.samples
        self.comboBox.addItems(self._photons.sample_names)

    @property
    def photons(self):
        return self._photons


class PDBLoad(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self)
        uic.loadUi("mfm/ui/proteinMCLoad.ui", self)
        self._data = None
        self._filename = ''

    def onLoadStructure(self, filename=None):
        if filename is None:
            filename = str(QtGui.QFileDialog.getOpenFileName(None, 'Open Structure', '', 'PDB-file (*.pdb)'))
        self.filename = filename
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


