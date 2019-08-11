import re

from PyQt4 import QtGui, QtCore, uic
import numpy as np

import mfm
from mfm.curve import Genealogy
from mfm.experiments import Setup
from mfm.io.txt_csv import Csv, CsvWidget
from mfm.io import sdtfile
from mfm.io.txt_csv import CsvWidget
from mfm.io.widgets import SpcFileWidget


class CsvTCSPC(object):

    def __init__(self, **kwargs):
        self.dt = kwargs.get('dt', 0.0141)
        self.rep_rate = kwargs.get('rep_rate', 15.0)
        self.is_jordi = kwargs.get('is_jordi', False)
        self.mode = kwargs.get('mode', 'vm')
        self.g_factor = kwargs.get('g_factor', 1.0)


class CsvTCSPCWidget(CsvTCSPC, QtGui.QWidget):

    def __init__(self, **kwargs):
        QtGui.QWidget.__init__(self)
        self.parent = kwargs.get('parent', None)
        uic.loadUi('mfm/ui/experiments/csvTCSPCWidget.ui', self)
        CsvTCSPC.__init__(self, **kwargs)

    @property
    def g_factor(self):
        return float(self.doubleSpinBox_3.value())

    @g_factor.setter
    def g_factor(self, v):
        return self.doubleSpinBox_3.setValue(v)

    @property
    def mode(self):
        if self.radioButton_3.isChecked():
            return 'vv'
        elif self.radioButton_2.isChecked():
            return 'vh'
        elif self.radioButton.isChecked():
            return 'vm'

    @mode.setter
    def mode(self, v):
        if v == 'vv':
            self.radioButton_3.setChecked(True)
        elif v == 'vh':
            self.radioButton_2.setChecked(True)
        elif v == 'vm':
            self.radioButton.setChecked(True)

    @property
    def is_jordi(self):
        return bool(self.checkBox_3.isChecked())

    @is_jordi.setter
    def is_jordi(self, v):
        return self.checkBox_3.setChecked(v)

    @property
    def rep_rate(self):
        return self.doubleSpinBox.value()

    @rep_rate.setter
    def rep_rate(self, rr):
        self.doubleSpinBox.setValue(rr)

    @property
    def dt(self):
        if self.checkBox_2.isChecked():
            return float(self.doubleSpinBox_2.value())
        else:
            return 1.0

    @dt.setter
    def dt(self, v):
        return self.doubleSpinBox_2.setValue(v)


class TcspcTTTRWidget(QtGui.QWidget):

    def __init__(self, parent):
        QtGui.QWidget.__init__(self, parent)
        self.parent = parent
        self.tcspcTTTRWidget = QtGui.QWidget(self)
        uic.loadUi('mfm/ui/experiments/tcspcTTTRWidget.ui', self.tcspcTTTRWidget)
        self.spcFileWidget = SpcFileWidget(self)
        layout = QtGui.QVBoxLayout(self)
        layout.setMargin(0)
        layout.setSpacing(0)
        layout.addWidget(self.spcFileWidget)
        layout.addWidget(self.tcspcTTTRWidget)

        self.connect(self.tcspcTTTRWidget.comboBox, QtCore.SIGNAL("currentIndexChanged(int)"), self.onTacDivChanged)
        self.connect(self.spcFileWidget.actionLoad_sample, QtCore.SIGNAL("triggered()"), self.onLoadFile)
        self.connect(self.spcFileWidget.actionDt_changed, QtCore.SIGNAL("triggered()"), self.onTacDivChanged)

    @property
    def nPh(self):
        return int(self.tcspcTTTRWidget.lineEdit_5.text())

    @nPh.setter
    def nPh(self, v):
        self.tcspcTTTRWidget.lineEdit_5.setText("%d" % v)

    @property
    def div(self):
        return int(self.tcspcTTTRWidget.comboBox.currentText())

    @property
    def rep_rate(self):
        return self.spcFileWidget.rep_rate

    @property
    def dt_min(self):
        return float(self.tcspcTTTRWidget.doubleSpinBox.value())

    @property
    def use_dtmin(self):
        return self.tcspcTTTRWidget.checkBox.isChecked()

    @property
    def histSelection(self):
        return str(self.tcspcTTTRWidget.lineEdit.text()).replace(" ", "").upper()

    @property
    def inverted_selection(self):
        return self.tcspcTTTRWidget.checkBox_2.isChecked()

    @property
    def nTAC(self):
        return int(self.tcspcTTTRWidget.lineEdit_4.text())

    @nTAC.setter
    def nTAC(self, v):
        self.tcspcTTTRWidget.lineEdit_4.setText("%d" % v)

    def makeHist(self):
        # get right data
        h5 = self.spcFileWidget._photons.h5
        nodeName = str(self.spcFileWidget.comboBox.currentText())
        table = h5.get_node('/' + nodeName, 'photons')
        selection_tac = np.ma.array([row['TAC'] for row in table.where(self.histSelection)])[:-1]

        if self.use_dtmin:
            if self.inverted_selection:
                print("inverted selection")
                selection_mask = np.diff(np.array([row['MT'] for row in table.where(self.histSelection)])) < self.dt_min
            else:
                print("normal selection")
                selection_mask = np.diff(np.array([row['MT'] for row in table.where(self.histSelection)])) > self.dt_min
            print("dMTmin: %s" % self.dt_min)
            selection_tac.mask = selection_mask
            self.nPh = selection_mask.sum()
        else:
            self.nPh = selection_tac.shape[0]
        print("nPh: %s" % self.nPh)

        ta = selection_tac.compressed().astype(np.int32)
        ta /= self.div
        hist = np.bincount(ta, minlength=self.nTAC)
        self.y = hist.astype(np.float64)
        self.x = np.arange(len(hist), dtype=np.float64) + 1.0
        self.x *= self.dt
        self.xt = self.x

        ex = r'(ROUT==\d+)'
        routCh = re.findall(ex, self.histSelection)
        self.chs = [int(ch.split('==')[1]) for ch in routCh]
        self.tcspcTTTRWidget.lineEdit_3.setText("%s" % self.chs)
        curve = mfm.DataCurve()
        curve.x = self.x
        curve.y = self.y
        self.emit(QtCore.SIGNAL('histDone'), self.nROUT, self.nTAC, self.chs, curve)

    def onTacDivChanged(self):
        self.dtBase = self.spcFileWidget.dt
        self.tacDiv = float(self.tcspcTTTRWidget.comboBox.currentText())
        self.nTAC = (self.spcFileWidget.nTAC + 1) / self.tacDiv
        self.dt = self.dtBase * self.tacDiv
        self.tcspcTTTRWidget.lineEdit_2.setText("%.3f" % self.dt)

    def onLoadFile(self):
        self.nROUT = self.spcFileWidget.nROUT
        self.onTacDivChanged()


class TCSPCSetup(Genealogy, Setup):

    def __init__(self, **kwargs):
        """

        Example
        -------
        >>> from mfm import experiments
        >>> tcspc_setup = experiments.TCSPCSetup()
        >>> print tcspc_setup
        TCSPCSetup:
          Name:         TCSPC-CSV
           dt [ns]:             0.01
           repetion rate [MHz]:         15.0
           TAC channels:        0
        >>> tcspc_setup = experiments.TCSPCSetup(rep_rate=72.0, dt=0.0141)
        >>> print tcspc_setup
        TCSPCSetup:
          Name:         TCSPC-CSV
           dt [ns]:             0.01
           repetion rate [MHz]:         72.0
           TAC channels:        0
        >>> data_set = tcspc_setup.load_data('./sample_data/ibh/Decay_577D.txt', verbose=True)
        >>> data_set.x
        array([  2.82000000e-02,   4.23000000e-02,   5.64000000e-02, ...,
         5.77254000e+01,   5.77395000e+01,   5.77536000e+01])
        >>> data_set.y
        array([ 0.,  0.,  0., ...,  0.,  0.,  0.])
        >>> max(data_set.y)
        50010.0

        :param kwargs:
        :return:
        """
        Genealogy.__init__(self)
        self.parent = kwargs.get('parent', None)
        self.name = kwargs.get('name', 'TCSPC:CSV')
        self.verbose = kwargs.get('verbose', mfm.verbose)
        self.csvTCSPC = CsvTCSPC(**kwargs)
        self.csvSetup = Csv(**kwargs)

    @property
    def skiprows(self):
        return self.csvSetup.skiprows

    @property
    def rep_rate(self):
        return self.csvTCSPC.rep_rate

    @staticmethod
    def autofitrange(data, threshold=5.0, area=0.999):
        return mfm.fluorescence.tcspc_fitrange(data.y, threshold, area)

    def load_data(self, filename=None, verbose=False, skiprows=None):

        """
        Loads data contained in csv-file into an mfm.DataCurve object
        :param skiprows: int or None
            if skiprows is None the first number of lines specified by the attribute
             skiptrows are used.
        :param filename: string
            Filename containing TCSPC-data (csv)
        :param verbose: bool
            Additional output on std-out. Default-value False
        :return: mfm.DataCurve

        Example
        -------
        >>> import experiments
        >>> tcspc_setup = experiments.TCSPCSetup()
        >>> data_set = tcspc_setup.load('./sample_data/ibh/Decay_577D.txt', verbose=True)
        >>> len(data_set)
        4095
        By skiprows the header and the first lines may be skipped
        >>> data_set = tcspc_setup.load('./sample_data/ibh/Decay_577D.txt', verbose=True, skiprows=20)
        4084
        >>> len(data_set)
        >>> data_set.x
        array([  2.82000000e-02,   4.23000000e-02,   5.64000000e-02, ...,
         5.77254000e+01,   5.77395000e+01,   5.77536000e+01])
        >>> data_set.y
        array([ 0.,  0.,  0., ...,  0.,  0.,  0.])
        >>> max(data_set.y)
        50010.0
        >>> print data_set
        filename: ./sample_data/ibh/Decay_577D.txt
        length  : 4095
        2.820e-02       2.820e-02
        4.230e-02       4.230e-02
        5.640e-02       5.640e-02
        ....
        5.773e+01       5.773e+01
        5.774e+01       5.774e+01
        5.775e+01       5.775e+01
        >>> print len(data_set)
        Without skipping the header the wrong number of columns are obtained

        >>> data_set = tcspc_setup.load('./sample_data/ibh/Decay_577D.txt', verbose=True, skiprows=7)
        """
        verbose = self.verbose or verbose
        skiprows = self.skiprows if skiprows is None else skiprows
        if verbose:
            print("Loading data:")
        d = mfm.DataCurve()
        d.setup = self
        if self.csvTCSPC.is_jordi:
            if verbose:
                print("Jordi-File format")
            self.csvSetup.x_on = False
            self.csvSetup.skiprows = 0
        if verbose:
            print("Using data-file passed as argument")
            print("Filename: %s" % filename)
        self.csvSetup.load(filename, skiprows=skiprows)
        d.filename = self.csvSetup.filename
        x = self.csvSetup.data_x * self.csvTCSPC.dt
        y = self.csvSetup.data_y

        if self.csvTCSPC.is_jordi:
            n = len(x) / 2
            x = x[:n]
            y1, y2 = y[:n], y[n:]
            if self.csvTCSPC.mode == 'vv':
                y = y1
            elif self.csvTCSPC.mode == 'vh':
                y = y2
            elif self.csvTCSPC.mode == 'vm':
                y = y1 + 2 * self.csvTCSPC.g_factor * y2

        d.x, d.y = x, y
        d.x, d.y = x, y
        w = mfm.fluorescence.tcspc_weights(y)
        d.set_weights(w)
        return d

    def __str__(self):
        s = 'TCSPCSetup:\n'
        s += '  Name: \t%s \n' % self.name
        s += '   dt [ns]:         \t%.2f \n' % self.csvTCSPC.dt
        s += '   repetion rate [MHz]: \t%.1f \n' % self.csvTCSPC.rep_rate
        s += '   TAC channels: \t%s \n' % self.csvSetup.n_points
        return s


class TCSPCSetupWidget(QtGui.QWidget, TCSPCSetup):

    def __init__(self, **kwargs):
        QtGui.QWidget.__init__(self)
        TCSPCSetup.__init__(self, **kwargs)
        # Overwrite non-widget attributes by widgets
        self.csvTCSPC = CsvTCSPCWidget(**kwargs)
        self.csvSetup = CsvWidget(parent=self)

        layout = QtGui.QVBoxLayout(self)
        self.layout = layout
        self.layout.setMargin(0)
        self.layout.setSpacing(0)
        self.layout.addWidget(self.csvTCSPC)
        self.layout.addWidget(self.csvSetup)
        self.dt = self.csvTCSPC.dt


class TCSPCSetupTTTR(Genealogy, Setup):

    @staticmethod
    def autofitrange(data, threshold=5.0, area=0.999):
        return mfm.fluorescence.tcspc_fitrange(data.y, threshold, area)

    def __init__(self, **kwargs):
        Genealogy.__init__(self)
        self.name = kwargs.get('name', 'TCSPC:TTTR')
        self.parent = kwargs.get('parent', None)

    def __str__(self):
        s = "TCSPCSetupTTTR:"
        s += '  Name: \t%s \n' % self.name
        s += '   dt [ns]:         \t%.2f \n' % self.tcspcTTTR.dt
        s += '   repetion rate [MHz]: \t%.1f \n' % self.tcspcTTTR.rep_rate
        return s


class TCSPCSetupTTTRWidget(QtGui.QWidget, TCSPCSetupTTTR):

    def __init__(self, **kwargs):
        QtGui.QWidget.__init__(self, **kwargs)
        TCSPCSetupTTTR.__init__(self, **kwargs)
        layout = QtGui.QVBoxLayout(self)
        self.layout = layout
        self.layout.setMargin(0)
        self.layout.setSpacing(0)

        self.tcspcTTTR = TcspcTTTRWidget(self)
        self.rep_rate = self.tcspcTTTR.rep_rate
        self.layout.addWidget(self.tcspcTTTR)

    def load_data(self, filename=None):
        d = mfm.DataCurve()
        self.tcspcTTTR.makeHist()
        d.filename = self.tcspcTTTR.spcFileWidget.sampleName + "_" + str(self.tcspcTTTR.chs)
        x = self.tcspcTTTR.x
        y = self.tcspcTTTR.y
        w = mfm.fluorescence.tcspc_weights(y)
        d.x, d.y = x, y
        d.set_weights(w)
        return d


class TcspcSDTWidget(QtGui.QWidget):

    @property
    def name(self):
        name = str(self.lineEdit_2.text())
        if len(name) > 0:
            return name + " - " + self.filename
        else:
            return str(self.curve_number) + self.filename

    @property
    def n_curves(self):
        n_data_curves = len(self.sdt.data)
        return n_data_curves

    @property
    def curve_number(self):
        """
        The currently selected curves
        """
        return int(self.comboBox.currentIndex())

    @curve_number.setter
    def curve_number(self, v):
        return self.comboBox.setCurrentIndex(int(v))

    @property
    def filename(self):
        return str(self.lineEdit.text())

    @filename.setter
    def filename(self, v):
        self._sdt = sdtfile.SdtFile(v)
        # refresh GUI
        self.comboBox.clear()
        l = [str(i) for i in range(self.n_curves)]
        self.comboBox.addItems(l)
        self.lineEdit.setText(str(v))

    @property
    def sdt(self):
        if self._sdt is None:
            self.onOpenFile()
        return self._sdt

    @property
    def times(self):
        """
        The time-array in nano-seconds
        """
        x = self._sdt.times[0] * 1e9
        return np.array(x, dtype=np.float64)

    @property
    def ph_counts(self):
        y = self._sdt.data[self.curve_number][0]
        return np.array(y, dtype=np.float64)

    @property
    def rep_rate(self):
        return float(self.doubleSpinBox.value())

    @rep_rate.setter
    def rep_rate(self, v):
        self.doubleSpinBox.setValue(float(v))

    @property
    def curve(self):
        """
        The currently selected curve as a :py:class:`mfm.Curve` object
        """
        y = self.ph_counts
        w = mfm.fluorescence.tcspc_weights(y)
        d = mfm.DataCurve(x=self.times, y=y, weights=w, name=self.name)
        return d

    def onOpenFile(self):
        filename = str(QtGui.QFileDialog.getOpenFileName(None, 'Open BH-SDT file', '', 'SDT-files (*.sdt)'))
        self.filename = filename
        self.lineEdit.setToolTip(str(self.sdt.info))

    def __init__(self, **kwargs):
        QtGui.QWidget.__init__(self)
        uic.loadUi('mfm/ui/experiments/sdtfile.ui', self)
        self._sdt = None
        self.rep_rate = kwargs.get('rep_rate', 16.0)
        self.connect(self.actionOpen_SDT_file, QtCore.SIGNAL('triggered()'), self.onOpenFile)


class TCSPCSetupSDTWidget(QtGui.QWidget, TCSPCSetup):

    @property
    def rep_rate(self):
        return self.tcspcSDT.rep_rate

    def __init__(self, **kwargs):
        QtGui.QWidget.__init__(self, **kwargs)
        TCSPCSetup.__init__(self, **kwargs)
        layout = QtGui.QVBoxLayout(self)
        self.layout = layout
        self.layout.setMargin(0)
        self.layout.setSpacing(0)

        self.tcspcSDT = TcspcSDTWidget()
        self.layout.addWidget(self.tcspcSDT)
        self.name = kwargs.get('name', 'TCSPC:SDT')

    def load_data(self, **kwargs):
        return self.tcspcSDT.curve


class TCSPCSetupDummy(TCSPCSetup):

    name = "Dummy-TCSPC"

    def __init__(self, **kwargs):
        TCSPCSetup.__init__(self, **kwargs)
        self.parent = kwargs.get('parent', None)
        self.sample_name = kwargs.get('sample_name', 'Dummy-sample')
        self.name = kwargs.get('name', "TCSPC:Dummy")
        self.verbose = kwargs.get('verbose', mfm.verbose)
        self.lifetime = kwargs.get('lifetime', 4.1)
        self.n_TAC = kwargs.get('n_TAC', 4096)
        self.dt = kwargs.get('dt', 0.0141)
        self.p0 = kwargs.get('p0', 10000.0)

    def load_data(self, filename=None, **kwargs):

        d = mfm.DataCurve()
        d.setup = self

        x = np.arange(self.n_TAC) * self.dt
        y = np.exp(-x/self.lifetime) * self.p0

        d.filename = self.sample_name
        d.x, d.y = x, y
        d.x, d.y = x, y
        w = mfm.fluorescence.tcspc_weights(y)
        d.set_weights(w)
        return d

    def __str__(self):
        s = 'TCSPCSetup:\n'
        s += '  Name: \t%s \n' % self.name
        s += '   dt [ns]:         \t%.2f \n' % self.csvTCSPC.dt
        s += '   repetion rate [MHz]: \t%.1f \n' % self.csvTCSPC.rep_rate
        s += '   TAC channels: \t%s \n' % self.csvSetup.n_points
        return s


class TCSPCSetupDummyWidget(QtGui.QWidget, TCSPCSetupDummy):

    @property
    def sample_name(self):
        name = str(self.lineEdit.text())
        return name

    @sample_name.setter
    def sample_name(self, v):
        pass

    @property
    def p0(self):
        return self.spinBox_2.value()

    @p0.setter
    def p0(self, v):
        pass

    @property
    def lifetime(self):
        return self.doubleSpinBox_2.value()

    @lifetime.setter
    def lifetime(self, v):
        pass

    @property
    def n_TAC(self):
        return self.spinBox.value()

    @n_TAC.setter
    def n_TAC(self, v):
        pass

    @property
    def dt(self):
        return self.doubleSpinBox.value()

    @dt.setter
    def dt(self, v):
        pass

    def __init__(self, **kwargs):
        QtGui.QWidget.__init__(self)
        TCSPCSetupDummy.__init__(self, **kwargs)
        uic.loadUi('mfm/ui/experiments/tcspcDummy.ui', self)

