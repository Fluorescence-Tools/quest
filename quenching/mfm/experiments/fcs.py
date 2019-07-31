from copy import deepcopy

from PyQt4 import QtGui, QtCore, uic
import numpy as np

import mfm
from . import Setup
from mfm.curve import Genealogy
from mfm.fluorescence import fcs
from mfm.io.txt_csv import CsvWidget
from mfm.io.widgets import SpcFileWidget


class Correlator(QtCore.QThread):

    procDone = QtCore.pyqtSignal(bool)
    partDone = QtCore.pyqtSignal(int)

    @property
    def data(self):
        if isinstance(self._data, mfm.DataCurve):
            return self._data
        else:
            return mfm.DataCurve()

    def __init__(self, parent):
        QtCore.QThread.__init__(self, parent)
        self.p = parent
        self.exiting = False
        self._data = None
        self._results = []
        self._dt1 = 0
        self._dt2 = 0

    def getWeightStream(self, tacWeighting):
        """
        :param tacWeighting: is either a list of integers or a numpy-array. If it's a list of integers\
        the integers correspond to channel-numbers. In this case all photons have an equal weight of one.\
        If tacWeighting is a numpy-array it should be of shape [max-routing, number of TAC channels]. The\
        array contains np.floats with weights for photons arriving at different TAC-times.
        :return: numpy-array with same length as photon-stream, each photon is associated to one weight.
        """
        print("Correlator:getWeightStream")
        photons = self.p.photon_source.donor_photons
        if type(tacWeighting) is list:
            print("channel-wise selection")
            print("Max-Rout: %s" % photons.nROUT)
            wt = np.zeros([photons.nROUT, photons.n_tac], dtype=np.float32)
            wt[tacWeighting] = 1.0
        elif type(tacWeighting) is np.ndarray:
            print("TAC-weighted")
            wt = tacWeighting
        w = fcs.getWeights(photons.rout, photons.tac, wt, photons.nPh)
        return w

    def run(self):
        data = mfm.DataCurve()

        w1 = self.getWeightStream(self.p.ch1)
        w2 = self.getWeightStream(self.p.ch2)
        print("Correlation running...")
        print("Correlation method: %s" % self.p.method)
        print("Fine-correlation: %s" % self.p.fine)
        print("Nbr. of correlations: %s" % self.p.split)
        photons = self.p.photon_source.donor_photons

        self._results = []
        n = len(photons)
        nGroup = n / self.p.split
        self.partDone.emit(0.0)
        for i in range(0, n - n % nGroup, nGroup):
            nbr = ((i + 1) / nGroup + 1)
            print("Correlation Nbr.: %s" % nbr)
            p = photons[i:i + nGroup]
            wi1, wi2 = w1[i:i + nGroup], w2[i:i + nGroup]
            if self.p.method == 'tp':
                np1, np2, dt1, dt2, tau, corr = fcs.tp(p.mt, p.tac, p.rout, p.cr_filter,
                                                          wi1, wi2, self.p.B, self.p.nCasc,
                                                          self.p.fine, photons.n_tac)
                cr = fcs.normalize(np1, np2, dt1, dt2, tau, corr, self.p.B)
                cr /= self.p.dt
                dur = float(min(dt1, dt2)) * self.p.dt / 1000  # seconds
                tau = tau.astype(np.float64)
                tau *= self.p.dt
                self._results.append([cr, dur, tau, corr])
            self.partDone.emit(float(nbr) / self.p.split * 100)

        # Calculate average correlations
        cors = []
        taus = []
        weights = []
        for c in self._results:
            cr, dur, tau, corr = c
            weight = self.weight(tau, corr, dur, cr)
            weights.append(weight)
            cors.append(corr)
            taus.append(tau)

        cor = np.array(cors)
        w = np.array(weights)

        data.x = np.array(taus).mean_xyz(axis=0)[1:]
        data.y = cor.mean_xyz(axis=0)[1:]
        data.ey = 1. / w.mean_xyz(axis=0)[1:]

        print("correlation done")
        self._data = data
        self.procDone.emit(True)
        self.exiting = True

    def weight(self, tau, cor, dur, cr):
        """
        tau-axis in milliseconds
        correlation amplitude
        dur = duration in seconds
        cr = count-rate in kHz
        """
        w = np.ones(tau.shape, dtype=np.float64)
        if self.p.weighting == 1:
            print("no-weighting")
            return w
        elif self.p.weighting == 0:
            print("Suren-weighting")
            w *= fcs.surenWeights(tau, cor, dur, cr)
            return w


class CorrelatorWidget(QtGui.QWidget):

    def __init__(self, parent, photon_source, ch1='0', ch2='8',
                 nCasc=25, B=30, split=1, weighting=0, fine=0):
        QtGui.QWidget.__init__(self)

        uic.loadUi('mfm/ui/experiments/correlatorWidget.ui', self)
        self.parent = parent
        self.cr = 0.0
        self.ch1 = ch1
        self.ch2 = ch2
        self.nCasc = nCasc
        self.B = B
        self.split = split
        self.weighting = weighting
        self.fine = fine
        self.photon_source = photon_source
        self.correlator_thread = Correlator(self)

        # fill widgets
        self.comboBox_3.addItems(fcs.weightCalculations)
        self.comboBox_2.addItems(fcs.correlationMethods)
        self.checkBox.setChecked(True)
        self.checkBox.setChecked(False)
        self.progressBar.setValue(0.0)

        # connect widgets
        self.connect(self.pushButton_3, QtCore.SIGNAL("clicked()"), self.correlator_thread.start)
        self.correlator_thread.partDone.connect(self.updatePBar)

    def updatePBar(self, val):
        self.progressBar.setValue(val)

    @property
    def data(self):
        return self.correlator_thread.data

    @property
    def dt(self):
        dt = self.photon_source.donor_photons.MTCLK
        if self.fine:
            dt /= self.photon_source.donor_photons.n_tac
        return dt

    @property
    def weighting(self):
        return self.comboBox_3.currentIndex()

    @weighting.setter
    def weighting(self, v):
        self.comboBox_3.setCurrentIndex(int(v))

    @property
    def ch1(self):
        return [int(x) for x in str(self.lineEdit_4.text()).split()]

    @ch1.setter
    def ch1(self, v):
        self.lineEdit_4.setText(str(v))

    @property
    def ch2(self):
        return [int(x) for x in str(self.lineEdit_5.text()).split()]

    @ch2.setter
    def ch2(self, v):
        self.lineEdit_5.setText(str(v))

    @property
    def fine(self):
        return int(self.checkBox.isChecked())

    @fine.setter
    def fine(self, v):
        self.checkBox.setCheckState(v)

    @property
    def B(self):
        return int(self.lineEdit_2.text())

    @B.setter
    def B(self, v):
        return self.lineEdit_2.setText(str(v))

    @property
    def nCasc(self):
        return int(self.lineEdit.text())

    @nCasc.setter
    def nCasc(self, v):
        return self.lineEdit.setText(str(v))

    @property
    def method(self):
        return str(self.comboBox_2.currentText())

    @property
    def split(self):
        return int(self.lineEdit_3.text())

    @split.setter
    def split(self, v):
        self.lineEdit_3.setText(str(v))


class CrFilterWidget(QtGui.QWidget):
    def __init__(self, parent, photon_source, cunk_size=60000, time_window=10000,
                 tolerance=3, verbose=False):
        QtGui.QWidget.__init__(self)
        uic.loadUi('mfm/ui/experiments/fcs-cr-filter.ui', self)
        self.photon_source = photon_source
        self.parent = parent

        self.verbose = verbose
        self.sample_name = self.photon_source.sampleName
        self.cunk_size = cunk_size
        self.tolerance = tolerance
        self.time_window = time_window

    @property
    def cunk_size(self):
        return float(self.lineEdit_6.text())

    @cunk_size.setter
    def cunk_size(self, v):
        self.lineEdit_6.setText(str(v))

    @property
    def time_window(self):
        return float(self.lineEdit_7.text())

    @time_window.setter
    def time_window(self, v):
        self.lineEdit_7.setText(str(v))

    @property
    def tolerance(self):
        return float(self.lineEdit_8.text())

    @tolerance.setter
    def tolerance(self, v):
        self.lineEdit_8.setText(str(v))

    @property
    def cr_filter_on(self):
        return bool(self.groupBox_2.isChecked())

    @property
    def photons(self):
        photons = self.photon_source.donor_photons
        if self.cr_filter_on:
            dt = photons.MTCLK
            nWindow = int(self.time_window / dt)
            nChunk = int(self.cunk_size / dt)
            tolerance = self.tolerance
            if self.verbose:
                print("Using count-rate filter:")
                print("Window-size [n(MTCLK)]: %s" % (nWindow))
                print("Chunk-size [n(MTCLK)]: %s" % (nChunk))
            filter = fcs.crFilter(photons.mt, photons.nPh, nWindow, nChunk, tolerance)
            photons.setCrFilter(filter)
            return photons
        else:
            return photons


class FCS(QtGui.QWidget, Genealogy, Setup):
    def __init__(self, name, experiment, parent=None):
        QtGui.QWidget.__init__(self)
        Genealogy.__init__(self)
        self.experiment = experiment
        self.hide()
        self.name = name
        self.parent = parent
        self.fit = mfm.FitQtThread()

    def autofitrange(self, data):
        return 0, len(data.data_x) - 1


class FCSCsv(FCS, CsvWidget):

    name = 'FCS-CSV'

    def __init__(self, name, experiment, **kwargs):
        QtGui.QWidget.__init__(self)
        FCS.__init__(self, name, experiment)
        CsvWidget.__init__(self, **kwargs)
        self.skiprows = 0
        self.use_header = False
        self.spinBox.setEnabled(False)
        self.parent = kwargs.get('parent', None)
        self.name = name

    def load_data(self, filename=None):
        d = mfm.DataCurve()
        d.setup = self

        if filename is None:
            self.load(skiprows=0, use_header=None, verbose=True)
        d.filename = self.filename

        # In Kristine file-type
        d.x, d.y = self.data[0], self.data[1]
        dur, cr = self.data[2, 0], self.data[2, 1]

        w = fcs.surenWeights(d.x, d.y, dur, cr)
        d.set_weights(w)
        return d


class FCStttr(FCS, QtGui.QWidget):
    def __init__(self, name, experiment, parent=None):
        FCS.__init__(self, name, experiment, parent)
        self.parent = parent
        layout = QtGui.QVBoxLayout(self)
        self.layout = layout
        self.layout.setMargin(0)
        self.layout.setSpacing(0)
        self.fileWidget = SpcFileWidget(self)
        self.countrateFilterWidget = CrFilterWidget(self, self.fileWidget)
        self.correlator = CorrelatorWidget(self, self.countrateFilterWidget)
        self.layout.addWidget(self.fileWidget)
        self.layout.addWidget(self.countrateFilterWidget)
        self.layout.addWidget(self.correlator)

    def load_data(self, **kwargs):
        d = self.correlator.data
        d.name = self.fileWidget.sampleName
        return deepcopy(d)

