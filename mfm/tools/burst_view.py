import glob

import numpy as np
from PyQt4 import QtCore, QtGui, uic
from guiqwt.plot import CurveDialog
from guiqwt.builder import make

import mfm


class Burst(object):

    @property
    def number_of_photons(self):
        return len(self.photons)

    @property
    def duration(self):
        return (self.photons.mt[-1] - self.photons.mt[0]) * self.photons.MTCLK

    @property
    def mean_delay(self):
        return self.photons.tac.mean_xyz() * self.photons.dt * 1e6

    @property
    def std_delay(self):
        return self.photons[0:10].tac.std() * self.photons.dt * 1e6

    def __init__(self, photons):
        self.photons = photons


class Bursts(object):

    def __init__(self, photons, bids):
        self.photons = photons
        self.bids = bids
        self.bursts = [Burst(photons[bid]) for bid in bids]
        self.delay_times = np.array([b.mean_delay for b in self.bursts])
        self.mean_delay_time = self.delay_times.mean_xyz()
        self.std_delay_time = self.delay_times.std()
        self.number_of_photons = np.array([b.number_of_photons for b in self.bursts])
        self.mean_number_of_photons = self.number_of_photons.mean_xyz()
        self.std_number_of_photons = self.number_of_photons.std()
        self.durations = np.array([b.duration for b in self.bursts])
        self.mean_duration = self.durations.mean_xyz()
        self.std_duration = self.durations.std()

    def __len__(self):
        return len(self.bursts)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.bursts[key]
        elif isinstance(key, np.ndarray):
            return self.bursts[key]
        else:
            start = None if key.start is None else key.start
            stop = None if key.stop is None else key.stop
            step = None if key.step is None else key.step
            return self.bursts[start:stop:step]


class BurstViewWidget(QtGui.QWidget):

    name = "Burst-View"

    def __init__(self, **kwargs):
        QtGui.QWidget.__init__(self)
        uic.loadUi('mfm/ui/tools/burst_view.ui', self)
        self.connect(self.actionOpen_HDF_File, QtCore.SIGNAL('triggered()'), self.onOpenHDF_file)
        self.connect(self.actionOpen_burst_IDs, QtCore.SIGNAL('triggered()'), self.onOpenBurstIDs)
        self.connect(self.actionCurrent_burst_changed, QtCore.SIGNAL('triggered()'), self.onCurrentBurstChanged)

        self.verbose = kwargs.get('verbose', mfm.verbose)
        self._photons = None
        self._bursts = None

        # Make plots
        # Burst duration
        fd = CurveDialog(edit=False, toolbar=False)
        plot = fd.get_plot()
        self.burst_duration_plot = make.histogram([1])
        plot.add_item(self.burst_duration_plot)
        self.gridLayout_4.addWidget(fd, 0, 0)
        # Number of photons
        fd = CurveDialog(edit=False, toolbar=False)
        plot = fd.get_plot()
        self.number_of_photons_plot = make.histogram([1])
        plot.add_item(self.number_of_photons_plot)
        self.gridLayout_4.addWidget(fd, 0, 1)
        # Mean burst delay times
        fd = CurveDialog(edit=False, toolbar=False)
        plot = fd.get_plot()
        self.delay_times_plot = make.histogram([1])
        plot.add_item(self.delay_times_plot)
        self.gridLayout_4.addWidget(fd, 1, 0)
        # Current burst decay
        fd = CurveDialog(edit=False, toolbar=False)
        plot = fd.get_plot()
        self.burst_decay_plot = make.histogram([1])
        plot.add_item(self.burst_decay_plot)
        self.gridLayout_4.addWidget(fd, 1, 1)

    @property
    def number_of_bursts(self):
        return len(self._bursts)

    @property
    def photons(self):
        self._photons.sample = self.current_sample_name
        return self._photons

    @photons.setter
    def photons(self, v):
        self._photons = mfm.io.photons.Photons(v, file_type='hdf')

    @property
    def current_burst_id(self):
        return int(self.spinBox.value()) - 1

    @property
    def joint_bursts(self):
        burst = Burst(self._joint_photons)
        return burst

    @property
    def current_burst(self):
        return self._bursts[self.current_burst_id]

    @property
    def burst_ids(self):
        return self._burst_ids

    @property
    def current_sample_name(self):
        return str(self.comboBox.currentText())

    @burst_ids.setter
    def burst_ids(self, directory):
        filenames = glob.glob(directory + '/*.bst')
        self._burst_ids = mfm.io.photons.read_BIDs(filenames)
        self._bursts = Bursts(self.photons, self._burst_ids)
        if self.verbose:
            print self._burst_ids

    @property
    def mean_delay(self):
        return self._bursts.mean_delay_time

    @property
    def std_delay(self):
        return self._bursts.std_delay_time

    def onOpenHDF_file(self):
        print "onOpenHDF_file"
        hdf_filename = str(QtGui.QFileDialog.getOpenFileName(None, 'Open HDF-Photon file', '', 'H5-files (*.h5)'))
        self.photons = hdf_filename
        self.lineEdit_9.setText(hdf_filename)
        self.comboBox.clear()
        self.comboBox.addItems(self._photons.sample_names)

    def onOpenBurstIDs(self, directory=None):
        print "onOpenBurstIDs"
        if directory is None:
            directory = str(QtGui.QFileDialog.getExistingDirectory(None, 'Open BST-directory'))
            self.burst_ids = directory
        self.lineEdit_8.setText(directory)
        self.spinBox.setMaximum(self.number_of_bursts)
        self.spinBox.setMinimum(1)
        self.lineEdit_3.setText("%.3f" % self.mean_delay)
        self.lineEdit_4.setText("%.3f" % self.std_delay)
        self.lineEdit.setText(str(self.number_of_bursts))
        self.lineEdit_6.setText("%i" % self._bursts.mean_number_of_photons)
        self.lineEdit_7.setText("%i" % self._bursts.std_number_of_photons)

        self.lineEdit_13.setText("%.3f" % self._bursts.mean_duration)
        self.lineEdit_14.setText("%.3f" % self._bursts.std_duration)

        self.delay_times_plot.set_hist_data(self._bursts.delay_times)
        self.burst_duration_plot.set_hist_data(self._bursts.durations)
        self.number_of_photons_plot.set_hist_data(self._bursts.number_of_photons)

    def onCurrentBurstChanged(self):
        burst = self.current_burst
        self.lineEdit_5.setText(str(burst.number_of_photons))
        self.lineEdit_2.setText("%.2f" % burst.mean_delay)
        self.lineEdit_12.setText("%.2f" % burst.duration)
        self.burst_decay_plot.set_hist_data(self.current_burst.donor_photons.tac)



if __name__ == "__main__":
    import sys
    from PyQt4 import QtGui

    app = QtGui.QApplication(sys.argv)

    widget = BurstViewWidget()
    widget.setWindowTitle('BurstViewer')
    widget.show()

    sys.exit(app.exec_())