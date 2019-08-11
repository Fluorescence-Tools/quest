from PyQt4 import QtCore, QtGui
from guiqwt.plot import CurveDialog
from guiqwt.builder import make
import numpy as np

from mfm.plots.plotbase import Plot


class ProteinMCPlot(Plot):

    name = "Trajectory-Plot"

    def __init__(self, fit):
        Plot.__init__(self)
        self.layout = QtGui.QVBoxLayout(self)

        self.trajectory = fit.model
        self.source = fit.model

        # RMSD - Curves
        top_left = QtGui.QFrame(self)
        top_left.setFrameShape(QtGui.QFrame.StyledPanel)
        l = QtGui.QVBoxLayout(top_left)

        top_right = QtGui.QFrame(self)
        top_right.setFrameShape(QtGui.QFrame.StyledPanel)
        r = QtGui.QVBoxLayout(top_right)

        splitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(top_left)
        splitter.addWidget(top_right)

        self.layout.addWidget(splitter)

        win = CurveDialog()
        self.rmsd_plot = win.get_plot()
        self.rmsd_plot.set_titles(ylabel='RMSD')
        self.rmsd_curve = make.curve([],  [], color="b", linewidth=1)
        self.rmsd_plot.add_item(self.rmsd_curve)
        l.addWidget(self.rmsd_plot)

        win = CurveDialog()
        self.drmsd_plot = win.get_plot()
        self.drmsd_plot.set_titles(ylabel='dRMSD')
        self.drmsd_curve = make.curve([],  [], color="r", linewidth=1)
        self.drmsd_plot.add_item(self.drmsd_curve)
        r.addWidget(self.drmsd_plot)

        # Energy - Curves
        top_left = QtGui.QFrame(self)
        top_left.setFrameShape(QtGui.QFrame.StyledPanel)
        l = QtGui.QVBoxLayout(top_left)

        top_right = QtGui.QFrame(self)
        top_right.setFrameShape(QtGui.QFrame.StyledPanel)
        r = QtGui.QVBoxLayout(top_right)

        splitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(top_left)
        splitter.addWidget(top_right)

        self.layout.addWidget(splitter)

        win = CurveDialog()
        self.fret_plot = win.get_plot()
        self.fret_plot.set_titles(ylabel='&chi;<sup>2</sup>')
        self.fret_curve = make.curve([],  [], color="m", linewidth=1)
        self.fret_plot.add_item(self.fret_curve)
        l.addWidget(self.fret_plot)

        win = CurveDialog()
        self.energy_plot = win.get_plot()
        self.energy_plot.set_titles(ylabel='Energy')
        self.energy_curve = make.curve([],  [], color="g", linewidth=1)
        self.energy_plot.add_item(self.energy_curve)
        r.addWidget(self.energy_plot)

    def update_all(self):

        rmsd = np.array(self.trajectory.rmsd)
        drmsd = np.array(self.trajectory.drmsd)
        energy = np.array(self.trajectory.energy)
        energy_fret = np.array(self.trajectory.chi2r)
        x = list(range(len(rmsd)))

        self.rmsd_curve.set_data(x, rmsd)
        self.drmsd_curve.set_data(x, drmsd)
        self.energy_curve.set_data(x, energy)
        self.fret_curve.set_data(x, energy_fret)

        self.energy_plot.do_autoscale()
        self.fret_plot.do_autoscale()
        self.rmsd_plot.do_autoscale()
        self.drmsd_plot.do_autoscale()
