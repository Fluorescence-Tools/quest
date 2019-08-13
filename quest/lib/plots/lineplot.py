from PyQt5 import QtGui, QtCore, uic, QtWidgets
import numpy as np
from guiqwt.plot import CurveDialog
from guiqwt.builder import make

from lib.plots.plotbase import Plot
from lib.math.functions import autocorr


class linePlotWidget(QtWidgets.QWidget):
    def __init__(self, parent=None, d_scalex='lin', d_scaley='log', r_scalex='lin', r_scaley='lin'):
        QtWidgets.QWidget.__init__(self)
        uic.loadUi('experiments/plots/ui/linePlotWidget.ui', self)
        self.parent = parent

        self.connect(self.checkBox, QtCore.SIGNAL("stateChanged (int)"), self.SetLog)
        self.connect(self.checkBox_2, QtCore.SIGNAL("stateChanged (int)"), self.SetLog)
        self.connect(self.checkBox_3, QtCore.SIGNAL("stateChanged (int)"), self.SetLog)
        self.connect(self.checkBox_4, QtCore.SIGNAL("stateChanged (int)"), self.SetLog)

        self.data_logy = d_scaley
        self.data_logx = d_scalex
        self.res_logx = r_scalex
        self.res_logy = r_scaley

    @property
    def data_logy(self):
        return 'log' if self.checkBox.isChecked() else 'lin'

    @data_logy.setter
    def data_logy(self, v):
        if v == 'lin':
            self.checkBox.setCheckState(0)
        else:
            self.checkBox.setCheckState(2)

    @property
    def data_logx(self):
        return 'log' if self.checkBox_2.isChecked() else 'lin'

    @data_logx.setter
    def data_logx(self, v):
        if v == 'lin':
            self.checkBox_2.setCheckState(0)
        else:
            self.checkBox_2.setCheckState(2)

    @property
    def res_logx(self):
        return 'log' if self.checkBox_3.isChecked() else 'lin'

    @res_logx.setter
    def res_logx(self, v):
        if v == 'lin':
            self.checkBox_3.setCheckState(0)
        else:
            self.checkBox_3.setCheckState(2)

    @property
    def res_logy(self):
        return 'log' if self.checkBox_4.isChecked() else 'lin'

    @res_logy.setter
    def res_logy(self, v):
        if v == 'lin':
            self.checkBox_4.setCheckState(0)
        else:
            self.checkBox_4.setCheckState(2)

    def SetLog(self):
        print("SetLog")
        self.parent.residualPlot.set_scales(self.res_logx, self.res_logy)
        self.parent.autoCorrPlot.set_scales(self.res_logx, self.res_logy)
        self.parent.dataPlot.set_scales(self.data_logx, self.data_logy)


class LinePlot(Plot):

    name = "Fit"

    def __init__(self, fit, d_scalex='lin', d_scaley='lin', r_scalex='lin', r_scaley='lin'):
        Plot.__init__(self)
        self.layout = QtGui.QVBoxLayout(self)
        self.fit = fit

        bottom = QtGui.QFrame(self)
        bottom.setFrameShape(QtGui.QFrame.StyledPanel)
        botl = QtGui.QVBoxLayout(bottom)

        top = QtGui.QFrame(self)
        top.setMaximumHeight(140)
        top.setFrameShape(QtGui.QFrame.StyledPanel)
        topl = QtGui.QVBoxLayout(top)

        splitter1 = QtGui.QSplitter(QtCore.Qt.Vertical)
        splitter1.addWidget(top)
        splitter1.addWidget(bottom)
        self.layout.addWidget(splitter1)

        # Data-Fit dialog
        fd = CurveDialog(edit=False, toolbar=True)
        #self.get_itemlist_panel().show()
        plot = fd.get_plot()
        self.data_curve = make.curve([1],  [1], color="b", linewidth=1)
        self.irf_curve = make.curve([1],  [1], color="r", linewidth=1)
        self.model_curve = make.curve([1],  [1], color="g", linewidth=4)
        plot.add_item(self.data_curve)
        plot.add_item(self.irf_curve)
        plot.add_item(self.model_curve)
        self.dataPlot = plot
        botl.addWidget(fd)

        splitter1 = QtGui.QSplitter(QtCore.Qt.Horizontal)
        topl.addWidget(splitter1)

        # Residual dialog
        win = CurveDialog(edit=False, toolbar=True)
        plot = win.get_plot()
        plot.do_autoscale(True)
        self.residual_curve = make.curve([1],  [1], color="r", linewidth=2)
        plot.add_item(self.residual_curve)
        self.chi2_label = make.label("", "R", (-10, 27), "R")
        plot.add_item(self.chi2_label)
        title = make.label("w.res.", "R", (0, -40), "R")
        plot.add_item(title)
        self.residualPlot = plot
        splitter1.addWidget(plot)

        win = CurveDialog(edit=False, toolbar=True)
        plot = win.get_plot()
        plot.do_autoscale(True)
        self.autocorr_curve = make.curve([1],  [1], color="r", linewidth=2)
        plot.add_item(self.autocorr_curve)
        title = make.label("auto.cor.", "R", (0, -40), "R")
        plot.add_item(title)
        self.autoCorrPlot = plot
        splitter1.addWidget(plot)

        self.pltControl = linePlotWidget(self, d_scalex, d_scaley, r_scalex, r_scaley)

    def updateAll(self):
        print("TCSPCPlot:update")
        fit = self.fit
        self.chi2_label.set_text("<b>&Chi;<sup>2</sup>=%.4f</b>" % fit.chi2r)
        if fit is not None:

            data_x, data_y = fit.data.x, fit.data.y
            idx = np.where(data_y > 0.0)[0] if self.pltControl.data_logy else list(range(len(data_x)))
            self.data_curve.set_data(data_x[idx],  data_y[idx])

            try:
                irf = fit.model.convolve.irf
                irf_y = irf.y
                idx = np.where(irf_y > 0.0)[0] if self.pltControl.data_logy else list(range(len(data_x)))
                self.irf_curve.set_data(irf.x[idx],  irf_y[idx])
            except AttributeError:
                print("No instrument response to plot.")
            try:
                model_x, model_y = fit[:]
                idx = np.where(model_y > 0.0)[0] if self.pltControl.data_logy else list(range(len(data_x)))
                if len(idx) > 0:
                    self.model_curve.set_data(model_x[idx],  model_y[idx])
            except ValueError:
                print("No model/no fitted model to plot")
            try:
                wres_y = fit.weighted_residuals
                self.residual_curve.set_data(model_x, wres_y)
                if len(wres_y) > 0:
                    ac = autocorr(wres_y)
                    self.autocorr_curve.set_data(model_x[1::], ac[1:])

            except (TypeError, AttributeError):
                pass
            self.residualPlot.do_autoscale()
            self.dataPlot.do_autoscale()
            self.autoCorrPlot.do_autoscale()
        print("END:TCSPCPlot:update")
