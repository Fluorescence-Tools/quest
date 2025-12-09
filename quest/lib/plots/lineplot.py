from qtpy import QtGui, QtCore, QtWidgets, uic
import numpy as np
import pyqtgraph as pg

from lib.plots.plotbase import Plot
from lib.math import autocorr


class _PGPlotAdapter:
    def __init__(self, title=None, xlabel=None, ylabel=None):
        self.widget = pg.PlotWidget(title=title)
        if xlabel:
            self.widget.setLabel('bottom', xlabel)
        if ylabel:
            self.widget.setLabel('left', ylabel)
        self._plot_item = self.widget.getPlotItem()

    def add_item(self, item):
        self._plot_item.addItem(item)

    def create_curve(self, pen=None, name=None):
        return self._plot_item.plot([], [], pen=pen, name=name)

    def set_scales(self, x_scale, y_scale):
        self.widget.setLogMode(x_scale == 'log', y_scale == 'log')

    def do_autoscale(self):
        self._plot_item.enableAutoRange(True, True)
        self._plot_item.autoRange()


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

        # Data-Fit plot
        fd = _PGPlotAdapter()
        pen_data = pg.mkPen(color='b', width=1)
        pen_irf = pg.mkPen(color='r', width=1)
        pen_model = pg.mkPen(color='g', width=4)
        self.data_curve = fd.create_curve(pen=pen_data, name="data")
        self.irf_curve = fd.create_curve(pen=pen_irf, name="irf")
        self.model_curve = fd.create_curve(pen=pen_model, name="model")
        self.dataPlot = fd
        botl.addWidget(fd.widget)

        splitter1 = QtGui.QSplitter(QtCore.Qt.Horizontal)
        topl.addWidget(splitter1)

        # Residual plot
        win = _PGPlotAdapter()
        self.residual_curve = win.create_curve(pen=pg.mkPen(color='r', width=2))
        self.chi2_label = pg.TextItem("")
        self.chi2_label.setAnchor((1, 0))
        self.chi2_label.setPos(0, 0)
        win.add_item(self.chi2_label)
        title = pg.TextItem("w.res.")
        title.setPos(0, 0)
        win.add_item(title)
        self.residualPlot = win
        splitter1.addWidget(win.widget)

        win = _PGPlotAdapter()
        self.autocorr_curve = win.create_curve(pen=pg.mkPen(color='r', width=2))
        title = pg.TextItem("auto.cor.")
        title.setPos(0, 0)
        win.add_item(title)
        self.autoCorrPlot = win
        splitter1.addWidget(win.widget)

        self.pltControl = linePlotWidget(self, d_scalex, d_scaley, r_scalex, r_scaley)

    def updateAll(self):
        print("TCSPCPlot:update")
        fit = self.fit
        self.chi2_label.setText("χ²=%.4f" % fit.chi2r)
        if fit is not None:

            data_x, data_y = fit.data.x, fit.data.y
            idx = np.where(data_y > 0.0)[0] if self.pltControl.data_logy else list(range(len(data_x)))
            self.data_curve.setData(data_x[idx],  data_y[idx])

            try:
                irf = fit.model.convolve.irf
                irf_y = irf.y
                idx = np.where(irf_y > 0.0)[0] if self.pltControl.data_logy else list(range(len(data_x)))
                self.irf_curve.setData(irf.x[idx],  irf_y[idx])
            except AttributeError:
                print("No instrument response to plot.")
            try:
                model_x, model_y = fit[:]
                idx = np.where(model_y > 0.0)[0] if self.pltControl.data_logy else list(range(len(data_x)))
                if len(idx) > 0:
                    self.model_curve.setData(model_x[idx],  model_y[idx])
            except ValueError:
                print("No model/no fitted model to plot")
            try:
                wres_y = fit.weighted_residuals
                self.residual_curve.setData(model_x, wres_y)
                if len(wres_y) > 0:
                    ac = autocorr(wres_y)
                    self.autocorr_curve.setData(model_x[1::], ac[1:])

            except (TypeError, AttributeError):
                pass
            self.residualPlot.do_autoscale()
            self.dataPlot.do_autoscale()
            self.autoCorrPlot.do_autoscale()
        print("END:TCSPCPlot:update")
