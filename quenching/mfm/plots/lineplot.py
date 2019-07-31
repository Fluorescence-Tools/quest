from PyQt4 import QtGui, QtCore, uic
import numpy as np
from guiqwt.plot import CurveDialog
from guiqwt.builder import make

import mfm
from mfm.plots import plotbase


class LinePlotControl(QtGui.QWidget):
    def __init__(self, parent=None, d_scalex='lin', d_scaley='log', r_scalex='lin', r_scaley='lin'):
        QtGui.QWidget.__init__(self)
        uic.loadUi('mfm/ui/plots/linePlotWidget.ui', self)
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
        """
        y-data is plotted logarithmically
        """
        return 'log' if self.checkBox.isChecked() else 'lin'

    @data_logy.setter
    def data_logy(self, v):
        if v == 'lin':
            self.checkBox.setCheckState(0)
        else:
            self.checkBox.setCheckState(2)

    @property
    def data_logx(self):
        """
        x-data is plotted logarithmically
        """
        return 'log' if self.checkBox_2.isChecked() else 'lin'

    @data_logx.setter
    def data_logx(self, v):
        if v == 'lin':
            self.checkBox_2.setCheckState(0)
        else:
            self.checkBox_2.setCheckState(2)

    @property
    def res_logx(self):
        """
        x-residuals is plotted logarithmically
        """
        return 'log' if self.checkBox_3.isChecked() else 'lin'

    @res_logx.setter
    def res_logx(self, v):
        if v == 'lin':
            self.checkBox_3.setCheckState(0)
        else:
            self.checkBox_3.setCheckState(2)

    @property
    def res_logy(self):
        """
        y-residuals is plotted logarithmically
        """
        return 'log' if self.checkBox_4.isChecked() else 'lin'

    @res_logy.setter
    def res_logy(self, v):
        if v == 'lin':
            self.checkBox_4.setCheckState(0)
        else:
            self.checkBox_4.setCheckState(2)

    def SetLog(self):
        self.parent.residualPlot.set_scales(self.res_logx, self.res_logy)
        self.parent.autoCorrPlot.set_scales(self.res_logx, self.res_logy)
        self.parent.dataPlot.set_scales(self.data_logx, self.data_logy)


class LinePlot(plotbase.Plot):
    """
    Started off as a plotting class to display TCSPC-data displaying the IRF, the experimental data, the residuals
    and the autocorrelation of the residuals. Now it is also used also for FCS-data.

    In case the model is a :py:class:`~experiment.model.tcspc.LifetimeModel` it takes the irf and displays it:

        irf = fit.model.convolve.irf
        irf_y = irf.y

    The model data and the weighted residuals are taken directly from the fit:

        model_x, model_y = fit[:]
        wres_y = fit.weighted_residuals

    """

    name = "Fit"

    def __init__(self, fit, d_scalex='lin', d_scaley='lin', r_scalex='lin', r_scaley='lin'):
        mfm.plots.Plot.__init__(self)
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
        fd = CurveDialog(edit=False, toolbar=False)
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

        self.pltControl = LinePlotControl(self, d_scalex, d_scaley, r_scalex, r_scaley)

    def update_all(self):
        fit = self.fit
        self.chi2_label.set_text("<b>&Chi;<sup>2</sup>=%.4f</b>" % fit.chi2r)
        if isinstance(fit, mfm.Fit):
            data_x, data_y = fit.data.x, fit.data.y

            ## Model function
            try:
                model_x, model_y = fit[:]
            except ValueError:
                model_x, model_y = np.ones(10), np.ones(10)
                print("No model/no fitted model to plot")

            ## IRF
            try:
                irf = fit.model.convolve.irf
                irf_y = irf.y
                irf_x = data_x
            except AttributeError:
                print("No instrument response to plot.")
                irf_y = np.ones(10)
                irf_x = np.ones(10)

            ## Weighted residuals + Autocorrelation
            wres_y = fit.weighted_residuals
            ac_y = mfm.math.signal.autocorr(wres_y)

        else:
            irf_x, irf_y = np.ones(10), np.ones(10)
            data_x, data_y = np.ones(10), np.ones(10)
            model_x, model_y = np.ones(10), np.ones(10)
            wres_x, wres_y = np.ones(10), np.ones(10)

        idx = np.where(irf_y > 0.0)[0] if self.pltControl.data_logy else list(range(len(irf_x)))
        self.irf_curve.set_data(irf_x[idx],  irf_y[idx])

        idx = np.where(model_y > 0.0)[0] if self.pltControl.data_logy else list(range(len(model_x)))
        self.model_curve.set_data(model_x[idx],  model_y[idx])

        idx = np.where(data_y > 0.0)[0] if self.pltControl.data_logy else list(range(len(data_x)))
        self.data_curve.set_data(data_x[idx],  data_y[idx])

        self.residual_curve.set_data(model_x, wres_y)
        self.autocorr_curve.set_data(model_x[1::], ac_y[1:])

        try:
            self.residualPlot.do_autoscale()
            self.autoCorrPlot.do_autoscale()
        except ValueError:
            print "Zero sized weighted residuals no plot update"
        self.dataPlot.do_autoscale()

