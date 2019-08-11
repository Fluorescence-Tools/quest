from PyQt4 import QtGui, QtCore
from guiqwt.plot import CurveDialog
from guiqwt.builder import make

from mfm.plots.plotbase import Plot


class GlobalFitPlot(Plot):
    name = "Global-Fits"

    def __init__(self, fit, logy=False, logx=False):
        Plot.__init__(self)
        self.layout = QtGui.QVBoxLayout(self)
        self.pltControl = None
        self.fit = fit
        self.logy = logy
        self.logx = logx

    def update_all(self):
        print("GlobalFitPlot:update")
        fit = self.fit
        layout = self.layout
        for i in reversed(list(range(layout.count()))):
            layout.itemAt(i).widget().deleteLater()

        splitter1 = QtGui.QSplitter(QtCore.Qt.Vertical)
        self.layout.addWidget(splitter1)
        xs = fit.model.xs()
        wrs = fit.model.weighted_residuals(data=fit.data, stack=False)
        for i, wr in enumerate(wrs):
            f = fit.model.fits[i]
            splitter1.addWidget(QtGui.QLabel(f.name))
            splitter2 = QtGui.QSplitter(QtCore.Qt.Horizontal)
            left = QtGui.QFrame(self)
            left.setFrameShape(QtGui.QFrame.StyledPanel)
            l = QtGui.QVBoxLayout(left)
            win = CurveDialog()
            w1_plot = win.get_plot()
            w1_curve = make.curve(xs[i], wr, color="r", linewidth=1)
            w1_plot.add_item(w1_curve)
            l.addWidget(w1_plot)
            splitter2.addWidget(left)

            right = QtGui.QFrame(self)
            right.setFrameShape(QtGui.QFrame.StyledPanel)
            r = QtGui.QVBoxLayout(right)
            win = CurveDialog()
            w2_plot = win.get_plot()
            w2_plot.set_scales(self.logx, self.logy)
            data_curve = make.curve(xs[i], f.data.y, color="b", linewidth=1)
            model_curve = make.curve(xs[i], f.model.y_values, color="g", linewidth=4)
            w2_plot.add_item(data_curve)
            w2_plot.add_item(model_curve)
            r.addWidget(w2_plot)
            splitter2.addWidget(right)

            splitter1.addWidget(splitter2)

            splitter1.addWidget(splitter2)

        print("END:GlobalFitPlot:update")
