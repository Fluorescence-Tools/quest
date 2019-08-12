from PyQt5 import QtGui, QtCore, uic, QtWidgets
import numpy as np
from guiqwt.plot import CurveDialog, ImageDialog
from guiqwt.builder import make
from lib.plots.plotbase import Plot
from lib import VariableWidget


class SurfacePlotWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        self.xmin = VariableWidget('', 0.0, model=parent, hide_fix_checkbox=True, hide_link=True)
        self.xmax = VariableWidget('', 0.0, model=parent, hide_fix_checkbox=True, hide_link=True)
        self.ymin = VariableWidget('', 0.0, model=parent, hide_fix_checkbox=True, hide_link=True)
        self.ymax = VariableWidget('', 0.0, model=parent, hide_fix_checkbox=True, hide_link=True)

        QtWidgets.QWidget.__init__(self)
        uic.loadUi('experiments/plots/ui/chi2Hist.ui', self)
        self.horizontalLayout.addWidget(self.xmin)
        self.horizontalLayout.addWidget(self.xmax)
        self.horizontalLayout_2.addWidget(self.ymin)
        self.horizontalLayout_2.addWidget(self.ymax)

        self.parent = parent

        self.connect(self.comboBox, QtCore.SIGNAL("currentIndexChanged(int)"), self.onAutoRangeX)
        self.connect(self.comboBox_2, QtCore.SIGNAL("currentIndexChanged(int)"), self.onAutoRangeY)
        self.connect(self.pushButton, QtCore.SIGNAL("clicked()"), self.onAddSelection)
        self.connect(self.pushButton_6, QtCore.SIGNAL("clicked()"), self.onClearSelection)

        self.connect(self.tableWidget, QtCore.SIGNAL("cellDoubleClicked (int, int)"), self.onSelectionTableClicked)
        self.connect(self.checkBox, QtCore.SIGNAL("stateChanged(int)"), self.parent.updatePlots)
        self.connect(self.checkBox_2, QtCore.SIGNAL("stateChanged(int)"), self.parent.updatePlots)
        self.connect(self.checkBox_3, QtCore.SIGNAL("stateChanged(int)"), self.parent.updatePlots)
        self.connect(self.checkBox_4, QtCore.SIGNAL("stateChanged(int)"), self.parent.updatePlots)

        self.connect(self.spinBox, QtCore.SIGNAL("valueChanged(QString)"), self.parent.updatePlots)
        self.connect(self.spinBox_2, QtCore.SIGNAL("valueChanged(QString)"), self.parent.updatePlots)
        self.connect(self.spinBox_3, QtCore.SIGNAL("valueChanged(QString)"), self.parent.updatePlots)
        self.connect(self.spinBox_4, QtCore.SIGNAL("valueChanged(QString)"), self.parent.updatePlots)

        self.connect(self.pushButton_4, QtCore.SIGNAL("clicked()"), self.onAutoRangeX)
        self.connect(self.pushButton_5, QtCore.SIGNAL("clicked()"), self.onAutoRangeY)

        self.selY = parent.selection_y
        self.selX = parent.selection_x

        self.update()

    def update(self):
        QtWidgets.QWidget.update(self)
        self.comboBox.blockSignals(True)
        self.comboBox_2.blockSignals(True)
        pn = self.parent.source.parameterNames
        self.comboBox.clear()
        self.comboBox_2.clear()
        self.comboBox.addItems(pn)
        self.comboBox_2.addItems(pn)

        self.comboBox.blockSignals(False)
        self.comboBox_2.blockSignals(False)
        self.parent.updatePlots

    @property
    def log_x(self):
        return bool(self.checkBox.isChecked())

    @property
    def log_y(self):
        return bool(self.checkBox_3.isChecked())

    @property
    def normed_hist_y(self):
        return bool(self.checkBox_4.isChecked())

    @property
    def normed_hist_x(self):
        return bool(self.checkBox_2.isChecked())

    @property
    def p1(self):
        idx = self.comboBox.currentIndex()
        name = self.comboBox.currentText()
        return idx, str(name)

    @property
    def binsX(self):
        return int(self.spinBox.value())

    @property
    def binsY(self):
        return int(self.spinBox_3.value())

    @property
    def bins2Y(self):
        return int(self.spinBox_4.value())

    @property
    def bins2X(self):
        return int(self.spinBox_2.value())

    @property
    def p2(self):
        idx = self.comboBox_2.currentIndex()
        name = self.comboBox_2.currentText()
        return idx, str(name)

    def onClearSelection(self):
        self.tableWidget.setRowCount(0)

    def onAutoRangeX(self):
        print("onAutoRangeX")
        xmin = self.parent.xmin
        print("xmin: %s" % xmin)
        xmax = self.parent.xmax
        print("xmax: %s" % xmax)
        self.xmin.blockSignals(True)
        self.xmax.blockSignals(True)
        self.xmin.setValue(xmin)
        self.xmax.setValue(xmax)
        self.xmin.blockSignals(False)
        self.xmax.blockSignals(False)
        self.parent.updatePlots()

    def onAutoRangeY(self):
        print("onAutoRangeY")
        ymin = self.parent.ymin
        ymax = self.parent.ymax
        print("ymin: %s" % ymin)
        print("ymax: %s" % ymax)
        self.ymin.blockSignals(True)
        self.ymax.blockSignals(True)
        self.ymin.setValue(ymin)
        self.ymax.setValue(ymax)
        self.ymin.blockSignals(False)
        self.ymax.blockSignals(False)
        self.parent.updatePlots()

    def onSelectionTableClicked(self):
        row = self.tableWidget.currentRow()
        self.tableWidget.removeRow(row)
        self.parent.updatePlots()

    def onAddSelection(self):
        p1Idx, p1Name = self.p1
        p2Idx, p2Name = self.p2
        xsel = self.selX.get_range()
        ysel = self.selY.get_range()

        table = self.tableWidget

        row = table.rowCount()
        table.setRowCount(row + 1)

        tmp = QtGui.QTableWidgetItem("%s" % p1Name)
        tmp.setFlags(QtCore.Qt.ItemIsEnabled)
        tmp.setData(1, p1Idx)
        table.setItem(row, 0, tmp)

        xmin = float(min(xsel))
        xmax = float(max(xsel))
        tmp = QtGui.QTableWidgetItem()
        tmp.setData(0, xmin)
        tmp.setFlags(QtCore.Qt.ItemIsEnabled)
        table.setItem(row, 1, tmp)

        tmp = QtGui.QTableWidgetItem()
        tmp.setFlags(QtCore.Qt.ItemIsEnabled)
        tmp.setData(0, xmax)
        table.setItem(row, 2, tmp)

        cbe = QtGui.QCheckBox(table)
        cbe.connect(cbe, QtCore.SIGNAL("stateChanged(int)"), self.parent.updatePlots)
        table.setCellWidget(row, 3, cbe)
        cbe.setChecked(True)

        cbe = QtGui.QCheckBox(table)
        cbe.connect(cbe, QtCore.SIGNAL("stateChanged(int)"), self.parent.updatePlots)
        table.setCellWidget(row, 4, cbe)
        cbe.setChecked(False)

        row += 1
        table.setRowCount(row + 1)
        ymin = float(min(ysel))
        ymax = float(max(ysel))

        tmp = QtGui.QTableWidgetItem()
        tmp.setText("%s" % p2Name)
        tmp.setData(1, p2Idx)
        tmp.setFlags(QtCore.Qt.ItemIsEnabled)
        table.setItem(row, 0, tmp)

        tmp = QtGui.QTableWidgetItem()
        tmp.setData(0, ymin)
        tmp.setFlags(QtCore.Qt.ItemIsEnabled)
        table.setItem(row, 1, tmp)

        tmp = QtGui.QTableWidgetItem()
        tmp.setData(0, ymax)
        tmp.setFlags(QtCore.Qt.ItemIsEnabled)
        table.setItem(row, 2, tmp)

        cbe = QtGui.QCheckBox(table)
        cbe.connect(cbe, QtCore.SIGNAL("stateChanged(int)"), self.parent.updatePlots)
        table.setCellWidget(row, 3, cbe)
        cbe.setChecked(True)

        cbe = QtGui.QCheckBox(table)
        cbe.connect(cbe, QtCore.SIGNAL("stateChanged(int)"), self.parent.updatePlots)
        table.setCellWidget(row, 4, cbe)
        cbe.setChecked(False)

        table.resizeRowsToContents()
        header = table.horizontalHeader()
        header.setStretchLastSection(True)

        self.parent.updatePlots()

    def getMask(self, data):
        table = self.tableWidget
        mask = np.ma.make_mask_none(data.shape)
        for r in range(table.rowCount()):
            idx = table.item(r, 0).data(1).toInt()[0]
            lower = table.item(r, 1).data(0).toFloat()[0]
            upper = table.item(r, 2).data(0).toFloat()[0]
            enabled = bool(table.cellWidget(r, 3).checkState())
            invert = bool(table.cellWidget(r, 4).checkState())
            print("idx: %s l: %s u: %s e: %s i: %s" % (idx, lower, upper, enabled, invert))
            if enabled:
                if invert:
                    mask[:, :] |= np.logical_and(data[idx, :] > lower, data[idx, :] < upper)
                else:
                    mask[:, :] |= np.logical_or(data[idx, :] < lower, data[idx, :] > upper)
        return mask

    def SetLog(self):
        print("SetLog")
        self.parent.resPlot.setLogMode(x=self.checkBox_3.isChecked(), y=self.checkBox_4.isChecked())
        self.parent.dataPlot.setLogMode(x=self.checkBox_2.isChecked(), y=self.checkBox.isChecked())


class SurfacePlot(Plot):

    name = "Surface-Plot"

    def __init__(self, fit):
        Plot.__init__(self)

        self.layout = QtGui.QVBoxLayout(self)
        self.fit = fit
        self.source = fit.surface
        self.d1 = np.array([0.0])
        self.d2 = np.array([0.0])

        top_left = QtGui.QFrame(self)
        top_left.setMaximumHeight(150)
        top_left.setFrameShape(QtGui.QFrame.StyledPanel)
        topl = QtGui.QVBoxLayout(top_left)

        top_right = QtGui.QFrame(self)
        top_right.setMaximumHeight(150)
        top_right.setFrameShape(QtGui.QFrame.StyledPanel)
        topr = QtGui.QVBoxLayout(top_right)

        bottom = QtGui.QFrame(self)
        bottom.setFrameShape(QtGui.QFrame.StyledPanel)
        bot = QtGui.QVBoxLayout(bottom)

        splitter1 = QtGui.QSplitter(QtCore.Qt.Horizontal)
        splitter1.addWidget(top_left)
        splitter1.addWidget(top_right)

        splitter2 = QtGui.QSplitter(QtCore.Qt.Vertical)
        splitter2.addWidget(splitter1)
        splitter2.addWidget(bottom)
        self.splitter = splitter2

        self.layout.addWidget(splitter2)

        # x-axis
        win = CurveDialog()
        self.g_xplot = win.get_plot()
        self.g_xhist_m = make.histogram([], color='#ff00ff')
        self.g_xhist_a = make.histogram([], color='#6f0000')
        self.g_xplot.add_item(self.g_xhist_a)
        self.g_xplot.add_item(self.g_xhist_m)
        topl.addWidget(self.g_xplot)

        # y-axis
        win = CurveDialog()
        self.g_yplot = win.get_plot()
        self.g_yhist_m = make.histogram([], color='#00ff00')
        self.g_yhist_a = make.histogram([], color='#006600')
        self.g_yplot.add_item(self.g_yhist_a)
        self.g_yplot.add_item(self.g_yhist_m)
        topr.addWidget(self.g_yplot)

        # 2D-Histogram
        self.g_hist2d_m = make.histogram2D(np.array([0.0, 0.0]), np.array([0.0, 0.0]), logscale=True)
        self.g_hist2d_a = make.histogram2D(np.array([0.0, 0.0]), np.array([0.0, 0.0]), logscale=True)
        self.g_hist2d_m.set_color_map('hot')
        self.g_hist2d_a.set_color_map('Blues')
        #self.g_hist2d_m.set_interpolation(INTERP_LINEAR)
        #self.g_hist2d_a.set_interpolation(INTERP_LINEAR)

        win = ImageDialog(edit=False, toolbar=False)
        self.g_xyplot = win.get_plot()
        self.g_xyplot.set_aspect_ratio(lock=False)
        self.g_xyplot.add_item(self.g_hist2d_a)
        self.g_xyplot.add_item(self.g_hist2d_m)
        bot.addWidget(win)

        #selection
        self.selection_x = make.range(.25, .5)
        self.g_xplot.add_item(self.selection_x)

        self.selection_y = make.range(.25, .5)
        self.g_yplot.add_item(self.selection_y)

        self.pltControl = SurfacePlotWidget(self)
        self.widgets = [self.pltControl]

    @property
    def mask(self):
        return self.pltControl.getMask(self.source.values)

    @property
    def maskedValues(self):
        x = np.ma.array(self.source.values, mask=self.mask)
        oCol, oRow = x.shape
        re = np.ma.compressed(x)
        nD = re.shape[0]
        re = re.reshape((oCol, nD/oCol))
        return re

    @property
    def xmin(self):
        try:
            return min(self.d1)
        except ValueError:
            return 0.0

    @property
    def xmax(self):
        try:
            return max(self.m1)
        except ValueError:
            return 0.0

    @property
    def ymin(self):
        try:
            return min(self.m2)
        except ValueError:
            return 0.0

    @property
    def ymax(self):
        try:
            return max(self.d2)
        except ValueError:
            return 0.0

    def updateAll(self):
        self.pltControl.update()
        self.updatePlots()

    def changeParameter(self):
        paras = self.source.values
        p1, p1Name = self.pltControl.p1
        p2, p2Name = self.pltControl.p2
        self.g_yplot.set_titles(ylabel=p2Name)
        self.g_xplot.set_titles(ylabel=p1Name)
        self.g_xyplot.set_titles(ylabel=p2Name, xlabel=p1Name)
        try:
            mv = self.maskedValues
            m1, m2 = mv[p1], mv[p2]
            d1, d2 = paras[p1], paras[p2]
        except IndexError:
            m1, m2 = [0.0], [0.0]
            d1, d2 = [0.0], [0.0]
        self.d1 = d1
        self.d2 = d2
        self.m1 = m1
        self.m2 = m2

    def updatePlots(self):
        self.changeParameter()

        m1 = self.m1
        m2 = self.m2
        d1 = self.d1
        d2 = self.d2

        # mask by range
        # x
        xmin = self.pltControl.xmin.value
        xmax = self.pltControl.xmax.value
        q1 = d1 > xmin
        q2 = d1 < xmax
        d1 = d1[q1 * q2]

        q1 = m1 > xmin
        q2 = m1 < xmax
        m1 = m1[q1 * q2]

        # y
        ymin = self.pltControl.ymin.value
        ymax = self.pltControl.ymax.value
        q1 = d2 > ymin
        q2 = d2 < ymax
        d2 = d2[q1 * q2]

        q1 = m2 > ymin
        q2 = m2 < ymax
        m2 = m2[q1 * q2]

        self.selection_x.set_range(self.xmin, self.xmax)
        self.selection_y.set_range(self.ymin, self.ymax)

        self.g_xhist_a.set_logscale(self.pltControl.log_x)
        self.g_xhist_m.set_logscale(self.pltControl.log_x)
        self.g_xhist_m.set_hist_data(m1)
        self.g_xhist_a.set_hist_data(d1)
        self.g_xplot.do_autoscale()

        self.g_yhist_a.set_logscale(self.pltControl.log_y)
        self.g_yhist_m.set_logscale(self.pltControl.log_y)
        self.g_yhist_m.set_hist_data(m2)
        self.g_yhist_a.set_hist_data(d2)
        self.g_yplot.do_autoscale()

        self.g_hist2d_m.set_bins(self.pltControl.bins2X, self.pltControl.bins2Y)
        self.pltControl.bins2Y
        try:
            self.g_hist2d_m.set_data(m1, m2)
            self.g_hist2d_a.set_data(d1, d2)
            self.g_xyplot.do_autoscale()
        except ValueError:
            print("No 2d-hist to update.")

