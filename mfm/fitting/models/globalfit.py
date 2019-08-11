import pickle

from PyQt4 import QtCore, QtGui, uic
import numpy as np

from mfm import plots
from mfm.fitting.models import Model
from mfm.fitting.fit import ErrorWidget
import mfm
from mfm.fitting.parameter import ParameterWidget, GlobalParameter


class GlobalFitModel(Model, mfm.Curve):

    name = "Global fit"

    @property
    def local_fits(self):
        return [s for s in mfm.rootNode.get_descendants() if isinstance(s, mfm.Fit) and s is not self.fit]

    @property
    def local_fit_names(self):
        return [f.name for f in self.local_fits]

    def __init__(self, fit):
        self.fits = []
        mfm.Curve.__init__(self)
        Model.__init__(self, fit=fit)
        self._clear_on_update = True
        self.fit = fit
        self._global_parameters = dict()
        self.parameters_calculated = []
        self.weights_of_local_fits = []
        self._links = []

    def weighted_residuals(self, data, stack=True):
        re = []
        for i, f in enumerate(self.fits):
            xmin, xmax = f.model.xmin, f.model.xmax
            x, m = f.model[xmin:xmax]
            x, d, w = f.data[xmin:xmax]
            ml = min([len(m), len(d)])
            lw = self.weights_of_local_fits[i]
            re.append(lw * np.array((d[:ml] - m[:ml]) * w[:ml], dtype=np.float64))
        if stack:
            return np.hstack(re)
        else:
            return re

    @property
    def fit_names(self):
        return [f.name for f in self.fits]

    @property
    def clear_on_update(self):
        return self._clear_on_update

    @clear_on_update.setter
    def clear_on_update(self, v):
        self._clear_on_update = v

    @property
    def ys(self):
        return [f.model.y_values[f.xmin:f.xmax] for f in self.fits]

    #@property
    def xs(self):
        """
        The x-axis of all fits as list
        :return:
        """
        return [f.model.x_axis[f.xmin:f.xmax] for f in self.fits]

    @property
    def y(self):
        return np.array(self.ys).ravel()

    @property
    def x(self):
        return np.arange(len(self.y), dtype=np.float64)

    @x.setter
    def x(self, x):
        pass

    @y.setter
    def y(self, x):
        pass

    @property
    def links(self):
        return self._links

    @links.setter
    def links(self, v):
        self._links = v if type(v) is list else []

    @property
    def n_points(self):
        nbr_points = 0
        for f in self.fits:
            nbr_points += f.model.n_points
        return nbr_points

    @property
    def global_parameters_all(self):
        return list(self._global_parameters.values())

    @property
    def global_parameters_all_names(self):
        return [p.name for p in self.global_parameters_all]

    @property
    def global_parameters(self):
        return [p for p in self.global_parameters_all if not p.isFixed]

    @property
    def global_parameters_names(self):
        return [p.name for p in self.global_parameters]

    @property
    def global_parameters_bound_all(self):
        return [pi.bounds for pi in self.global_parameters_all]

    @property
    def global_parameter_linked_all(self):
        return [p.linkEnabled for p in self.global_parameters_all]

    @property
    def parameters(self):
        p = []
        for f in self.fits:
            p += f.model.parameters
        p += self.global_parameters
        return p

    @property
    def parameter_names(self):
        try:
            re = []
            for i, f in enumerate(self.fits):
                if f.model is not None:
                    re += ["%i:%s" % (i + 1, p.name) for p in f.model.parameters]
            re += self.global_parameters_names
            return re
        except AttributeError:
            return []

    @property
    def parameters_all(self):
        try:
            re = []
            for i, f in enumerate(self.fits):
                if f.model is not None:
                    re += [p for p in f.model.parameters_all]
            re += self.global_parameters_all
            return re
        except AttributeError:
            return []

    @property
    def global_parameters_values_all(self):
        return [g.value for g in self.global_parameters_all]

    @property
    def global_parameters_fixed_all(self):
        return [p.isFixed for p in self.global_parameters_all]

    @property
    def parameter_names_all(self):
        try:
            re = []
            for i, f in enumerate(self.fits):
                if f.model is not None:
                    re += ["%i:%s" % (i + 1, p.name) for p in f.model.parameters_all]
            re += self.global_parameters_all_names
            return re
        except AttributeError:
            return []

    def setLinks(self):
        self.parameters_calculated = []
        if self.clear_on_update:
            self.clear_all_links()
        f = [fit.model.parameters_all_dict for fit in self.fits]
        g = self._global_parameters
        for i, link in enumerate(self.links):
            en, origin_fit, origin_name, formula = link
            if not en:
                continue
            try:
                origin_parameter = f[origin_fit][origin_name]
                target_parameter = GlobalParameter(f, g, formula)

                origin_parameter.linkVar = target_parameter
                origin_parameter.linkEnabled = True
                target_parameter.add_child(origin_parameter)
                print("f[%s][%s] linked to %s" % (origin_fit, origin_parameter.name, target_parameter.name))
            except IndexError:
                print "not enough fits index out of range"

    def autofitrange(self, fit):
        self.xmin, self.xmax = None, None
        return self.xmin, self.xmax

    def clear_local_fits(self):
        self.fits = []

    def onRemoveLocalFit(self, nbr):
        del self.fits[nbr]

    def clear_all_links(self):
        for fit in self.fits:
            for p in fit.model.parameters_all:
                p.deleteLink()
                p.clear_children()

    def clear_listed_links(self):
        self.links = []

    def update_model(self):
        for f in self.fits:
            f.model.update_model()

    def finalize(self):
        for fit in self.fits:
            fit.model.finalize()

    def __str__(self):
        s = "\n"
        s += "Model: Global-fit\n"
        s += "Global-parameters:"
        p0 = list(zip(self.global_parameters_all_names, self.global_parameters_values_all,
                 self.global_parameters_bound_all, self.global_parameters_fixed_all,
                 self.global_parameter_linked_all))
        s += "Parameter \t Value \t Bounds \t Fixed \t Linked\n"
        for p in p0:
            s += "%s \t %.4f \t %s \t %s \t %s \n" % p
        for fit in self.fits:
            s += "\n"
            s += fit.name + "\n"
            s += str(fit.model) + "\n"
        s += "\n"
        return s


class GlobalFitModelWidget(GlobalFitModel, QtGui.QWidget):
    # TODO: Origin-equation f[0]['x'] -> f[1]['x'] instead of 0, x -> f[1]['x']

    plot_classes = [(plots.GlobalFitPlot, {'logy': 'lin',
                                           'logx': 'lin'}),
                    (plots.SurfacePlot, {})
    ]

    def __init__(self, fit):
        QtGui.QWidget.__init__(self)
        GlobalFitModel.__init__(self, fit=fit)
        uic.loadUi("mfm/ui/fitting/models/globalfit_2.ui", self)
        self.groupBox.setChecked(False)
        self.groupBox_2.setChecked(False)
        self.groupBox_3.setChecked(False)
        self.fittingWidget = mfm.FittingWidget(fit=self.fit, auto_range=True, hide_range=True)
        self.errorWidget = ErrorWidget(self.fit)
        self.layout.addWidget(self.fittingWidget)
        self.layout.addWidget(self.errorWidget)

        self.connect(self.pushButton, QtCore.SIGNAL("clicked()"), self.onAddToLocalFitList)
        self.connect(self.pushButton_2, QtCore.SIGNAL("clicked()"), self.clear_local_fits)
        self.connect(self.pushButton_3, QtCore.SIGNAL("clicked()"), self.onSaveTable)
        self.connect(self.pushButton_4, QtCore.SIGNAL("clicked()"), self.onLoadTable)
        self.connect(self.pushButton_6, QtCore.SIGNAL("clicked()"), self.onAddGlobalVariable)
        self.connect(self.pushButton_5, QtCore.SIGNAL("clicked()"), self.clear_listed_links)
        self.connect(self.pushButton_7, QtCore.SIGNAL("clicked()"), self.onClearVariables)
        self.connect(self.pushButton_8, QtCore.SIGNAL("clicked()"), self.setLinks)
        self.connect(self.addGlobalLink, QtCore.SIGNAL("clicked()"), self.onAddLink)
        self.connect(self.comboBox_gfOriginFit, QtCore.SIGNAL("currentIndexChanged(int)"), self.update_parameter_origin)
        self.connect(self.comboBox_gfTargetFit, QtCore.SIGNAL("currentIndexChanged(int)"), self.update_parameter_target)
        self.connect(self.comboBox_gfTargetParameter, QtCore.SIGNAL("currentIndexChanged(int)"), self.update_link_text)
        self.connect(self.comboBox_gfOriginParameter, QtCore.SIGNAL("currentIndexChanged(int)"), self.update_link_text)
        self.connect(self.table_GlobalLinks, QtCore.SIGNAL("cellDoubleClicked (int, int)"), self.table_GlobalLinksDoubleClicked)
        self.connect(self.tableWidget, QtCore.SIGNAL("cellDoubleClicked (int, int)"), self.onRemoveLocalFit)
        self.connect(self.checkBox_2, QtCore.SIGNAL("stateChanged (int)"), self.update_parameter_origin)
        self.connect(self.actionUpdate_widgets, QtCore.SIGNAL('triggered()'), self.update_widgets)

    def update_link_text(self):
        self.lineEdit_2.setText(self.current_link_formula)
        self.lineEdit_3.setText(self.current_origin_link_formula)

    @property
    def current_origin_formula(self):
        return str(self.lineEdit_3.text())

    @property
    def add_all_fits(self):
        return bool(self.checkBox.isChecked())

    @property
    def current_global_variable_name(self):
        return str(self.lineEdit.text())

    @property
    def current_fit_index(self):
        return self.comboBox.currentIndex()

    @property
    def link_all_of_type(self):
        return not self.checkBox_2.isChecked()

    @property
    def clear_on_update(self):
        return self.checkBox_3.isChecked()

    @clear_on_update.setter
    def clear_on_update(self, v):
        self.checkBox_3.setChecked(v)

    def onRemoveLocalFit(self):
        row = self.tableWidget.currentRow()
        self.tableWidget.removeRow(row)
        GlobalFitModel.onRemoveLocalFit(self, row)

    def clear_local_fits(self):
        GlobalFitModel.clear_local_fits(self)
        self.tableWidget.setRowCount(0)

    def table_GlobalLinksDoubleClicked(self):
        row = self.table_GlobalLinks.currentRow()
        self.table_GlobalLinks.removeRow(row)

    def onAddGlobalVariable(self):
        print("onAddVariable")
        if len(self.current_global_variable_name) > 0 and \
                        self.current_global_variable_name not in list(self._global_parameters.keys()):
            l = self.verticalLayout
            w = ParameterWidget(self.current_global_variable_name, 1.0, self, digits=3, layout=l)
            self._global_parameters[self.current_global_variable_name] = w
        else:
            print("No variable name defined.")

    def onClearVariables(self):
        print("onClearVariables")
        self._global_parameters = dict()
        layout = self.verticalLayout
        for i in reversed(list(range(layout.count()))):
            layout.itemAt(i).widget().deleteLater()

    def onAddToLocalFitList(self):
        localFits = self.local_fits
        fit_indeces = range(len(localFits)) if self.add_all_fits else [self.current_fit_index]
        for fitIndex in fit_indeces:
            fit = localFits[fitIndex]
            if not fit in self.fits:
                self.fits.append(fit)
                self.weights_of_local_fits.append(self.current_weight)

                table = self.tableWidget
                table.insertRow(table.rowCount())
                rc = table.rowCount() - 1

                tmp = QtGui.QTableWidgetItem(fit.name)
                tmp.setFlags(QtCore.Qt.ItemIsEnabled)
                table.setItem(rc, 1, tmp)

                tmp = QtGui.QTableWidgetItem()
                tmp.setFlags(QtCore.Qt.ItemIsEnabled)
                tmp.setData(0, self.current_weight)
                table.setItem(rc, 0, tmp)

                header = table.horizontalHeader()
                header.setStretchLastSection(True)
                table.resizeRowsToContents()

                self.update_widgets(fit_combo=False)

    @property
    def current_weight(self):
        return float(self.doubleSpinBox.value())

    @property
    def current_target_formula(self):
        return str(self.lineEdit_2.text())

    @property
    def origin_fit_number(self):
        return int(self.comboBox_gfOriginFit.currentIndex())  # origin fit nbr

    @property
    def origin_fit(self):
        ofNbr = self.origin_fit_number
        return self.fits[ofNbr]

    @property
    def origin_parameter(self):
        return self.origin_fit.model.parameters_all_dict[self.origin_parameter_name]

    @property
    def origin_parameter_name(self):
        return str(self.comboBox_gfOriginParameter.currentText())

    @property
    def target_fit_number(self):
        return int(self.comboBox_gfTargetFit.currentIndex())  # target fit nbr

    @property
    def target_fit(self):
        tfNbr = self.target_fit_number
        return self.fits[tfNbr]

    @property
    def target_parameter_name(self):
        return str(self.comboBox_gfTargetParameter.currentText())

    @property
    def target_parameter(self):
        return self.target_fit.model.parameters_all_dict[self.target_parameter_name]

    @property
    def current_link_formula(self):
        return "f[%s]['%s']" % (self.target_fit_number, self.target_parameter_name)

    @property
    def current_origin_link_formula(self):
        if self.link_all_of_type:
            return "f[i]['%s']" % (self.origin_parameter_name)
        else:
            return "f[%s]['%s']" % (self.origin_fit_number, self.origin_parameter_name)

    @property
    def links(self):
        table = self.table_GlobalLinks
        links = []
        for r in range(table.rowCount()):
            # self.tableWidget_2.item(r, 2).data(0).toInt()
            en = bool(table.cellWidget(r, 0).checkState())
            fitA = int(table.item(r, 1).data(0)) - 1
            pA = str(table.item(r, 2).text())
            fB = str(table.item(r, 3).text())
            links.append([en, fitA, pA, fB])
        print(links)
        return links

    def onAddLink(self, links=None):
        table = self.table_GlobalLinks
        if links is None:
            links = []
            if self.link_all_of_type:
                print("Link all of one kind: %s" % self.link_all_of_type)
                for fit_nbr, fit in enumerate(self.fits):
                    fit = self.fits[fit_nbr]
                    if self.origin_parameter_name not in fit.model.parameter_names_all:
                        continue
                    origin_parameter = self.fits[fit_nbr].model.parameters_all_dict[self.origin_parameter_name]
                    if origin_parameter is self.target_parameter:
                        continue
                    links.append(
                        [True, fit_nbr, origin_parameter.name,
                         self.current_target_formula]
                    )
            else:
                links.append(
                    [True, self.origin_fit_number, self.origin_parameter_name,
                     self.current_target_formula]
                )
        print("Added links:")
        print(links)
        for link in links:
            en, origin_fit, origin_parameter, formula = link

            rc = table.rowCount()
            table.insertRow(table.rowCount())

            cbe = QtGui.QCheckBox(table)
            cbe.setChecked(en)
            table.setCellWidget(rc, 0, cbe)
            table.resizeRowsToContents()
            cbe.setChecked(True)

            tmp = QtGui.QTableWidgetItem()
            tmp.setData(0, int(origin_fit + 1))
            tmp.setFlags(QtCore.Qt.ItemIsEnabled)
            table.setItem(rc, 1, tmp)

            tmp = QtGui.QTableWidgetItem(origin_parameter)
            tmp.setFlags(QtCore.Qt.ItemIsEnabled)
            table.setItem(rc, 2, tmp)

            tmp = QtGui.QTableWidgetItem(formula)
            tmp.setFlags(QtCore.Qt.ItemIsEnabled)
            table.setItem(rc, 3, tmp)

    def update_parameter_origin(self):
        self.comboBox_gfOriginParameter.clear()
        if len(self.fits) > 0:
            if not self.link_all_of_type:
                self.comboBox_gfOriginParameter.addItems([p for p in self.origin_fit.model.parameter_names_all])
            else:
                names = set([p for f in self.fits for p in f.model.parameter_names_all])
                names = list(names)
                names.sort()
                self.comboBox_gfOriginParameter.addItems(names)

    def update_parameter_target(self):
        self.comboBox_gfTargetParameter.clear()
        if len(self.fits) > 0:
            ftIndex = self.comboBox_gfTargetFit.currentIndex()
            ft = self.fits[ftIndex]
            self.comboBox_gfTargetParameter.addItems([p for p in ft.model.parameter_names_all])

    def update_widgets(self, fit_combo=True):
        Model.update_widgets(self)
        self.comboBox.clear()
        self.comboBox.addItems(self.local_fit_names)

        self.comboBox_gfOriginFit.clear()
        self.comboBox_gfTargetFit.clear()
        usedLocalFitNames = [str(i + 1) for i, f in enumerate(self.fits)]
        self.comboBox_gfOriginFit.addItems(usedLocalFitNames)
        self.comboBox_gfTargetFit.addItems(usedLocalFitNames)

    def update(self):
        self.update_model()
        self.update_widgets()
        self.update_plots()
        for f in self.fits:
            f.model.update_plots()

    def onSaveTable(self):
        filename = str(QtGui.QFileDialog.getSaveFileName(self, 'Save link-table', '.p'))
        pickle.dump(self.links, open(filename, "wb"))

    def onLoadTable(self):
        filename = str(QtGui.QFileDialog.getOpenFileName(None, 'Open link-table', '', 'link file (*.p)'))
        links = pickle.load(open(filename, "rb"))
        self.onAddLink(links)

    def clear_listed_links(self):
        self.table_GlobalLinks.setRowCount(0)

    @property
    def local_fit_first(self):
        return self.checkBoxLocal.isChecked()

    @local_fit_first.setter
    def local_fit_first(self, v):
        if v is True:
            self.checkBoxLocal.setCheckState(2)
        else:
            self.checkBoxLocal.setCheckState(0)

    def clean(self, fit):
        new = GlobalFitModel(fit=fit)
        new.links = self.links
        new.fits = [f.clean() for f in self.fits]
        self.weights_of_local_fits = self.weights_of_local_fits
        new.setLinks()
        return new


