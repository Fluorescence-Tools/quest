import json

from PyQt4 import QtGui, uic, QtCore
import numpy as np
from mfm.fitting.models import Model

from mfm import plots
from mfm.fitting.parameter import ParameterWidget
from mfm.math.reaction.continuous import ReactionSystem
from parse import ParseModelWidget
import mfm


class ParseStoppedFlowWidget(ParseModelWidget):

    plot_classes = [(plots.LinePlot, {'d_scalex': 'lin',
                                                   'd_scaley': 'lin',
                                                   'r_scalex': 'lin',
                                                   'r_scaley': 'lin',
                                                   }),
                    (plots.SurfacePlot, {})
    ]

    def __init__(self, fit):
        ParseModelWidget.__init__(self, fit, model_file='./settings/stopped_flow.model.json')


class ReactionWidget(QtGui.QWidget, ReactionSystem, Model):

    name = "Reaction-System"


    plot_classes = [(plots.LinePlot, {'d_scalex': 'lin',
                                                   'd_scaley': 'lin',
                                                   'r_scalex': 'lin',
                                                   'r_scaley': 'lin',
                                                   })
    ]

    @property
    def autoscale(self):
        return bool(self.checkBox.isChecked())

    @property
    def y_values(self):
        try:
            y = self.signal_intensity * self.scaleing.value
            if self.autoscale:
                s = self.fit.data.y[self.xmin:self.xmax].sum()
                ys = sum(y[self.xmin:self.xmax])
                y *= s / ys
            y += self.background.value
            y = np.array(y, dtype=np.float64)
            return y
        except (ValueError, IndexError):
            print "Problem with y-values"
            return np.ones(10)

    @y_values.setter
    def y_values(self, v):
        pass

    @property
    def times(self):
        return self.x_axis[self.xmin:self.xmax]

    @property
    def new_brightness_fixed(self):
        return bool(self.checkBox_3.isChecked())

    @property
    def new_concentration_fixed(self):
        return bool(self.checkBox_4.isChecked())

    @property
    def xmax(self):
        return self.fitting_widget.xmax

    @xmax.setter
    def xmax(self, v):
        self.fitting_widget.xmax = int(v)

    @property
    def xmin(self):
        return self.fitting_widget.xmin

    @xmin.setter
    def xmin(self, v):
        self.fitting_widget.xmin = int(v)

    def clear(self):
        ReactionSystem.clear(self)
        mfm.widgets.clearLayout(self.verticalLayout_10)
        mfm.widgets.clearLayout(self.verticalLayout_7)

    def __init__(self, **kwargs):
        self.scaleing = ParameterWidget('scaling', 1.0, self)
        self.background = ParameterWidget('background', 0.0, self)
        self.timeshift = ParameterWidget('timeshift', 0.0, self)

        ReactionSystem.__init__(self, **kwargs)
        parameter = kwargs.get('parameter', None)
        QtGui.QWidget.__init__(self)
        uic.loadUi('mfm/ui/models/reaction.ui', self)
        self.connect(self.actionPlot, QtCore.SIGNAL('triggered()'), self.onPlot)
        self.connect(self.actionIntegrate, QtCore.SIGNAL('triggered()'), self.calc)
        self.connect(self.actionLoad_reaction, QtCore.SIGNAL('triggered()'), self.onLoadReaction)
        self.connect(self.actionUpdate_reaction, QtCore.SIGNAL('triggered()'), self.onUpdateReaction)
        self.connect(self.actionSave_reaction, QtCore.SIGNAL('triggered()'), self.onSaveLabelingFile)
        Model.__init__(self, **kwargs)
        self.setParameter(parameter)
        self.fitting_widget = mfm.fitting.fit.FittingWidget(fit=self.fit)
        self.verticalLayout_4.addWidget(self.fitting_widget)
        self.verticalLayout_4.addWidget(self.scaleing)
        self.verticalLayout_4.addWidget(self.background)
        self.verticalLayout_4.addWidget(self.timeshift)

    def setParameter(self, parameter):
        self.clear()
        if isinstance(parameter, dict):
            reactions = parameter['reactions']
            species = parameter['species']
            for reaction in reactions:
                self.add_reaction(**reaction)
            for s in species:
                print s
                self.onAddSpecies(**s)

    def onPlot(self):
        self.calc()
        self.plot()

    def onLoadReaction(self):
        self.clear()
        filename = str(QtGui.QFileDialog.getOpenFileName(self, 'Open Reaction-File', '.rc.json', 'Reaction-Files (*.rc.json)'))
        j = json.load(open(filename))
        self.setParameter(j)
        self.lineEdit_6.setText(filename)
        self.plainTextEdit.setPlainText(open(filename).read())

    def onSaveLabelingFile(self):
        txt = str(self.plainTextEdit.toPlainText())
        json_file = str(QtGui.QFileDialog.getSaveFileName(self, 'Save Reaction-JSON File',
                                                                  '.rc.json', 'JSON-Files (*.rc.json)'))
        open(json_file, 'w').write(txt)

    def onUpdateReaction(self):
        self.clear()
        txt = str(self.plainTextEdit.toPlainText())
        j = json.loads(txt)
        self.setParameter(j)

    def onAddSpecies(self, **kwargs):
        brightness = kwargs.get('brightness', 1.0)
        brightness_fixed = kwargs.get('brightness_fixed', True)
        concentration = kwargs.get('concentration', 1.0)
        concentration_fixed = kwargs.get('concentration_fixed', True)
        species_name = kwargs.get('species', '-')

        species = self.n_species + 1
        b = ParameterWidget(name="Q(%s)" % species_name, value=brightness, lb=0.0, ub=1000 , model=self,
                           fixed=brightness_fixed, bounds_on=True, hide_bounds=True)
        c = ParameterWidget(name="c(%s)" % species_name, value=concentration, lb=0.0, ub=1000 , model=self,
                           fixed=concentration_fixed, bounds_on=True, hide_bounds=True)
        self._initial_concentrations.append(c)
        self._species_brightness.append(b)
        l = QtGui.QHBoxLayout()
        l.addWidget(c)
        l.addWidget(b)
        self.verticalLayout_10.addLayout(l)

    def add_reaction(self, **kwargs):
        educts = kwargs.get('educts', 0)
        products = kwargs.get('products', 0)
        educt_stoichiometry = np.array(kwargs.get('educt_stoichiometry', 1), dtype=np.float64)
        product_stoichometry = np.array(kwargs.get('product_stoichometry', 1), dtype=np.float64)
        rate = kwargs.get('rate', 0.1)

        self.educts.append(educts)
        self.products.append(products)
        self.educts_stoichometry.append(educt_stoichiometry)
        self.products_stoichometry.append(product_stoichometry)
        v = ParameterWidget(name="k(%i)" % self.n_reactions, value=rate, model=self, hide_bounds=True,
                           bounds_on=True, **kwargs)
        self.rates.append(v)
        self.verticalLayout_7.addWidget(v)




