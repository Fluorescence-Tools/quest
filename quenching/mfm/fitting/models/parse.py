import json
from collections import defaultdict

from PyQt4 import QtGui, QtCore, uic
from numpy import *
import sympy

import mfm
from mfm.fitting.fit import ErrorWidget, FittingWidget
from mfm.fitting.models import Model
from mfm.fitting.parameter import ParameterWidget, Parameter, AggregatedParameters


try:
    from re import Scanner
except ImportError:
    import sre
    from sre import Scanner


class GenerateSymbols(defaultdict):
    def __missing__(self, key):
        return sympy.Symbol(key)


class ParseFormula(AggregatedParameters):

    @property
    def parameters(self):
        return self._parameters

    @property
    def func(self):
        return self._func

    @func.setter
    def func(self, v):
        self._func = v

    def var_found(self, scanner, name):
        if name in ['caller','e','pi']:
            return name
        if name not in self._keys:
            self._keys.append(name)
            ret = 'a[%d]' % self._count
            self._count += 1
        else:
            ret = 'a[%d]' % (self._keys.index(name))
        return ret

    def parse(self):
        scanner = Scanner([
            (r"x", lambda y,x: x),
            (r"[a-zA-Z]+\.", lambda y,x: x),
            (r"[a-z]+\(", lambda y,x: x),
            (r"[a-zA-Z_]\w*", self.var_found),
            (r"\d+\.\d*", lambda y,x: x),
            (r"\d+", lambda y,x: x),
            (r"\+|-|\*|/", lambda y,x: x),
            (r"\s+", None),
            (r"\)+", lambda y,x: x),
            (r"\(+", lambda y,x: x),
            (r",", lambda y,x: x),
            ])
        self._count = 0
        self._keys = []
        parsed, rubbish = scanner.scan(self.func)
        parsed = ''.join(parsed)
        if rubbish != '':
            raise Exception('parsed: %s, rubbish %s' % (parsed, rubbish))
        self.code = parsed

    def update_parameters(self):
        self.parse()
        self._parameters = [Parameter(name=n, value=1.0) for n in self._keys]

    def __init__(self, func="x*0"):
        self._parameters = []
        self.code = func
        self._keys = []
        self._count = 0
        self._func = func


class ParseWidget(ParseFormula, QtGui.QWidget):

    def __init__(self, parent, model_file=None, **kwargs):
        self.name = kwargs.get('name', 'Parsing')
        self._model_file = None
        self.parent = parent

        ParseFormula.__init__(self)
        QtGui.QWidget.__init__(self, **kwargs)
        uic.loadUi('mfm/ui/models/parseWidget.ui', self)

        self.model_file = model_file
        self.connect(self.radioButton_2, QtCore.SIGNAL("toggled()"), self.onPrettyEquation)
        self.connect(self.comboBox, QtCore.SIGNAL("currentIndexChanged(int)"), self.onModelChanged)
        self.connect(self.textEdit, QtCore.SIGNAL("textChanged()"), self.update_parameters)
        self.radioButton.setChecked(True)
        self.textEdit.hide()
        self.checkBox.setChecked(False)

    @property
    def func(self):
        text = str(self.textEdit.toPlainText()).strip()
        return text

    @func.setter
    def func(self, v):
        self.textEdit.setPlainText(v)

    @property
    def initial_values(self):
        try:
            ivs = self.models[self.model_name]['initial']
        except AttributeError:
            ivs = [1.0] * len(self._keys)
        return ivs

    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, v):
        self._models = v
        self.comboBox.clear()
        self.comboBox.addItems(list(v.keys()))

    @property
    def model_name(self):
        return list(self.models.keys())[self.comboBox.currentIndex()]

    @property
    def model_file(self):
        return self._model_file

    @model_file.setter
    def model_file(self, v):
        self._model_file = v
        self.onLoadModelFile(v)

    def updateWidget(self):
        pass

    def update_parameters(self):
        layout = self.layoutParameter
        for i in reversed(list(range(layout.count()))):
            layout.itemAt(i).widget().deleteLater()
        self.parse()
        self._parameters = []
        ivs = self.initial_values
        for key in self._keys:
            try:
                v = ivs[key]
            except KeyError:
                v = 1.0
            p = ParameterWidget(name=key, value=v, layout=self.layoutParameter,
                               model=self.parent, bounds_on=False, digits=6)
            self._parameters.append(p)
        self.onPrettyEquation()
        self.onDescription()

    def onDescription(self):
        t = self.models[self.model_name]['description']
        self.textEdit_3.setText(t)

    def onPrettyEquation(self):
        f = eval(self.func, GenerateSymbols())
        s = sympy.pretty(f, wrap_line=False, use_unicode=False)
        self.textEdit_2.setText(str(s))

    def onModelChanged(self):
        self.func = self.models[self.model_name]['equation']
        self.update_parameters()
        self.parent.update()

    def onLoadModelFile(self, filename=None):
        if filename is None:
            filename = str(QtGui.QFileDialog.getOpenFileName(None, 'Open model-file', '', 'link file (*.json)'))
        print("Opened file: %s" % filename)
        fp = open(self.model_file, 'r')
        self.models = json.load(fp)

    def clean(self):
        new = ParseFormula(self.code)
        new.parameters = [p.clean() for p in self.parameters]
        return new


class ParseModel(mfm.Curve, Model):

    name = "Parse-Model"

    def __init__(self, fit, **kwargs):
        mfm.Curve.__init__(self)
        Model.__init__(self, fit=fit)
        self.fit = fit
        self.parse = kwargs.get('parse', [])

    def update_model(self, **kwargs):
        a = [p.value for p in self.parse]
        x = self.x_axis
        try:
            y = eval(self.parse.code)
        except:
            y = zeros_like(x) + 1.0
        self.y_values = y

    def clean(self, fit):
        parse = ParseFormula(self.parse.code)
        parameters = [p.clean() for p in self.parameters]
        parse.parameters = parameters
        return ParseModel(fit, parse=parse)


class ParseModelWidget(QtGui.QWidget,  ParseModel):

    def __init__(self, fit, model_file):
        QtGui.QWidget.__init__(self)
        self.fit = fit
        self.parse = ParseWidget(self, model_file=model_file)
        ParseModel.__init__(self, fit=fit, parse=self.parse)
        self.parse.onModelChanged()
        self.error_widget = ErrorWidget(self.fit)
        self.fitting_widget = FittingWidget(fit=fit)

        self.layout = QtGui.QVBoxLayout(self)
        self.layout.setSpacing(0)
        self.layout.setMargin(0)
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        self.layout.addWidget(self.fitting_widget)

        self.layout.addWidget(self.parse)
        self.layout.addWidget(self.error_widget)

