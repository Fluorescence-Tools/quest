from PyQt4 import QtGui, uic, QtCore
from mfm.curve import Genealogy
import mfm
import itertools


class Parameter(Genealogy):
    """
    All variables used in fitting have to be of the type `Variable`. Variables have
    a value, a lower-, a upper-bound, are either fixed or not. Furthermore, the bounds
    may be switched on using the attribute `bounds_on`.

    :param value: float
    :param lb: float or None
    :param ub: float or None
    :param name: string
    :param model: Model
    :param fixed: bool
    :param bounds_on: bool
    :param link_enabled: bool

    Variables may be linked to other variables. If a variable a is linked to another variable b
    the value of the linked variable a is the value of the variable b.

    Examples
    --------

    >>> a = Parameter(1.0, name='a')
    >>> print a
    Variable
    --------
    name: a
    internal-value: 1.0
    value: 1.0
    >>> b = Parameter(2.0, name='b')
    Variable
    --------
    name: b
    internal-value: 2.0
    value: 2.0
    >>> a.linkVar = b
    >>> print a
    Variable
    --------
    name: b
    internal-value: 2.0
    value: 2.0
    >>> a.linkEnabled = True
    >>> print a
    Variable
    --------
    name: b
    internal-value: 2.0
    value: 2.0
    >>> a.value
    2.0
    >>> a.linkEnabled = False
    >>> a.value
    1.0

    Variables may be used to perform simple calculations. Here, however the result is a float and not an
    object of the type `Variable`

    >>> type(a + b)
    float
    >>> a + b
    3.0
    """

    def __init__(self, value=None, lb=None, ub=None, name='', model=None,
                 fixed=False, bounds_on=False, link_enabled=False):
        Genealogy.__init__(self)
        self._children = []
        self._value = value
        self._lb, self._ub = lb, ub
        self._fixed = fixed
        self._linkToVar = None
        self._boundsOn = bounds_on
        self._model = None
        self._link_enabled = link_enabled
        self.name = name

    @property
    def link_name(self):
        try:
            return str(self.linkVar.model.fit.name + "\n" + self.linkVar.name)
        except AttributeError:
            return "unlinked"

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, v):
        if isinstance(v, Genealogy):
            self._model = v
            self.link(v)

    @property
    def bounds(self):
        if self.bounds_on:
            return self._lb, self._ub
        else:
            return None, None

    @bounds.setter
    def bounds(self, b):
        self._lb, self._ub = b

    @property
    def bounds_on(self):
        return self._boundsOn

    @bounds_on.setter
    def bounds_on(self, v):
        self._boundsOn = bool(v)

    @property
    def value(self):
        if self.isLinked and self.linkEnabled:
            return self._linkToVar.value
        else:
            if self._value is not None:
                return float(self._value)
            else:
                return 1.0

    @value.setter
    def value(self, value):
        self._value = float(value)
        if self.isLinked and self.linkEnabled:
            self._linkToVar.value = value

    @property
    def isFixed(self):
        return self._fixed

    def add_child(self, child):
        self._children.append(child)

    def clear_children(self):
        self._children = []

    @property
    def linkVar(self):
        if self.isLinked:
            return self._linkToVar
        else:
            return None

    @linkVar.setter
    def linkVar(self, link):
        self._linkToVar = link

    @property
    def linkEnabled(self):
        return self._link_enabled

    @linkEnabled.setter
    def linkEnabled(self, v):
        self._link_enabled = bool(v)

    @property
    def isLinked(self):
        return isinstance(self._linkToVar, Parameter)

    def deleteLink(self):
        self._linkToVar = None
        self._link_enabled = False

    def set_value(self, v):
        """
        Convenience function equivalent to set the attribute value directly
        """
        self.value = v

    def __invert__(self):
        return float(1.0/self.value)

    def __float__(self):
        return self.value

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Parameter(self.value + other)
        else:
            return Parameter(self.value + other.value)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Parameter(self.value * other)
        else:
            return Parameter(self.value * other.value)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return Parameter(self.value - other)
        else:
            return Parameter(self.value - other.value)

    def __div__(self, other):
        if isinstance(other, (int, float)):
            return Parameter(self.value / other)
        else:
            return Parameter(self.value / other.value)

    def __str__(self):
        s = ""
        s += "Variable\n"
        s += "--------\n"
        s += "name: %s\n" % self.name
        s += "internal-value: %s\n" % self._value
        if isinstance(self.linkVar, Parameter):
            s += "linked to: %s\n" % self.linkVar.name
            s += "link-enabled: %s\n" % self.linkEnabled
        s += "value: %s\n" % self.value
        return s


class ParameterWidget(QtGui.QWidget, Parameter):

    def make_linkcall(self, target):

        def linkcall():
            tooltip = " linked to " + target.name
            print(self.name + tooltip)
            self.linkVar = target
            self.linkEnabled = True
            self.add_child(target)
            self.widget_link.setToolTip(tooltip)
        return linkcall

    def contextMenuEvent(self, event, old_style=False):
        # TODO: change somehow to new-style with sub-sub menus
        if old_style:
            menu = QtGui.QMenu(self)
            menu.setTitle("Link " + self.name + " to:")

            fits = [f for f in mfm.rootNode.get_descendants() if isinstance(f, mfm.Fit)]
            for f in fits:
                submenu = QtGui.QMenu(menu)
                submenu.setTitle(f.name)
                if isinstance(f.model, mfm.fitting.models.globalfit.GlobalFitModel):
                    parameters = f.model.global_parameters_all
                    parameter_names = f.model.global_parameters_all_names
                else:
                    parameters = f.model.parameters_all
                    parameter_names = f.model.parameter_names_all
                for p in zip(parameter_names, parameters):
                    if p[1] is not self:
                        Action = submenu.addAction(p[0])
                        Action.triggered.connect(self.make_linkcall(p[1]))
                menu.addMenu(submenu)
            menu.exec_(event.globalPos())
        else:
            menu = QtGui.QMenu(self)
            menu.setTitle("Link " + self.name + " to:")

            fits = [f for f in mfm.rootNode.get_descendants() if isinstance(f, mfm.Fit)]
            for f in fits:
                submenu = QtGui.QMenu(menu)
                submenu.setTitle(f.name)
                aggregated_parameter = [p for p in f.model.__dict__.values() if isinstance(p, AggregatedParameters)]
                for a in aggregated_parameter:
                    action_submenu = QtGui.QMenu(submenu)
                    action_submenu.setTitle(a.name)
                    for p in zip(a.parameter_names, a.parameters):
                        if p[1] is not self:
                            Action = action_submenu.addAction(p[0])
                            Action.triggered.connect(self.make_linkcall(p[1]))
                    submenu.addMenu(action_submenu)
                menu.addMenu(submenu)
            menu.exec_(event.globalPos())

    def __str__(self):
        return ""

    def __init__(self, name, value, model=None,
                 ub=None, lb=None, layout=None, **kwargs):
        parent = kwargs.get('parent', None)
        tooltip = kwargs.get('tooltip', '')

        QtGui.QWidget.__init__(self, parent)
        uic.loadUi('mfm/ui/variable_widget.ui', self)
        if layout is not None:
            layout.addWidget(self)
        # TODO: replace normal DoubleSpinBox
        # use validator
        # http://snorf.net/blog/2014/08/09/using-qvalidator-in-pyqt4-to-validate-user-input/

        hide_bounds = kwargs.get('hide_bounds', False)
        hide_link = kwargs.get('hide_link', False)
        fixable = kwargs.get('fixable', True)
        hide_fix_checkbox = kwargs.get('hide_fix_checkbox', False)
        fixed = kwargs.get('fixed', False)
        digits = kwargs.get('digits', 5)
        bounds_on = kwargs.get('bounds_on', False)
        if model is not None:
            self.update_function = kwargs.get('update_function', model.update)
        else:
            self.update_function = None
        label_text = kwargs.get('text', name)
        if kwargs.get('hide_label', False):
            self.label.hide()

        # Display of values
        self.widget_value.setValue(float(value))
        self.widget_value.setDecimals(digits)
        self.label.setToolTip(tooltip)

        if not fixable or hide_fix_checkbox:
            self.widget_fix.hide()

        # variable bounds
        if not bounds_on:
            self.widget_bounds_on.setCheckState(QtCore.Qt.Unchecked)
        else:
            self.widget_bounds_on.setCheckState(QtCore.Qt.Checked)
        self.widget_bounds_on.setDisabled(hide_bounds)
        self.widget.setHidden(hide_bounds)

        # variable link
        self.widget_link.setDisabled(hide_link)
        self.label.setText(label_text.ljust(5))
        Parameter.__init__(self, lb=lb, ub=ub, value=value, name=name, model=model, fixed=fixed, bounds_on=bounds_on)

        #self.connect(self.widget_value, QtCore.SIGNAL("valueChanged (double)"), self.updateValues)
        self.connect(self.actionValueChanged, QtCore.SIGNAL('triggered()'), self.updateValues)
        self.update_widget()

    def updateChildren(self):
        for child in self._children:
            child.update_widget()

    @property
    def _fixed(self):
        return bool(self.widget_fix.isChecked())

    @_fixed.setter
    def _fixed(self, v):
        if v is True:
            self.widget_fix.setCheckState(2)
        else:
            self.widget_fix.setCheckState(0)

    @property
    def bounds_on(self):
        return self.widget_bounds_on.isChecked()

    @bounds_on.setter
    def bounds_on(self, v):
        v = bool(v)
        self.widget_bounds_on.setChecked(v)

    @property
    def _ub(self):
        return float(self.widget_upper_bound.value())

    @_ub.setter
    def _ub(self, v):
        value = float(self.widget_value.value())
        mv = v if v is not None else value + 0.5 * abs(value)
        self.widget_upper_bound.setValue(mv)
        self.widget_upper_bound.setSingleStep(abs(self.widget_value.value())/50.)

    @property
    def _lb(self):
        return float(self.widget_lower_bound.value())

    @_lb.setter
    def _lb(self, v):
        value = float(self.widget_value.value())
        mv = v if v is not None else value - 0.5 * abs(value)
        self.widget_lower_bound.setValue(mv)
        self.widget_lower_bound.setSingleStep(abs(self.widget_value.value())/50.)

    def set_value(self, v):
        self.setValue(v)

    def setValue(self, v):
        self.value = v
        self.update_widget()

    def updateValues(self):
        lower_bound, upper_bound = self.bounds
        value = self.widget_value.value()
        if isinstance(lower_bound, float):
            value = max(lower_bound, value)
        if isinstance(upper_bound, float):
            value = min(upper_bound, value)

        self.widget_value.setValue(value)
        self.value = float(value)
        self.widget_value.setToolTip(self.link_name)
        self.update_function()

    def update_widget(self):
        self.widget_value.blockSignals(True)
        self.widget_value.setValue(float(self.value))
        self.widget_value.setSingleStep(abs(self.value)/50.)
        self.widget_value.blockSignals(False)

    @property
    def isFixed(self):
        return self.widget_fix.isChecked()

    @property
    def _link_enabled(self):
        return self.widget_link.isChecked()

    @_link_enabled.setter
    def _link_enabled(self, v):
        self.widget_link.setChecked(v)

    def clean(self):
        re = Parameter(value=self.value, lb=self._lb, ub=self._ub, name=self.name,
                      fixed=self.isFixed, bounds_on=self._boundsOn)
        re._link_enabled = self.linkEnabled
        return re


class AggregatedParameters(object):

    @property
    def name(self):
        try:
            return self._name
        except AttributeError:
            return self.__class__.__name__

    @name.setter
    def name(self, v):
        self._name = v

    @property
    def parameters(self):
        return mfm.find_object_type(self.__dict__.values(), Parameter)

    @property
    def parameter_names(self):
        return [p.name for p in self.parameters]

    @property
    def n_parameters(self):
        return len(self.parameters)

    def __repr__(self):
        return str(self)

    def __str__(self):
        s = "Aggregated-parameter: %s\n\n" % self.name
        for p in self.parameters:
            n, v = p.name, p.value
            s += "\t%s:\t%.4f\n" % (n, v)
        return s

    def __getitem__(self, key):
        nPara = self.n_parameters
        if isinstance(key, int):
            return self.parameters[key]
        else:
            start = None if key.start is None else key.start % nPara
            stop = None if key.stop is None else key.stop % nPara
            step = None if key.step is None else key.step % nPara
            return self.parameters[start:stop:step]

    def __init__(self, **kwargs):
        self.verbose = kwargs.get('verbose', mfm.verbose)
        self._name = None


class GlobalParameter(Parameter):

    @property
    def value(self):
        g = self.g
        f = self.f
        r = eval(self.formula)
        return r.value

    @value.setter
    def value(self, v):
        pass

    @property
    def name(self):
        return self.formula

    @name.setter
    def name(self, v):
        pass

    def __init__(self, f, g, formula):
        Parameter.__init__(self, fixed=False)
        self.f, self.g = f, g
        self.formula = formula