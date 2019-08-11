import multiprocessing
import os

import numpy as np
from PyQt4 import QtCore, QtGui, uic
from scipy.stats import f as fdist
import emcee

from lib import Genealogy
from lib import widgets
import lib


def walk_emcee(fit, steps, thin, nwalkers, chi2max, std):
    ndim = fit.model.nFree
    sampler = emcee.EnsembleSampler(nwalkers, ndim, fit.lnprob, args=[chi2max])
    std = np.array(fit.model.parameterValues) * std
    pos = [fit.model.parameterValues + std*np.random.randn(ndim) for i in range(nwalkers)]
    sampler.run_mcmc(pos, steps, thin=thin)
    return [sampler.flatlnprobability, sampler.flatchain]


def walk_mcmc(fit, steps, step_size, chi2max, temp, thin):
    dim = fit.model.nFree
    state_initial = fit.model.parameterValues
    n_samples = steps // thin
    # initialize arrays
    lnp = np.empty(n_samples)
    parameter = np.zeros((n_samples, dim))
    n_accepted = 0
    state_prev = np.copy(state_initial)
    lnp_prev = np.array(fit.lnprob(state_initial))
    while n_accepted < n_samples:
        state_next = state_prev + np.random.normal(0.0, step_size, dim) * state_initial
        lnp_next = fit.lnprob(state_next, chi2max)
        if not np.isfinite(lnp_next):
            continue
        if (-lnp_next + lnp_prev)/temp > np.log(np.random.rand()):
            # save results
            parameter[n_accepted] = state_next
            lnp[n_accepted] = lnp_next
            # switch previous and next
            np.copyto(state_prev, state_next)
            np.copyto(lnp_prev, lnp_next)
            n_accepted += 1
    return [lnp, parameter]


def lnprior(bounds, parameters):
    for (bound, value) in zip(bounds, parameters):
        lb, ub = bound
        if lb is not None:
            if value < lb:
                return -np.inf
        if ub is not None:
            if value > ub:
                return -np.inf
    return 0.0


class Variable(Genealogy):

    def __init__(self, value=None, lb=None, ub=None, name='', model=None,
                 fixed=False, bounds_on=False, link_enabled=False):
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
        Example
        -------
        >>> import lib
        >>> a = Variable(1.0, name='a')
        >>> print a
        Variable
        --------
        name: a
        internal-value: 1.0
        value: 1.0
        >>> b = Variable(2.0, name='b')
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
            # TODO fix tooltip after linking
            #return str(self.linkVar.model.fit.name + "\n" + self.linkVar.name)
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
        return isinstance(self._linkToVar, Variable)

    def deleteLink(self):
        self._linkToVar = None
        self._link_enabled = False

    def __invert__(self):
        return float(1.0/self.value)

    def __float__(self):
        return self.value

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return self.value + other
        else:
            return self.value + other.value

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return self.value * other
        else:
            return self.value * other.value

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return self.value - other
        else:
            return self.value - other.value

    def __div__(self, other):
        if isinstance(other, (int, float)):
            return self.value / other
        else:
            return self.value / other.value

    def __str__(self):
        s = ""
        s += "Variable\n"
        s += "--------\n"
        s += "name: %s\n" % self.name
        s += "internal-value: %s\n" % self._value
        if isinstance(self.linkVar, Variable):
            s += "linked to: %s\n" % self.linkVar.name
            s += "link-enabled: %s\n" % self.linkEnabled
        s += "value: %s\n" % self.value
        return s


class Fit(Genealogy):

    def __init__(self, **kwargs):
        """
        All fits are objects of the `Fit` class

        Example
        -------
        >>> import lib
        >>> import experiments
        >>> tcspc_setup = experiments.TCSPCSetup(rep_rate=15.0, dt=0.0141, skiprows=9)
        >>> data_set = tcspc_setup.loadData('./sample_data/ibh/Decay_577D.txt', verbose=True)
        Loading data:
        Using data-file passed as argument
        Filename: ./sample_data/ibh/Decay_577D.txt
        >>> irf_data = tcspc_setup.loadData('./sample_data/ibh/Prompt.txt')
        >>> fit = lib.Fit(data=data_set, verbose=True)
        >>> print fit
        Data not fitted yet.
        >>> fit.model = experiments.models.tcspc.LifetimeModel(fit, irf=irf_data, dt=0.0141)
        >>> fit.model.lifetimes.add_lifetime(0.5, 2.0, 0, 1, False, False)
        >>> fit.xmin, fit.xmax = 0, 3000
        >>> fit.run(verbose=True)
        >>> print fit
        Fitting:
        Dataset:
        filename: ./sample_data/ibh/Decay_577D.txt
        length  : 4095
        2.820e-02       2.820e-02
        4.230e-02       4.230e-02
        5.640e-02       5.640e-02
        ....
        5.773e+01       5.773e+01
        5.774e+01       5.774e+01
        5.775e+01       5.775e+01
        Fit-result:
        ---------------------------------------------
        fitrange: 0..3000

        chi2:   1.6829
        ---------------------------------------------
        Model: Lifetime fit
        Parameter       Value   Bounds  Fixed   Linked
        bg      5.1478  (None, None)    False   False
        sc      0.0323  (None, None)    False   False
        t(L,1)  4.1454  (None, None)    False   False
        x(L,1)  1.0000  (None, None)    False   False
        r0      0.3800  (None, None)    False   False
        l1      0.0308  (None, None)    False   False
        l2      0.0368  (None, None)    False   False
        g       1.0000  (None, None)    False   False
        dt      0.0141  (None, None)    True    False
        start   0.0000  (None, None)    True    False
        stop    57.7254 (None, None)    True    False
        rep     15.0000 (None, None)    True    False
        lb      0.2253  (None, None)    False   False
        ts      -1.8259 (None, None)    False   False
        p0      1.0000  (None, None)    True    False

        Plotting of the fitting results
        >>> x_fit, y_fit = fit[1:4000]
        >>> import pylab as p
        >>> p.semilogy(fit.data.x, fit.data.y)
        >>> p.semilogy(irf_data.x, irf_data.y)
        >>> p.semilogy(x_fit, y_fit)
        >>> p.show()
        :param kwargs:
        """
        Genealogy.__init__(self)
        self.results = None
        self.isFitted = False
        self.data = kwargs.get('data', None)
        self.name = kwargs.get('name', None)
        self._model = kwargs.get('model', None)
        self.minChi2 = kwargs.get('minChi2', np.inf)
        self.xmin = kwargs.get('xmin', 0)
        self.xmax = kwargs.get('xmax', 0)
        self.verbose = kwargs.get('verbose', True)
        self.isFitted = False

    @property
    def weighted_residuals(self):
        return self.model.weighted_residuals(self.data)

    @property
    def chi2r(self):
        return self.model.chi2r(self.data)

    @property
    def name(self):
        if self.__name is not None:
            return self.__name
        try:
            return self.data.name + " : " + self.model.name
        except AttributeError:
            return "no name"

    @name.setter
    def name(self, n):
        self.__name = n

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, v):
        self._model = v
        v.link(self)
        v.link(self.data)

    def get_wres(self, parameter):
        self.model.parameterValues = parameter
        return self.model.weighted_residuals(self.data)

    def lnprior(self, parameters):
        bounds = self.model.parameter_bounds
        return lnprior(bounds, parameters)

    def lnprob(self, parameters, chi2max=np.inf):
        lp = self.lnprior(parameters)
        if not np.isfinite(lp):
            return -np.inf
        else:
            self.model.parameterValues = parameters
            self.model.calc()
            chi2 = self.chi2r
            return -0.5 * self.chi2r * self.model.numberOfPoints if chi2 < chi2max else -np.inf

    def save(self, path, name):
        np.savetxt(os.path.join(path, name+'_wr.txt'), np.array(self.weighted_residuals).T,
                   delimiter='\t')
        np.savetxt(os.path.join(path, name+'_fit.txt'), np.array(self.model[:]).T,
                   delimiter='\t')
        if self.data is not None:
            np.savetxt(os.path.join(path, name+'_data.txt'), np.array(self.data[:]).T,
                       delimiter='\t')
        fp = open(os.path.join(path, name+'_info.txt'), 'w')
        fp.write(str(self))
        fp.close()

    def __getitem__(self, key):
        self.model.calc()
        xmin, xmax = self.model.xmin, self.model.xmax
        start = int(xmin) if key.start is None else key.start
        stop = int(xmax) if key.stop is None else key.stop
        step = 1 if key.step is None else key.step
        x, y = self.model[start:stop:step]
        return x, y

    def __str__(self):
        s = "Fitting:"
        s += "\n"
        s += "Dataset:\n"
        s += str(self.data)
        if self.isFitted:
            s += "Fit-result: \n"
            s += "---------------------------------------------\n"
            s += "fitrange: %i..%i\n" % (self.xmin, self.xmax)
            s += "\nchi2:\t%.4f \n" % self.chi2r
            s += "---------------------------------------------\n"
            s += str(self.model)
        else:
            s = "Data not fitted yet."
        return s

    def run(self, xmin=None, xmax=None, verbose=False):
        self.xmin = xmin if xmin is not None else self.xmin
        self.xmax = xmax if xmax is not None else self.xmax
        verbose = verbose and self.verbose
        if verbose:
            print("fitrange: %s..%s" % (self.xmin, self.xmax))
            print("Parameter-name\tInitial-value\tBounds")
            for p in zip(self.model.parameterNames, self.model.parameterValues, self.model.parameter_bounds_all):
                print("%s \t %s \t %s" % p)
            print("Fitting using leastsqt...")
        self.results = None
        self.isFitted = True
        self.minChi2 = min(self.minChi2, self.chi2r)
        if verbose:
            print("Fitting-result")
            print("--------------")
            print(self)


class FitQtThread(Fit, QtCore.QThread):

    def __init__(self, **kwargs):
        Fit.__init__(self, **kwargs)
        QtCore.QThread.__init__(self)
        self.surface = SurfaceThread(self)
        self.__exiting = False

    def clean(self):
        new = Fit()
        new.xmin, new.xmax = self.xmin, self.xmax
        new.data = self.data.clean()
        new.model = self.model.clean(new)
        return new

    def run(self):
        Fit.run(self)
        self.__exiting = True


class Surface(object):

    methods = ['emcee', 'mcmc']

    def setSamplingParameter(self, steps, step_size, nprocs, temperature, thin,
                             nwalker, method='emcee', maxchi2=1e12):
        self.nprocs = nprocs
        self.steps = steps
        self.temperature = temperature
        self.step_size = step_size
        self.thin = thin
        self.nwalker = nwalker
        self.method = method
        self.maxchi2 = maxchi2

    def __init__(self, fit):
        self.fit = fit
        self._activeRuns = []
        self._chi2 = []
        self._parameter = []
        self.parameterNames = []

    def saveSurface(self, filename):
        fp = open(filename, 'w')
        s = ""
        for ph in self.parameterNames:
            s += ph + "\t"
        s += "\n"
        # first different runs
        print(self.values.shape)
        for l in self.values.T:
            for p in l:
                s += "%.5f\t" % p
            s += "\n"
        fp.write(s)
        fp.close()

    def clear(self):
        self._chi2 = []
        self._parameter = []

    @property
    def chi2s(self):
        return np.hstack(self._chi2)

    @property
    def values(self):
        try:
            re = np.vstack(self._parameter)
            re = np.column_stack((re, self.chi2s))
            return re.T
        except ValueError:
            return np.array([[0], [0]]).T

    def run(self):
        print("Surface:run")
        self.clear()
        fit = self.fit.clean()
        self.parameterNames = fit.model.parameterNames + ['chi2r']
        self._parameterNames = self.fit.model.parameterNames

        pool = multiprocessing.Pool()
        if self.method == 'emcee':
            results = [pool.apply_async(walk_emcee, (fit, self.steps,
                                                     self.thin, self.nwalker, self.maxchi2, self.step_size))
                       for i in range(self.nprocs)]
        elif self.method == 'mcmc':
            results = [pool.apply_async(walk_mcmc, (fit, self.steps, self.step_size, self.maxchi2,
                                                    self.temperature, self.thin))
                       for i in range(self.nprocs)]
        for r in results:
            lnProb, parameter = r.get()
            chi2 = lnProb * -2.0 / float(self.fit.model.numberOfPoints - self.fit.model.nFree - 1.0)
            mask = np.where(np.isfinite(chi2))
            self._chi2.append(chi2[mask])
            self._parameter.append(parameter[mask])


class SurfaceThread(Surface, QtCore.QThread):

    def __init__(self, fit, parent=None):
        Surface.__init__(self, fit)
        QtCore.QThread.__init__(self, parent)
        self.exiting = False
        self.connect(self, QtCore.SIGNAL("finished()"), self.onRunDone)

    def __del__(self):
        self.exiting = True
        self.wait()

    def onRunDone(self):
        print("onRunDone")
        self.fit.model.updatePlots()



class FittingWidget(QtGui.QWidget):

    def __init__(self, fit=None, **kwargs):
        QtGui.QWidget.__init__(self)
        hide_fit_button = kwargs.get('hide_fit_button', False)
        auto_range = kwargs.get('auto_range', True)
        hide_range = kwargs.get('hide_range', False)
        hide_fitting = kwargs.get('hide_fitting', False)

        uic.loadUi("lib/ui/fittingWidget.ui", self)
        self.fit = fit
        self.connect(self.checkBox_autorange, QtCore.SIGNAL("stateChanged(int)"), self.onAutoFitRange)
        self.connect(self.actionFit, QtCore.SIGNAL('triggered()'), self.onRunFit)
        self.connect(self.checkBox_autorange, QtCore.SIGNAL("stateChanged(int)"), self.onAutoFitRange)

        self.connect(self.spinBox, QtCore.SIGNAL("valueChanged (int)"), self.onFitRangeChanged)
        self.connect(self.spinBox_2, QtCore.SIGNAL("valueChanged (int)"), self.onFitRangeChanged)

        self.connect(self.fit, QtCore.SIGNAL("finished()"), self.onFitDone)
        self.connect(self.fit, QtCore.SIGNAL("stopped()"), self.onFitDone)
        self.spinBox.setDisabled(auto_range)
        self.spinBox_2.setDisabled(auto_range)
        self.checkBox_autorange.setChecked(auto_range)
        if hide_fit_button:
            self.pushButton_fit.hide()
        if hide_range:
            self.checkBox_autorange.hide()
            self.spinBox.hide()
            self.spinBox_2.hide()
        if hide_fitting:
            self.hide()

    def onFitDone(self):
        self.fit.model.updateAll()
        widgets.MyMessageBox('Fitting finished', info=str(self.fit))
        self.pushButton_fit.setEnabled(True)

    def onFitRangeChanged(self):
        xmt = self.spinBox.value()
        xMt = self.spinBox_2.value()
        xmin = float(xmt)
        xmax = float(xMt)
        self.fit.model.xmin, self.fit.model.xmax = xmin - 1, xmax - 1
        self.fit.model.updateAll()

    def onRunFit(self):
        #self.pushButton_fit.setEnabled(False)
        self.fit.run()
        self.fit.model.updateAll()

    def onAutoFitRange(self):
        autofit = self.checkBox_autorange.isChecked()
        if autofit:
            try:
                xmin, xmax = self.fit.data.setup.autofitrange(self.fit.data)
                self.fit.xmin, self.fit.xmax = xmin, xmax
            except AttributeError:
                xmin, xmax = 1, 1
            self.spinBox.setValue(int(xmin))
            self.spinBox_2.setValue(int(xmax))


class VariableWidget(QtGui.QWidget, Variable):

    def make_linkcall(self, target):
        def linkcall():
            print(self.name + " is now linked to " + target.name)
            self.linkVar = target
            self.linkEnabled = True
            self.add_child(target)
        return linkcall

    def contextMenuEvent(self, event):
        menu = QtGui.QMenu(self)
        menu.setTitle("Link " + self.name + " to:")

        fits = [f for f in lib.rootNode.get_descendants() if isinstance(f, lib.Fit)]
        for f in fits:
            submenu = QtGui.QMenu(menu)
            submenu.setTitle(f.name)
            for p in zip(f.model.parameterNames_all, f.model.parameters_all):
                if p[1] is not self:
                    Action = submenu.addAction(p[0])
                    Action.triggered.connect(self.make_linkcall(p[1]))
            menu.addMenu(submenu)
        menu.exec_(event.globalPos())


    def __str__(self):
        return ""

    def __init__(self, name, value, model=None,
                 ub=None, lb=None, layout=None, **kwargs):
        parent = kwargs.get('parent', None)
        QtGui.QWidget.__init__(self, parent)
        uic.loadUi('lib/ui/variable_widget.ui', self)
        if layout is not None:
            layout.addWidget(self)

        hide_bounds = kwargs.get('hide_bounds', False)
        hide_link = kwargs.get('hide_link', False)
        fixable = kwargs.get('fixable', True)
        hide_fix_checkbox = kwargs.get('hide_fix_checkbox', False)
        fixed = kwargs.get('fixed', False)
        digits = kwargs.get('digits', 5)
        bounds_on = kwargs.get('bounds_on', False)
        self.update_function = kwargs.get('update_function', model.updateAll)
        label_text = kwargs.get('text', name)
        if kwargs.get('hide_label', False):
            self.label.hide()


        # Display of values
        self.widget_value.setValue(float(value))
        self.widget_value.setDecimals(digits)
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
        Variable.__init__(self, lb=lb, ub=ub, value=value, name=name, model=model, fixed=fixed, bounds_on=bounds_on)

        #self.connect(self.widget_value, QtCore.SIGNAL("valueChanged (double)"), self.updateValues)
        self.connect(self.actionValueChanged, QtCore.SIGNAL('triggered()'), self.updateValues)
        self.updateWidget()

    def updateChildren(self):
        for child in self._children:
            child.updateWidget()

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

    def setValue(self, v):
        self.value = v
        self.updateWidget()

    def updateValues(self):
        self.value = float(self.widget_value.value())
        self.widget_value.setToolTip(self.link_name)
        self.update_function()

    def updateWidget(self):
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
        re = Variable(value=self.value, lb=self._lb, ub=self._ub, name=self.name,
                      fixed=self.isFixed, bounds_on=self._boundsOn)
        re._link_enabled = self.linkEnabled
        return re


class ErrorWidget(QtGui.QWidget):

    def __init__(self, fit=None, **kwargs):
        QtGui.QWidget.__init__(self)
        uic.loadUi("lib/ui/errorWidget_2.ui", self)
        self.fit = fit
        self.parent = kwargs.get('parent', None)
        if kwargs.get('hide_error_analysis', False):
            self.hide()

        self.connect(self.pushButton_runChi2, QtCore.SIGNAL("clicked()"), self.onChi2Surface_Run)
        self.connect(self.pushButton_4, QtCore.SIGNAL("clicked()"), self.onSaveChi2Surface)
        self.connect(self.pushButton_5, QtCore.SIGNAL("clicked()"), self.onClearChi2Surface)
        self.connect(self.doubleSpinBox_2, QtCore.SIGNAL("valueChanged (double)"), self.onConfLevelChanged)
        self.connect(self.doubleSpinBox, QtCore.SIGNAL("valueChanged (double)"), self.onChi2MaxChanged)
        self.connect(self.fit, QtCore.SIGNAL("finished()"), self.onChi2MinChanged) # update interface after fitting
        self.connect(self.fit.surface, QtCore.SIGNAL("finished()"), self.onFinished)
        self.connect(self.fit.surface, QtCore.SIGNAL("stopped()"), self.onFinished)
        self._chi2Max = 1e12
        self._confLevel = 0.9999
        self.comboBox.addItems(fit.surface.methods)
        self.groupBox.setChecked(False)

    def onFinished(self):
        self.pushButton_runChi2.setEnabled(True)
        widgets.MyMessageBox('Error-analysis finished')
        self.fit.model.updateAll()

    def onChi2MinChanged(self):
        self.lineEdit_4.setText("%.4f" % self.fit.minChi2)
        self._chi2Max = self.chi2Min * (1.0 + float(self.npars) / self.nu * fdist.isf(1. - self.confLevel, self.npars, self.nu))
        self.doubleSpinBox.blockSignals(True)
        self.doubleSpinBox.setValue(self._chi2Max)
        self.doubleSpinBox.blockSignals(False)
        self.lineEdit_2.setText("%d" % self.npars)
        self.lineEdit_3.setText("%d" % self.nu)
        self.nwalker = self.npars * 2

    def onChi2MaxChanged(self):
        self.chi2Max = max(self.chi2Min, self.doubleSpinBox.value())
        self.doubleSpinBox_2.setValue(self.confLevel)

    def onConfLevelChanged(self):
        confLevel = min(1.0, float(self.doubleSpinBox_2.value()))
        self.confLevel = confLevel

    def onClearChi2Surface(self):
        if self.fit is not None:
            self.fit.surface.clear()
            self.fit.model.updatePlots()

    def onSaveChi2Surface(self):
        filename = str(QtGui.QFileDialog.getSaveFileName(self, 'Save File', '.'))
        self.fit.surface.saveSurface(filename)

    @property
    def method(self):
        return str(self.comboBox.currentText())

    @property
    def thinning(self):
        try:
            return int(self.spinBox_4.value())
        except ValueError:
            return 1

    @property
    def confLevel(self):
        return self._confLevel

    @confLevel.setter
    def confLevel(self, c):
        self._confLevel = c
        self.doubleSpinBox.blockSignals(True)
        self.doubleSpinBox.setValue(self._chi2Max)
        self.doubleSpinBox.blockSignals(False)
        self._chi2Max = self.chi2Min * (1.0 + float(self.npars) / self.nu * fdist.isf(1. - self.confLevel, self.npars, self.nu))

    @property
    def chi2Min(self):
        return self.fit.minChi2

    @property
    def chi2Max(self):
        return self._chi2Max

    @chi2Max.setter
    def chi2Max(self, c):
        self._chi2Max = c
        fValue = (c / self.chi2Min - 1.0) * self.nu / self.npars
        self._confLevel = fdist.cdf(fValue, self.npars, self.nu)

    @property
    def temperature(self):
        return float(self.doubleSpinBox_4.value())

    @property
    def nu(self):
        return self.fit.model.numberOfPoints

    @property
    def npars(self):
        return self.fit.model.nFree

    @property
    def nprocs(self):
        return int(self.spinBox.value())

    @property
    def nwalker(self):
        return int(self.spinBox_2.value())

    @nwalker.setter
    def nwalker(self, v):
        self.spinBox_2.setValue(int(v))

    @property
    def step_size(self):
        return float(self.doubleSpinBox_3.value())

    @property
    def n_steps(self):
        return int(self.spinBox_3.value())

    @property
    def use_maxchi2(self):
        return bool(self.checkBox.isChecked())

    def onChi2Surface_Run(self):
        self.pushButton_runChi2.setEnabled(False)
        chi2max = self.chi2Max if self.use_maxchi2 else 1e12
        self.fit.surface.setSamplingParameter(steps=self.n_steps, nprocs=self.nprocs, step_size=self.step_size,
                                              temperature=self.temperature,
                                              thin=self.thinning, nwalker=self.nwalker,
                                              method=self.method, maxchi2=chi2max)
        self.fit.surface.run()

