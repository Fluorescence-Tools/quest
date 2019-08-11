import multiprocessing
import os

import numpy as np
from PyQt4 import QtCore, QtGui, uic
from scipy.stats import f as fdist
import emcee
import scipy.optimize
from mfm.curve import Genealogy

from mfm.fitting.optimization.leastsqbound import leastsqbound
from mfm import Surface
from mfm import widgets
import mfm


def chi2r(self, data=None):
    data = self.fit.data if data is None else data
    try:
        wr = self.weighted_residuals(data)
        chi2 = np.sum(wr ** 2) / float(self.n_points - self.n_free - 1.0)
        return np.inf if np.isnan(chi2) else chi2
    except:
        return np.inf


def get_chi2(parameter, model, data, reduced=True):
    try:
        wres = get_wres(parameter, model, data)
        divisor = float(model.n_points - model.n_free - 1.0) if reduced is True else 1.0
        chi2 = np.sum(wres**2) / divisor
        return np.inf if np.isnan(chi2) else chi2
    except:
        return np.inf


def get_wres(parameter, model, data):
    model.parameter_values = parameter
    return model.weighted_residuals(data=data)


def walk_emcee(fit, steps, thin, nwalkers, chi2max, std):
    ndim = fit.model.n_free
    sampler = emcee.EnsembleSampler(nwalkers, ndim, fit.lnprob, args=[chi2max])
    std = np.array(fit.model.parameter_values) * std
    pos = [fit.model.parameter_values + std*np.random.randn(ndim) for i in range(nwalkers)]
    sampler.run_mcmc(pos, steps, thin=thin)
    return [sampler.flatlnprobability, sampler.flatchain]


def walk_mcmc(fit, steps, step_size, chi2max, temp, thin):
    dim = fit.model.n_free
    state_initial = fit.model.parameter_values
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


class Fit(Genealogy):
    """
    All fits are objects of the `Fit` class

    Examples
    --------

    >>> from mfm import experiments    >>> import mfm
    >>> tcspc_setup = experiments.TCSPCSetup(rep_rate=15.0, dt=0.0141, skiprows=9)
    >>> data_set = tcspc_setup.load_data('./sample_data/ibh/Decay_577D.txt', verbose=True)
    Loading data:
    Using data-file passed as argument
    Filename: ./sample_data/ibh/Decay_577D.txt
    >>> irf_data = tcspc_setup.load_data('./sample_data/ibh/Prompt.txt')
    >>> fit = mfm.Fit(data=data_set, verbose=True)
    >>> print fit
    Data not fitted yet.
    >>> fit.model = mfm.models.tcspc.LifetimeModel(fit, irf=irf_data, dt=0.0141)
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

    fitrange: 0..3000

    chi2:   1.6829

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

    Parameters
    ----------

    data:
    name:
    model:
    minChi2:
    xmin:
    xmax:
    verbose:
    kwargs:

    """

    def __init__(self, data=None, model=None, **kwargs):
        Genealogy.__init__(self)
        self.__name = "No name set."
        self.results = None
        self.is_fitted = False
        self.data = data
        self._model = model
        self.name = kwargs.get('name', None)
        self.min_chi2 = kwargs.get('min_chi2', np.inf)
        self.xmin = kwargs.get('xmin', 0)
        self.xmax = kwargs.get('xmax', 0)
        self.verbose = kwargs.get('verbose', mfm.verbose)
        self.fitting_parameters = dict()

    @property
    def weighted_residuals(self):
        return self.model.weighted_residuals(data=self.data)

    @property
    def chi2r(self):
        return self.model.chi2r(self.data)

    @property
    def name(self):
        if self.__name is not None:
            return self.__name
        try:
            return self.model.name + " - " + self.data.name
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

    def lnprior(self, parameters):
        bounds = self.model.parameter_bounds
        return lnprior(bounds, parameters)

    def lnprob(self, parameters, chi2max=np.inf):
        lp = self.lnprior(parameters)
        if not np.isfinite(lp):
            return -np.inf
        else:
            self.model.parameter_values = parameters
            self.model.update_model()
            chi2 = self.chi2r
            return -0.5 * self.chi2r * self.model.n_points if chi2 < chi2max else -np.inf

    def save(self, path, name, mode='txt'):
        """
        Saves the fitting results to text-files

        :param path:
        :param name:
        """
        filename = os.path.join(path, name)
        if mode == 'txt':
            csv = mfm.io.txt_csv.Csv()
            csv.save(self.weighted_residuals, filename+'_wr.txt')
            csv.save(self.model[:], filename+'_fit.txt')
            if isinstance(self.data, mfm.Curve):
                self.data.save(filename+'_data.txt')
                with open(filename+'_info.txt', 'w') as fp:
                    fp.write(str(self))

    def __getitem__(self, key):
        self.model.update_model()
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
        s += "--------\n"
        s += str(self.data)
        s += "\n\nFit-result: \n"
        s += "----------\n"
        s += "fitrange: %i..%i\n" % (self.xmin, self.xmax)
        s += "chi2:\t%.4f \n" % self.chi2r
        s += "\n"
        s += str(self.model)
        return s

    def run(self):
        """
        Runs a fit

        :param xmin: int
        :param xmax: int
        :param verbose: bool
        """
        kwargs = self.fitting_parameters

        self.xmin = kwargs.get('xmin', self.xmin)
        self.xmax = kwargs.get('xmax', self.xmax)
        fitting_options = kwargs.get('fitting_options', dict())
        fitting_type = kwargs.get('fitting_type', 'leastsq')

        verbose = kwargs.get('verbose', mfm.verbose)
        if verbose:
            print("fitrange: %s..%s" % (self.xmin, self.xmax))
            print("Parameter-name\tInitial-value\tBounds")
            print self.model
        if fitting_type == 'leastsq':
            if verbose:
                print("Fitting using leastsqt...")
                print "Fitting parameters"
                for p in zip(self.model.parameter_names, self.model.parameter_values, self.model.parameter_bounds):
                    print "%s\t%s\t%s" % (p[0], p[1], p[2])
            self.results = leastsqbound(get_wres, self.model.parameter_values, args=(self.model, self.data),
                                        bounds=self.model.parameter_bounds, **fitting_options)
        elif fitting_type == 'brute':
            if verbose:
                print("Fitting using brute force...")
                print "Ranges:"
                ranges = self.model.parameter_bounds
                print ranges

            save_output = fitting_options.pop('save_full_output', False)
            output_filename = fitting_options.pop('output_file', 'brute_force_fit')
            r = scipy.optimize.brute(get_chi2, args=(self.model, self.data), ranges=ranges, **fitting_options)
            self.results = r[0]
            if save_output:
                if verbose:
                    print "Saving fitting results:"
                    print "Filename: %s" % output_filename
                np.savetxt(output_filename+"_grid.csv", r[2].flatten())
                np.savetxt(output_filename+"_jout.csv", r[3].flatten())

        self.is_fitted = True
        self.min_chi2 = min(self.min_chi2, self.chi2r)
        if verbose:
            print("Fitting-result:\n")
            print(self)
        self.model.finalize()

    def update(self):
        self.model.update()


class FitQtThread(Fit, QtCore.QThread):

    def __init__(self, **kwargs):
        parent = kwargs.get('parent', None)
        self.parent = parent
        Fit.__init__(self, **kwargs)
        QtCore.QThread.__init__(self)
        self.surface = ErrorSurfaceThread(self)
        self.__exiting = False

    def close(self):
        if isinstance(self.parent, QtGui.QWidget):
            self.parent.close()

    def clean(self):
        new = Fit()
        new.xmin, new.xmax = self.xmin, self.xmax
        new.data = self.data.clean()
        new.model = self.model.clean(new)
        return new

    def run(self):
        Fit.run(self)
        self.__exiting = True


class ErrorSurface(Surface):

    methods = ['emcee', 'mcmc']

    def __init__(self, **kwargs):
        Surface.__init__(self, **kwargs)
        self.nprocs = kwargs.get('nprocs', 1)
        self.steps = kwargs.get('steps', 1000)
        self.temperature = kwargs.get('temperature', 1.0)
        self.step_size = kwargs.get('step_size', 0.01)
        self.thin = kwargs.get('thin', 10)
        self.nwalker = kwargs.get('nwalker', 10)
        self.method = kwargs.get('method', 'emcee')
        self.maxchi2 = kwargs.get('maxchi2', 1e12)

    def set_sampling_parameter(self, steps, step_size, nprocs, temperature, thin,
                             nwalker, method='emcee', maxchi2=1e12):
        self.nprocs = nprocs
        self.steps = steps
        self.temperature = temperature
        self.step_size = step_size
        self.thin = thin
        self.nwalker = nwalker
        self.method = method
        self.maxchi2 = maxchi2

    def run(self):
        print("Surface:run")
        self.clear()
        fit = self.fit.clean()
        self.parameter_names = fit.model.parameter_names + ['chi2r']
        self.parameter_names = self.fit.model.parameter_names

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
            chi2 = lnProb * -2.0 / float(self.fit.model.n_points - self.fit.model.n_free - 1.0)
            mask = np.where(np.isfinite(chi2))
            self._chi2.append(chi2[mask])
            self._parameter.append(parameter[mask])


class ErrorSurfaceThread(ErrorSurface, QtCore.QThread):

    def __init__(self, fit, parent=None):
        ErrorSurface.__init__(self, fit=fit)
        QtCore.QThread.__init__(self, parent)
        self.exiting = False
        self.connect(self, QtCore.SIGNAL("finished()"), self.onRunDone)

    def __del__(self):
        self.exiting = True
        self.wait()

    def onRunDone(self):
        print("onRunDone")
        self.fit.model.update_plots()


class FittingWidget(QtGui.QWidget):

    @property
    def name(self):
        if self._name is None:
            return self.fit.name
        else:
            return self._name

    @name.setter
    def name(self, v):
        self._name = v

    @property
    def xmin(self):
        return int(self.spinBox.value())

    @xmin.setter
    def xmin(self, v):
        self.spinBox.setValue(int(v))

    @property
    def xmax(self):
        return int(self.spinBox_2.value())

    @xmax.setter
    def xmax(self, v):
        self.spinBox_2.setValue(int(v))

    @property
    def fit_types(self):
        return mfm.settings['fitting'].keys()

    @property
    def current_fit_type(self):
        return str(self.comboBox.currentText())

    @property
    def current_fitting_options(self):
        return mfm.settings['fitting'][self.current_fit_type]

    def change_dataset(self):
        # TODO: sth is broken here anyway a lot has to be changed
        if mfm.verbose:
            print "Change dataset: %s" % self.name
        self.fit.data = self.curve_select.selected_curve
        self.fit.update()
        self.fit.unlink(self.fit.data)
        self.fit.link(self.curve_select.selected_curve)
        self.lineEdit.setText(self.curve_select.curve_name)

    def __init__(self, fit=None, **kwargs):
        QtGui.QWidget.__init__(self)
        self.curve_select = widgets.CurveSelector(parent=None, fit=self, change_event=self.change_dataset)
        self.curve_select.hide()
        hide_fit_button = kwargs.get('hide_fit_button', False)
        auto_range = kwargs.get('auto_range', True)
        hide_range = kwargs.get('hide_range', False)
        hide_fitting = kwargs.get('hide_fitting', False)
        self.fit = fit
        self._name = kwargs.get('name', None)

        uic.loadUi("mfm/ui/fittingWidget.ui", self)
        self.lineEdit.setText(self.fit.data.filename)

        self.connect(self.checkBox_autorange, QtCore.SIGNAL("stateChanged(int)"), self.onAutoFitRange)
        self.connect(self.actionFit, QtCore.SIGNAL('triggered()'), self.onRunFit)
        self.connect(self.actionChange_dataset, QtCore.SIGNAL('triggered()'), self.curve_select.show)
        self.connect(self.checkBox_autorange, QtCore.SIGNAL("stateChanged(int)"), self.onAutoFitRange)
        self.connect(self.actionFit_range_changed, QtCore.SIGNAL('triggered()'), self.onFitRangeChanged)
        self.connect(self.fit, QtCore.SIGNAL("finished()"), self.onFitDone)
        self.connect(self.fit, QtCore.SIGNAL("stopped()"), self.onFitDone)

        self.comboBox.addItems(self.fit_types)
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
        self.fit.model.update()
        if mfm.settings['fitting_message']:
            widgets.MyMessageBox('Fitting finished', info=str(self.fit))
        self.pushButton_fit.setEnabled(True)

    def onFitRangeChanged(self):
        print "onFitRangeChanged"
        print "xmin: %s xmax: %s" % (self.xmin, self.xmax)
        self.fit.xmin = self.xmin - 1
        self.fit.xmax = self.xmax - 1
        self.fit.model.update()

    def onRunFit(self):
        #self.pushButton_fit.setEnabled(False)
        self.fit.fitting_parameters = {
            "fitting_options": self.current_fitting_options,
            "fitting_type": self.current_fit_type
        }
        self.fit.run()
        self.fit.model.update()

    def onAutoFitRange(self):
        autofit = self.checkBox_autorange.isChecked()
        if autofit:
            try:
                xmin, xmax = self.fit.data.setup.autofitrange(self.fit.data)
                self.fit.xmin = xmin
                self.fit.xmax = xmax
            except AttributeError:
                xmin, xmax = 1, 1
            self.spinBox.setValue(int(xmin))
            self.spinBox_2.setValue(int(xmax))


class ErrorWidget(QtGui.QWidget):

    def __init__(self, fit=None, **kwargs):
        QtGui.QWidget.__init__(self)
        uic.loadUi("mfm/ui/fitting/errorWidget_2.ui", self)
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
        self.fit.model.update()

    def onChi2MinChanged(self):
        self.lineEdit_4.setText("%.4f" % self.fit.min_chi2)
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
            self.fit.model.update_plots()

    def onSaveChi2Surface(self):
        filename = str(QtGui.QFileDialog.getSaveFileName(self, 'Save File', '.'))
        self.fit.surface.save(filename)

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
        return self.fit.min_chi2

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
        return self.fit.model.n_points

    @property
    def npars(self):
        return self.fit.model.n_free

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
        self.fit.surface.set_sampling_parameter(steps=self.n_steps, nprocs=self.nprocs, step_size=self.step_size,
                                              temperature=self.temperature,
                                              thin=self.thinning, nwalker=self.nwalker,
                                              method=self.method, maxchi2=chi2max)
        self.fit.surface.run()


class FitSubWindow(QtGui.QTabWidget):

    def __init__(self, fit, parent=None):
        QtGui.QTabWidget.__init__(self)
        self.setWindowTitle(fit.name)
        self.setTabShape(QtGui.QTabWidget.Triangular)
        self.setTabPosition(QtGui.QTabWidget.South)

        plots = []
        for plot_class, kwargs in fit.model.plot_classes:
            plot = plot_class(fit, **kwargs)
            plot.link(fit.model)
            plots.append(plot)
            self.addTab(plot, plot.name)
            try:
                parent.plotOptionsLayout.addWidget(plot.pltControl)
                plot.pltControl.hide()
            except AttributeError:
                print("Plot %s does not implement control widget as attribute pltControl." % type(plot_class))
        fit.model.plots = plots
        self.connect(self, QtCore.SIGNAL("currentChanged(int)"), parent.onPlotChanged)
        fit.model.update()
        self.fit = fit
        self.action_parent = parent
        self.confirm = mfm.settings['confirm_close_fit']

    def updateStatusBar(self, msg):
        self.statusBar().showMessage(msg)

    def closeEvent(self, event):
        if self.confirm:
            reply = QtGui.QMessageBox.question(self, 'Message',
                                               "Are you sure to close this fit?:\n%s" % self.fit.name,
                                               QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)

            if reply == QtGui.QMessageBox.Yes:
                event.accept()
                self.action_parent.on_close_current_fit()
            else:
                event.ignore()