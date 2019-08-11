from copy import deepcopy
import math

from PyQt4 import QtGui, QtCore, uic
import numpy as np
import pandas as pd
import mfm
from mfm import DataCurve, plots
from mfm.fitting import Fit, FittingWidget, ErrorWidget
from mfm.fitting.models import Model, ModelWidget
from mfm.fitting.parameter import ParameterWidget, Parameter, AggregatedParameters
from mfm.widgets import CurveSelector
import mfm.fluorescence.tcspc
from mfm.fluorescence.general import e1tn, elte2, distribution2rates, rates2lifetimes, \
    species_averaged_lifetime, fluorescence_averaged_lifetime, transfer_efficency2fdfa
import mfm.math.signal
from mfm import Curve


class Lifetime(AggregatedParameters):

    @property
    def species_averaged_lifetime(self):
        return species_averaged_lifetime(self.lifetime_spectrum)

    @property
    def fluorescence_averaged_lifetime(self):
        return fluorescence_averaged_lifetime(self.lifetime_spectrum)

    @property
    def amplitudes(self):
        vs = np.array([math.sqrt(x.value ** 2) for x in self._amplitudes])
        vs /= vs.sum()
        return vs

    @amplitudes.setter
    def amplitudes(self, vs):
        for i, v in enumerate(vs):
            self._amplitudes[i].value = v

    @property
    def lifetimes(self):
        vs = np.array([math.sqrt(x.value ** 2) for x in self._lifetimes])
        for i, v in enumerate(vs):
            self._lifetimes[i].value = v
        return vs

    @lifetimes.setter
    def lifetimes(self, vs):
        for i, v in enumerate(vs):
            self._lifetimes[i].value = v

    @property
    def lifetime_spectrum(self):
        """
        The interleaved lifetime spectrum of the model
        """
        decay = np.empty(2 * len(self), dtype=np.float64)
        decay[0::2] = self.amplitudes
        decay[1::2] = self.lifetimes
        return decay

    @lifetime_spectrum.setter
    def lifetime_spectrum(self, v):
        for i, amplitude in enumerate(v[::2]):
            lifetime = v[i + 1]
            try:
                self._lifetimes[i].value = lifetime
                self._amplitudes[i].value = amplitude
            except IndexError:
                self.append(amplitude, lifetime)

    @property
    def n(self):
        return len(self._amplitudes)

    def update(self):
        """
        This updates the values of the fitting parameters
        """
        amplitudes = self.amplitudes
        for i, a in enumerate(self._amplitudes):
            a.value = amplitudes[i]

    def append(self, amplitude, lifetime, lower_bound_amplitude=None, upper_bound_amplitude=None, fixed=False,
               bound_on=False, lower_bound_lifetime=None, upper_bound_lifetime=None):
        n = len(self)
        a = Parameter(lb=lower_bound_amplitude, ub=upper_bound_amplitude,
                     value=amplitude, name='x(%s,%i)' % (self.short, n + 1),
                     fixed=fixed, bounds_on=bound_on)
        t = Parameter(lb=lower_bound_lifetime, ub=upper_bound_lifetime,
                     value=lifetime, name='t(%s,%i)' % (self.short, n + 1),
                     fixed=fixed, bounds_on=bound_on)
        self._amplitudes.append(a)
        self._lifetimes.append(t)

    def pop(self):
        a = self._amplitudes.pop()
        l = self._lifetimes.pop()
        return a, l

    def __init__(self, short='L', **kwargs):
        self.short = short
        self._amplitudes = kwargs.get('amplitudes', [])
        self._lifetimes = kwargs.get('lifetimes', [])
        self._name = kwargs.get('name', 'Lifetimes')

    def __len__(self):
        return len(self._amplitudes)

    def clean(self):
        new = Lifetime(self.short)
        new._amplitudes = [a.clean() for a in self._amplitudes]
        new._lifetimes = [a.clean() for a in self._lifetimes]
        return new


class LifetimeWidget(Lifetime, QtGui.QWidget):

    def update(self, *__args):
        QtGui.QWidget.update(self, *__args)
        Lifetime.update(self)
        self._tauf_widget.setText("%.4f" % self.fluorescence_averaged_lifetime)
        self._taux_widget.setText("%.4f" % self.species_averaged_lifetime)

    def __init__(self, gtitle='', model=None, short='', **kwargs):
        QtGui.QWidget.__init__(self)
        Lifetime.__init__(self, short=short)
        self.model = model

        self.layout = QtGui.QVBoxLayout(self)
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        self.layout.setSpacing(0)
        self.layout.setMargin(0)

        self.gb = QtGui.QGroupBox()
        self.gb.setTitle(gtitle)
        self.lh = QtGui.QVBoxLayout()
        self.lh.setSpacing(0)
        self.lh.setMargin(0)
        self.gb.setLayout(self.lh)
        self.layout.addWidget(self.gb)

        lh = QtGui.QHBoxLayout()
        addDonor = QtGui.QPushButton("add")
        lh.addWidget(addDonor)
        removeDonor = QtGui.QPushButton("del")
        lh.addWidget(removeDonor)
        lh.connect(addDonor, QtCore.SIGNAL("clicked()"), self.append)
        lh.connect(removeDonor, QtCore.SIGNAL("clicked()"), self.pop)
        self.lh.addLayout(lh)

        lh = QtGui.QHBoxLayout()

        lh.addWidget(QtGui.QLabel("<tau>x"))
        self._taux_widget = QtGui.QLineEdit()
        lh.addWidget(self._taux_widget)

        self._tauf_widget = QtGui.QLineEdit()
        lh.addWidget(QtGui.QLabel("<tau>F"))
        lh.addWidget(self._tauf_widget)

        self.lh.addLayout(lh)

        self.append(1.0, 4.0, update=False)

    def append(self, x=None, l=None, update=True):
        x = 1.0 if x is None else x
        lt = np.random.ranf() * 10 if l is None else l

        l = QtGui.QHBoxLayout()
        l.setSpacing(0)
        l.setMargin(0)

        n = len(self)
        self._amplitudes.append(
            ParameterWidget('x(%s,%i)' % (self.short, n + 1), x, layout=l, model=self.model, digits=4,
                           text='x(%s,%i)' % (self.short, n + 1))
        )
        self._lifetimes.append(
            ParameterWidget('t(%s,%i)' % (self.short, n + 1), lt, layout=l, model=self.model, digits=4,
                           text='<b>&tau;</b>(%s,%i)' % (self.short, n + 1))
        )
        self.lh.addLayout(l)
        if update:
            self.model.update()

    def pop(self):
        self._amplitudes.pop().close()
        self._lifetimes.pop().close()
        self.model.update()


class FretParameter(AggregatedParameters):
    """
    The FRET-class :py:class:`.FRET` aggregates the parameters relevant for the FRET-rate (kFRET).

    1. :py:attribute:`.FRET.forster_radius`
    2. :py:attribute:`.FRET.kappa2`
    3. :py:attribute:`.FRET.tau0`

    """

    @property
    def forster_radius(self):
        return self._R0.value

    @forster_radius.setter
    def forster_radius(self, v):
        self._R0.value = v

    @property
    def kappa2(self):
        return self._kappa2.value

    @kappa2.setter
    def kappa2(self, v):
        self._kappa2.value = v

    @property
    def tau0(self):
        return self._t0.value

    @tau0.setter
    def tau0(self, v):
        self._t0.value = v

    def __init__(self, tau0=4.0, forster_radius=52.0, name='FRET-rate', kappa2=2. / 3., model=None, **kwargs):
        AggregatedParameters.__init__(self, **kwargs)

        t0 = tau0
        forster_radius = forster_radius
        self._name = name
        kappa2 = kappa2
        self.model = model

        self._t0 = Parameter(name='t0', value=t0, fixed=True)
        self._R0 = Parameter(name='R0', value=forster_radius, fixed=True)
        self._kappa2 = Parameter(name='k2', value=kappa2, fixed=True, lb=0.0, ub=4.0, bounds_on=False)


class FretParameterWidget(FretParameter, QtGui.QWidget):
    # TODO use this widget in all fits!!!
    def __init__(self, model=None, **kwargs):
        QtGui.QWidget.__init__(self)
        FretParameter.__init__(self, **kwargs)
        self.model = model

        self.layout = QtGui.QVBoxLayout(self)
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        self.layout.setSpacing(0)
        self.layout.setMargin(0)

        self.gb = QtGui.QGroupBox()
        self.gb.setTitle('FRET')
        lh = QtGui.QVBoxLayout()
        self.lh.setSpacing(0)
        self.lh.setMargin(0)
        self.gb.setLayout(lh)
        self.layout.addWidget(self.gb)

        t0 = kwargs.get('tau0', 4.0)
        forster_radius = kwargs.get('forster_radius', 52.0)
        kappa2 = kwargs.get('kappa2', 2./3.)

        self._R0 = ParameterWidget('R0', forster_radius, layout=lh, model=self.model, digits=2, fixed=True,
                                  text='R<sub>0</sub>')
        self._t0 = ParameterWidget('t0', t0, layout=lh, model=self.model, digits=3, fixed=True,
                                  text='&tau;<sub>0</sub>')
        self._kappa2 = ParameterWidget('k2', kappa2, layout=lh, model=self.model, digits=3, fixed=True,
                                      lb=0.0, ub=4.0, hide_bounds=True, bounds_on=False,
                                      text='&kappa;<sup>2</sup>')


class Convolve(AggregatedParameters):

    @property
    def len_irf(self):
        return self.irf.y.shape[0]

    @property
    def dt(self):
        return self._dt.value

    @dt.setter
    def dt(self, v):
        self._dt = v

    @property
    def lamp_background(self):
        return self._lb.value

    @lamp_background.setter
    def lamp_background(self, v):
        self._lb.value = v

    @property
    def timeshift(self):
        return self._ts.value

    @timeshift.setter
    def timeshift(self, v):
        self._ts.value = v

    @property
    def start(self):
        return int(self._start.value / self.dt)

    @start.setter
    def start(self, v):
        self._start.value = v

    @property
    def stop(self):
        stop = int(self._stop.value / self.dt)
        return stop

    @stop.setter
    def stop(self, v):
        self._stop.value = v

    @property
    def rep_rate(self):
        return self._rep.value

    @rep_rate.setter
    def rep_rate(self, v):
        self._rep.value = float(v)

    @property
    def scaleing(self):
        return self._scaleing

    @scaleing.setter
    def scaleing(self, v):
        self._scaleing = bool(v)

    #@property
    #def parameters(self):
    #    return [self._dt, self._start, self._stop, self._rep, self._lb, self._ts, self._p0]

    @property
    def do_convolution(self):
        return self._do_convolution

    @do_convolution.setter
    def do_convolution(self, v):
        self._do_convolution = bool(v)

    @property
    def irf(self):
        irf = self._irf
        if isinstance(irf, Curve):
            irf = self._irf
            irf = (irf - self.lamp_background) << self.timeshift
            irf.y[irf.y < 0.0] = 0.0
            return irf
        else:
            x = np.copy(self.fit.data.x)
            y = np.zeros_like(self.fit.data.y)
            y[0] = 1.0
            curve = mfm.Curve(x=x, y=y)
            return curve

    @property
    def _irf(self):
        return self.__irf

    @_irf.setter
    def _irf(self, v):
        self.__irf = v
        if isinstance(v, DataCurve):
            dt = np.average(v.dt)
            self._rep.value = v.setup.rep_rate
            self._dt.value = dt
            self._stop.value = (len(v) - 1) * dt

    @property
    def p0(self):
        return self._p0.value

    def scale(self, decay, data, bg=0.0, **kwargs):
        """
        Scales the decay either by a given number *p0* or by the total fluorescence within a range of (start, stop).
        Here *start* and *stop* are two integers representing the array-index of the decay; the range where the
        fluorescence intensity of the data and the decay should be equal.

        :param decay: numpy-array
            The model fluorescence intensity decay
        :param data: numpy-array
            The experimental fluorescence intensity decay
        :param bg: float
            The background of the experimental fluorescence intensity decay
        :param kwargs: Optional parameters
            If these parameters are not provided the current values of the class instance are taken
            autoscale: bool
                If this is True the model fluorescence intensity decay is scaled to the experimental decay
            start: int
                Defines the start for scaling the decay
            stop: int
                Defines the stop for scaling the decay
        :return: numpy-array
            The rescaled model fluorescence intensity decay
        """
        autoscale = kwargs.get('autoscale', self.autoscale)
        start = kwargs.get('start', self.start)
        stop = kwargs.get('stop', self.stop)
        p0 = kwargs.get('p0', self.p0)

        if autoscale:
            mfm.fluorescence.tcspc.rescale_w_bg(decay, data.y, data.weights, bg, start, stop)
        else:
            decay *= p0
        return decay

    def convolve(self, data, **kwargs):
        """

        :param data: numpy-array
            either an interleaved lifetime spectrum, or a numpy array representing a intensity decay calculated
            at given times starting from zero.
        :param kwargs:
            verbose: bool
                If True more output, default False
            mode: string
                This string defines the convolution mode. Thus, this defines whether the parameter `lifetimes` is
                interpreted as lifetime spectrum or as intensity decay. If *mode* is `per` or `exp` the parameter
                *data* is treated as lifetime spectrum. `per` considers periodic convolution and is a bit slower
                `exp` does not consider periodic effects. If *mode* is `decay` it is treated as intensity decay.
            rep_rate: float
                Optional the repetition rate in MHz. If not provided the current parameters of the class
                :py:attribute:`.rep_rate` are taken.
            irf: numpy-array
                Optional


        :return:
        """
        verbose = kwargs.get('verbose', self.verbose)
        mode = kwargs.get('mode', self.mode)
        dt = kwargs.get('dt', self.dt)
        rep_rate = kwargs.get('rep_rate', self.rep_rate)
        irf = kwargs.get('irf', self.irf)
        scatter = kwargs.get('scatter', 0.0)

        self._decay = np.zeros_like(self.fit.data.y)
        len_decay = self._decay.shape[0]

        if isinstance(irf, mfm.DataCurve):
            irf_y = np.copy(irf.y)
        else:
            irf_y = np.zeros_like(self._decay)
            irf_y[0] = 1.0

        if verbose:
            print("------------")
            print("Convolution:")
            print("Lifetimes: %s" % data)
            print("dt: %s" % dt)

        stop = min([int(self.stop / dt), len_decay])

        if verbose:
            print("Irf: %s" % (irf.name))
            print("Max(Irf): %s" % max(irf.y))
            print("Stop: %s" % stop)
            print("dt: %s" % dt)
            print("Convolution mode: %s" % mode)

        if self.stop > 0:
            if mode == "per":
                mfm.fluorescence.tcspc.fconv_per_cs(self._decay, data, irf_y, start=self.start, stop=stop,
                                                    n_points=irf_y.shape[0], period=1000. / rep_rate,
                                                    dt=dt, conv_stop=stop)
            elif mode == "full":
                self._decay = np.convolve(data, irf_y, mode='full')[:irf_y.shape[0]]
            elif mode == "exp":
                mfm.fluorescence.tcspc.fconv(self._decay, data, irf_y, stop, dt)

        if verbose:
            print("Decay: %s" % self._decay)
            print("Max(Decay): %s" % max(self._decay))

        self._decay += (scatter * irf_y)
        return self._decay

    def __init__(self, fit, **kwargs):
        """

        :param fit:
        :param kwargs:

        Example
        -------

        >>> import mfm
        >>> import mfm.experiments as experimets
        >>> tcspc_setup = experiments.TCSPCSetup(skiprows=9)
        >>> data_set = tcspc_setup.load('./sample_data/ibh/Decay_577D.txt')
        >>> irf_data = tcspc_setup.load('./sample_data/ibh/Prompt.txt')
        >>> fit = mfm.Fit(data=data_set)
        >>> convolve = mfm.models.tcspc.Convolve(fit)
        """
        AggregatedParameters.__init__(self)
        if isinstance(fit, Fit):
            self.data = fit.data
            self.fit = fit
        self._decay = None
        self.verbose = kwargs.get('verbose', mfm.verbose)
        self._p0 = kwargs.get('p0', Parameter(value=1.0, name='p0', fixed=True))
        self._name = kwargs.get('name', 'Convolution')

        dt = kwargs.get('dt', 1.0)
        if isinstance(dt, Parameter):
            self._dt = dt
        else:
            self._dt = Parameter(value=dt, name='dt', fixed=True)

        self._rep = kwargs.get('rep', Parameter(value=10.0, name='rep', fixed=True))
        self._start = kwargs.get('start', Parameter(value=0, name='start', fixed=True))
        self._stop = kwargs.get('stop', Parameter(value=20.0, name='stop', fixed=True))
        self._lb = kwargs.get('lb', Parameter(value=0.0, name='lb'))
        self._ts = kwargs.get('ts', Parameter(value=0.0, name='ts'))
        self._do_convolution = kwargs.get('do_convolution', True)
        self._scaleing = kwargs.get('scaleing', True)
        self.mode = kwargs.get('convolution_mode', '')
        self.autoscale = kwargs.get('autoscale', True)
        self._irf = kwargs.get('irf', None)

    def clean(self, fit):
        dt = self._dt.clean()
        p0 = self._p0.clean()
        lb = self._lb.clean()
        ts = self._ts.clean()
        start = self._start.clean()
        stop = self._stop.clean()
        rep = self._rep.clean()
        new = Convolve(fit, scaleing=self.autoscale, mode=self.mode, p0=p0, lb=lb, ts=ts, start=start, stop=stop,
                       rep=rep, do_convolution=self.do_convolution, dt=dt, irf=self._irf)
        new.autoscale = self.autoscale
        return new


class ConvolveWidget(Convolve, QtGui.QWidget):

    """
    This widget is responsible for convolutions in TCSPC-fits.
    The instrument-response is set using the :py:class:`mfm.widgets.CurveSelector` attribute
    :py:attribute`.irf_select`.


    Examples
    --------


    """

    def __init__(self, fit=None, model=None, **kwargs):
        QtGui.QWidget.__init__(self)
        uic.loadUi('mfm/ui/fitting/models/tcspc/convolveWidget.ui', self)
        show_convolution_mode = kwargs.get('show_convolution_mode', True)
        hide_convolve = kwargs.get('hide_convolve', False)
        hide_curve_convolution = kwargs.get('hide_curve_convolution', False)

        self.hide_curve_convolution(hide_curve_convolution)
        if hide_convolve:
            self.hide()

        if fit is None:
            dt = kwargs.get('dt', 1.0)
            t_stop = 50.0
            t_stop_max = 50.0
        else:
            dt = kwargs.get('dt', fit.data.dt)
            dt = np.average(dt)
            t_stop = len(fit.data) * dt if fit.data is not None else 1.0
            t_stop_max = len(fit.data) * dt if fit.data is not None else 1e6

        if show_convolution_mode is False:
            self.widget.hide()

        l = QtGui.QHBoxLayout()
        l.setSpacing(0)
        l.setMargin(0)
        dt = ParameterWidget('dt', dt, layout=l, model=model, digits=4, fixed=True,
                            lb=1e-4, ub=100, bounds_on=True, hide_bounds=True)
        p0 = ParameterWidget('scale', 1.0, layout=l, model=model, digits=4, fixed=True, text='scale')

        self.verticalLayout_2.addLayout(l)

        l = QtGui.QHBoxLayout()
        l.setSpacing(0)
        l.setMargin(0)
        start = ParameterWidget('start', 0, layout=l, model=model, digits=1, fixed=True,
                               lb=0, ub=t_stop_max, bounds_on=True, hide_bounds=True,
                               text='start')
        stop = ParameterWidget('stop [ns]', t_stop, layout=l, model=model, digits=1, fixed=True,
                              lb=0, ub=t_stop_max, bounds_on=True, hide_bounds=True,
                              text='stop')
        self.verticalLayout_2.addLayout(l)

        l = QtGui.QHBoxLayout()
        l.setSpacing(0)
        l.setMargin(0)
        self.verticalLayout_2.addLayout(l)
        lb = ParameterWidget('lb', 0.0, layout=l, model=model, digits=4)
        ts = ParameterWidget('ts', 0.0, layout=l, model=model, digits=4)

        try:
            rep_rate = fit.data.setup.rep_rate
        except AttributeError:
            rep_rate = 10.0
        rep = ParameterWidget('rate', rep_rate, layout=self.horizontalLayout_3, model=model, digits=1, fixed=True,
                             lb=1e-4, ub=1e9, bounds_on=False, hide_bounds=True,
                             text='r[MHz]')
        kwargs['dt'] = dt
        Convolve.__init__(self, fit, lb=lb, ts=ts, start=start, stop=stop, rep=rep, p0=p0, **kwargs)

        self.irf_select = CurveSelector(parent=None, change_event=self.change_irf, fit=self.fit)
        self.connect(self.actionSelect_IRF, QtCore.SIGNAL('triggered()'), self.irf_select.show)

    def hide_curve_convolution(self, v):
        """
        Hides the convolution option by curves
        :type v: bool
        :return:
        """
        self.radioButton_3.setVisible(not v)

    def change_irf(self, irf_index=None):
        """
        This changes the irf.

        :param irf_index: int
            If this is not provided the irf defined by :py:attribute:`.irf_select` is returned. If this
            is provided the :py:class`mfm.DataCurve` of all loaded curves with the respective index is used as irf.
        :return:
        """
        if isinstance(irf_index, int):
            self.irf_select.selected_index = irf_index
        self._irf = self.irf_select.selected_curve

        self.lineEdit.setText(self.irf_select.currentItem().text())
        self.fit.model.update()

    @property
    def mode(self):
        if self.radioButton_2.isChecked():
            return "exp"
        elif self.radioButton.isChecked():
            return "per"
        elif self.radioButton_3.isChecked():
            return "full"

    @mode.setter
    def mode(self, v):
        if v == "exp":
            self.radioButton_2.setChecked(True)
        elif v == "decay":
            self.radioButton_3.setChecked(True)
        elif v == "per":
            self.radioButton.setChecked(True)

    @property
    def autoscale(self):
        return bool(self.checkBox.checkState())

    @autoscale.setter
    def autoscale(self, v):
        if v is True:
            self.checkBox.setCheckState(2)
        else:
            self.checkBox.setCheckState(0)

    @property
    def do_convolution(self):
        return bool(self.groupBox.isChecked())

    @do_convolution.setter
    def do_convolution(self, v):
        return self.groupBox.setChecked(v)

    @autoscale.setter
    def scaleing(self, v):
        if v is True:
            self.checkBox.setCheckState(2)
        else:
            self.checkBox.setCheckState(0)


class Anisotropy(AggregatedParameters):
    @property
    def r0(self):
        return self._r0.value

    @r0.setter
    def r0(self, v):
        self._r0.value = v

    @property
    def l1(self):
        return self._l1.value

    @l1.setter
    def l1(self, v):
        self._r0.value = v

    @property
    def l2(self):
        return self._l2.value

    @l2.setter
    def l2(self, v):
        self._l2.value = v

    @property
    def g(self):
        return self._g.value

    @g.setter
    def g(self, v):
        self._g.value = v

    @property
    def rho(self):
        r = np.array([rho.value for rho in self._rhos], dtype=np.float64)
        r = r ** 2
        r = np.sqrt(r)
        for i, v in enumerate(r):
            self._rhos[i].value = v
        return r

    @property
    def b(self):
        a = np.sqrt(np.array([g.value for g in self._bs]) ** 2)
        a /= a.sum()
        a *= self.r0
        for i, g in enumerate(self._bs):
            g.value = a[i]
        return a

    @property
    def rotation_spectrum(self):
        rot = np.empty(2 * len(self), dtype=np.float64)
        rot[0::2] = self.b
        rot[1::2] = self.rho
        return rot

    def get_decay(self, lifetime_spectrum):
        a = self.rotation_spectrum
        f = lifetime_spectrum

        d = elte2(a, f)
        vv = np.hstack([f, e1tn(d, 2)])
        vh = e1tn(np.hstack([f, e1tn(d, -1)]), self.g)
        if self.polarization_type.upper() == 'VH':
            return np.hstack([e1tn(vv, self.l2), e1tn(vh, 1 - self.l2)])
        elif self.polarization_type.upper() == 'VV':
            r = np.hstack([e1tn(vv, 1 - self.l1), e1tn(vh, self.l1)])
            return r
        else:
            return f

    def __len__(self):
        return len(self._bs)

    def add_rotation(self, b, rho, lb, ub, fixed, bound_on):
        b = Parameter(lb=lb, ub=ub, value=rho, name='b(%s,%i)' % (self.short, len(self) + 1),
                     fixed=fixed, bounds_on=bound_on)
        rho = Parameter(lb=lb, ub=ub, value=b, name='rho(%s,%i)' % (self.short, len(self) + 1),
                       fixed=fixed, bounds_on=bound_on)
        self._rhos.append(rho)
        self._bs.append(b)

    def remove_rotation(self):
        self._rhos.pop().close()
        self._bs.pop().close()

    def __init__(self, short='aniso', polarization_type='vm', **kwargs):
        self._rhos = []
        self._bs = []
        self.short_name = short
        self.polarization_type = polarization_type

        self._r0 = Parameter(name='r0', value=0.38)
        self._g = Parameter(name='g', value=1.00)
        self._l1 = Parameter(name='l1', value=0.0308)
        self._l2 = Parameter(name='l2', value=0.0368)
        self._name = kwargs.get('name', 'Anisotropy')

    def clean(self):
        new = Anisotropy(short=self.short_name, polarization_type=self.polarization_type)
        new._g = self._g.clean()
        new._l1 = self._l1.clean()
        new._l2 = self._l2.clean()
        new._r0 = self._r0.clean()
        new._rhos = [a.clean() for a in self._rhos]
        new._bs = [a.clean() for a in self._bs]
        return new


class AnisotropyWidget(Anisotropy, QtGui.QWidget):

    @property
    def polarization(self):
        if self.radioButtonVM.isChecked():
            return 0.0, 54.7
        elif self.radioButtonVV.isChecked():
            return 0.0, 0.0
        elif self.radioButtonVH.isChecked():
            return 0.0, 90.0

    @property
    def polarization_type(self):
        if self.radioButtonVM.isChecked():
            return 'vm'
        elif self.radioButtonVV.isChecked():
            return 'vv'
        elif self.radioButtonVH.isChecked():
            return 'vh'

    @polarization_type.setter
    def polarization_type(self, v):
        if v == 'vm':
            self.radioButtonVM.setChecked(True)
        elif v == 'vv':
            self.radioButtonVV.setChecked(True)
        elif v == 'vh':
            self.radioButtonVH.setChecked(True)

    def __init__(self, model, short='', show_selector=True, **kwargs):
        """
        param show_selector : if show_selector is True a selector for VV, VH, VM is displayed
        """
        QtGui.QWidget.__init__(self)
        if kwargs.get('hide_anisotropy', False):
            self.hide()
        self.model = model
        self.radioButtonVM = QtGui.QRadioButton("VM")
        self.radioButtonVV = QtGui.QRadioButton("VV")
        self.radioButtonVH = QtGui.QRadioButton("VH")

        Anisotropy.__init__(self, short=short)

        self.layout = QtGui.QVBoxLayout(self)
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        self.layout.setSpacing(0)
        self.layout.setMargin(0)
        self.setLayout(self.layout)

        self.gb = QtGui.QGroupBox()
        self.gb.setCheckable(True)
        self.gb.setTitle("Rotational-times")
        lh = QtGui.QHBoxLayout()
        lh.setSpacing(0)
        lh.setMargin(0)
        self.gb.setLayout(lh)
        self.layout.addWidget(self.gb)

        gb_widget = QtGui.QWidget()
        self.lh = QtGui.QVBoxLayout()
        self.lh.setSpacing(0)
        self.lh.setMargin(0)
        gb_widget.setLayout(self.lh)
        lh.addWidget(gb_widget)

        if show_selector:
            l = QtGui.QHBoxLayout()
            l.setSpacing(0)
            l.setMargin(0)
            l.addWidget(self.radioButtonVM)
            l.addWidget(self.radioButtonVV)
            l.addWidget(self.radioButtonVH)
            spacerItem = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
            l.addItem(spacerItem)
            self.lh.addLayout(l)

        l = QtGui.QHBoxLayout()
        self._r0 = ParameterWidget('r0', 0.38, layout=l, model=self.model, digits=3, fixed=True)
        self._g = ParameterWidget('g', 1.00, layout=l, model=self.model, digits=2, fixed=True)
        self.lh.addLayout(l)

        l = QtGui.QHBoxLayout()
        self._l1 = ParameterWidget('l1', 0.0308, layout=l, model=self.model, digits=4, fixed=True)
        self._l2 = ParameterWidget('l2', 0.0368, layout=l, model=self.model, digits=4, fixed=True)
        self.lh.addLayout(l)

        l = QtGui.QHBoxLayout()
        l.setSpacing(0)
        l.setMargin(0)

        add_rho = QtGui.QPushButton()
        add_rho.setText("add")
        l.addWidget(add_rho)
        l.connect(add_rho, QtCore.SIGNAL("clicked()"), self.add_rotation)

        remove_rho = QtGui.QPushButton()
        remove_rho.setText("del")
        l.addWidget(remove_rho)
        l.connect(remove_rho, QtCore.SIGNAL("clicked()"), self.remove_rotation)

        self.lh.addLayout(l)
        self.connect(self.gb, QtCore.SIGNAL("toggled(bool)"), self.onToggeled)
        self.widgets_hide = [self._r0, self._g, self._l1, self._l2, add_rho, remove_rho, gb_widget]
        self.gb.setChecked(False)

    def onToggeled(self, value):
        if value is False:
            self.polarization_type = 'vm'
            for w in self.widgets_hide:
                w.hide()
        else:
            for w in self.widgets_hide:
                w.show()

    def add_rotation(self):
        print("onAddRotation")
        l = QtGui.QHBoxLayout()
        l.setSpacing(0)
        self.lh.addLayout(l)
        rho = ParameterWidget('rho(%i)' % (len(self) + 1), '5.0', layout=l,
                             model=self.model, digits=2)
        x = ParameterWidget('b(%i)' % (len(self) + 1), '0.5', layout=l,
                           model=self.model, digits=2)
        self._rhos.append(rho)
        self._bs.append(x)
        self.model.update()

    def remove_rotation(self):
        print("onRemoveRotation")
        self._rhos.pop().close()
        self._bs.pop().close()
        self.model.update()


class Generic(AggregatedParameters):

    @property
    def scatter(self):
        return self._sc.value

    @scatter.setter
    def scatter(self, v):
        self._sc.value = v

    @property
    def background(self):
        return self._bg.value

    @background.setter
    def background(self, v):
        self._bg.value = v

    def __init__(self, **kwargs):
        self._name = kwargs.get('name', 'Nuisance')
        if len(kwargs) > 0:
            self._sc = kwargs.get('sc', Parameter(0.0, name='sc'))
            self._bg = kwargs.get('bg', Parameter(0.0, name='bg'))

    def clean(self):
        sc = self._sc.clean()
        bg = self._bg.clean()
        return Generic(sc=sc, bg=bg)


class GenericWidget(Generic, QtGui.QWidget):

    def __init__(self, model, **kwargs):
        Generic.__init__(self)
        QtGui.QWidget.__init__(self)

        self.model = model
        self.parent = kwargs.get('parent', None)
        if kwargs.get('hide_generic', False):
            self.hide()

        self.layout = QtGui.QVBoxLayout(self)
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        self.layout.setSpacing(0)
        self.layout.setMargin(0)
        gb = QtGui.QGroupBox()
        gb.setTitle("Generic")
        self.layout.addWidget(gb)

        gbl = QtGui.QVBoxLayout()
        gbl.setSpacing(0)
        gbl.setMargin(0)

        gb.setLayout(gbl)
        # Generic parameters
        l = QtGui.QHBoxLayout()
        l.setSpacing(0)
        l.setMargin(0)
        gbl.addLayout(l)
        self._sc = ParameterWidget('sc', 0.0, layout=l, model=self.model, digits=4)
        self._bg = ParameterWidget('bg', 0.0, layout=l, model=self.model, digits=4)

        l = QtGui.QHBoxLayout()
        l.setSpacing(0)
        l.setMargin(0)
        gbl.addLayout(l)


class Corrections(AggregatedParameters):

    @property
    def lintable(self):
        if self.enabled:
            return self._lintable[::-1] if self.reverse else self._lintable
        else:
            return None
    @property
    def measurement_time(self):
        return self._measurement_time.value

    @measurement_time.setter
    def measurement_time(self, v):
        self._measurement_time.value = v

    @property
    def dead_time(self):
        return self._dead_time.value

    @dead_time.setter
    def dead_time(self, v):
        self._dead_time.value = v

    def calc_lin_range(self):
        y = self.data.y
        self.xmin = np.where(y > 5000)[0][0]
        self.xmax = np.where(y > 5000)[0][-1]

    def _calcLinTable(self):
        if self.enabled:
            x2 = deepcopy(self.data.y)
            x2 /= x2[self.xmin: self.xmax].mean()
            mnx = np.ma.array(x2)
            mask1 = x2 < self.threshold
            mask2 = np.array([i < self.xmin or i > self.xmax for i in range(len(x2))])
            mnx.mask = mask1 + mask2
            mnx.fill_value = 1.0
            mnx /= mnx.mean()
            yn = mnx.filled()
            self._lintable = mfm.math.signal.window(yn,
                                                    self.window_length,
                                                    mfm.math.signal.windowTypes[self.window_function])

    def __init__(self, fit, model, **kwargs):
        AggregatedParameters.__init__(self)
        self.fit = fit
        self.model = model
        self.threshold = kwargs.get('threshold', 0.9)
        self.reverse = kwargs.get('reverse', False)
        self.enabled = kwargs.get('enabled', False)
        self.xmin = kwargs.get('xmin', 0)
        self.xmax = kwargs.get('xmax', 0)
        self.window_length = kwargs.get('window_length', 17)
        self.window_function = kwargs.get('window_function', 17)
        self.correct_pile_up = kwargs.get('correct_pile_up', False)
        self._measurement_time = kwargs.get('measurement_time', Parameter(1000.0, name='measurement_time'))
        self._dead_time = kwargs.get('dead_time', Parameter(1000.0, name='dead_time'))
        self._name = kwargs.get('name', 'Corrections')

    def clean(self, fit):
        cor = Corrections(fit, threshold=self.threshold, reverse=self.reverse,
                          enabled=self.enabled, xmin=self.xmin, xmax=self.xmax)
        if self.enabled:
            cor._lintable = self._lintable
        return cor


class CorrectionsWidget(Corrections, QtGui.QWidget):

    def __init__(self, fit, model, **kwargs):
        QtGui.QWidget.__init__(self)
        if kwargs.get('hide_corrections', False):
            self.hide()
        uic.loadUi("mfm/ui/fitting/models/tcspc/tcspcCorrections.ui", self)
        self.groupBox.setChecked(False)
        Corrections.__init__(self, fit, model=model, threshold=0.9, reverse=False, enabled=False)
        self.comboBox.addItems(mfm.math.signal.windowTypes)
        self.connect(self.pushButton_2, QtCore.SIGNAL("clicked()"), self.plot_lintable)
        self.connect(self.comboBox_2, QtCore.SIGNAL("currentIndexChanged(int)"), self.onLinTableChanged)
        self.connect(self.spinBox, QtCore.SIGNAL("valueChanged (int)"), self._calcLinTable)
        self.connect(self.spinBox_2, QtCore.SIGNAL("valueChanged (int)"), self._calcLinTable)
        self.connect(self.spinBox_3, QtCore.SIGNAL("valueChanged (int)"), self._calcLinTable)
        self.connect(self.doubleSpinBox, QtCore.SIGNAL("valueChanged (double)"), self._calcLinTable)
        self.connect(self.comboBox, QtCore.SIGNAL("currentIndexChanged(int)"), self._calcLinTable)
        self.connect(self.checkBox_3, QtCore.SIGNAL("stateChanged (int)"), self.calc_lin_range)
        self.connect(self.groupBox_2, QtCore.SIGNAL('clicked()'), self.on_update_lin)

        self._measurement_time = ParameterWidget('meas.t[s]', 1000.0, layout=self.horizontalLayout_2,
                                                model=self.model, digits=0, lb=0.0,
                                                ub=1e6, bounds_on=True, hide_bounds=True, fixed=True)
        self._dead_time = ParameterWidget('dead[ns]', 85.0, layout=self.horizontalLayout_2,
                                         model=self.model, digits=0, lb=0.0,
                                         ub=1e6, bounds_on=True, hide_bounds=True, fixed=True)

    def on_update_lin(self):
        self._lintables = mfm.get_data_curves()
        names = [d.name for d in self._lintables]
        self.comboBox_2.clear()
        self.comboBox_2.addItems(names)

    def onLinTableChanged(self):
        self.calc_lin_range()

    @property
    def correct_pile_up(self):
        return bool(self.groupBox_3.isChecked())

    @correct_pile_up.setter
    def correct_pile_up(self, v):
        if v:
            self.groupBox_3.setChecked(True)
        else:
            self.groupBox_3.setChecked(False)

    @property
    def data(self):
        try:
            return self._lintables[self.comboBox_2.currentIndex()]
        except:
            return None

    @data.setter
    def data(self, v):
        # data-setter should have no effect in widget
        pass

    def _calcLinTable(self):
        Corrections._calcLinTable(self)
        try:
            self.fit.model.update()
        except AttributeError:
            print("no fit assigned to correction widget ")

    @property
    def enabled(self):
        return bool(self.groupBox_2.isChecked())

    @enabled.setter
    def enabled(self, v):
        self.groupBox_2.setChecked(v)

    @property
    def reverse(self):
        return bool(self.checkBox_2.isChecked())

    @reverse.setter
    def reverse(self, v):
        self.checkBox_2.setCheckState(v)

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
    def threshold(self):
        return float(self.doubleSpinBox.value())

    @threshold.setter
    def threshold(self, v):
        self.doubleSpinBox.setValue(float(v))

    @property
    def window_length(self):
        return int(self.spinBox_3.value())

    @window_length.setter
    def window_length(self, v):
        self.spinBox_3.setValue(int(v))

    @property
    def window_function(self):
        return self.comboBox.currentIndex()

    @window_function.setter
    def window_function(self, v):
        self.comboBox.setCurrentIndex(v)

    def plot_lintable(self):
        import pylab as p

        p.plot(self.lintable)
        p.show()


class LifetimeModel(Model, Curve):

    name = "Lifetime fit"

    def __str__(self):
        s = Model.__str__(self)
        s += "\n------------------\n"
        s += "\nAverage Lifetimes:\n"
        s += "<tau>x: %.3f\n<tau>F: %.3f\n" % (self.species_averaged_lifetime, self.fluorescence_averaged_lifetime)
        s += "\n------------------\n"
        return s

    def __init__(self, fit, **kwargs):
        self.fit = fit
        Curve.__init__(self)
        Model.__init__(self, fit=fit)
        self.generic = kwargs.get('generic', Generic(**kwargs))
        self.corrections = kwargs.get('corrections', Corrections(fit, model=self, **kwargs))
        self.anisotropy = kwargs.get('anisotropy', Anisotropy(**kwargs))
        self.donors = kwargs.get('donors', Lifetime(**kwargs))
        self.convolve = kwargs.get('convolve', Convolve(fit, **kwargs))
        self.verbose = kwargs.get('verbose', mfm.verbose)

    @property
    def species_averaged_lifetime(self):
        return species_averaged_lifetime(self.lifetime_spectrum)

    @property
    def fluorescence_averaged_lifetime(self):
        return fluorescence_averaged_lifetime(self.lifetime_spectrum, self.species_averaged_lifetime)

    @property
    def lifetime_spectrum(self):
        return self.donors.lifetime_spectrum

    def finalize(self):
        self.donors.update()

    def decay(self, time):
        """This is returns the decay-pattern of the model

        :param time: numpy array
            The times at which the intensity is calculated
        :return: numpy-array
            The decay-pattern
        """
        x, l = self.lifetime_spectrum.reshape((self.lifetime_spectrum.shape[0]/2), 2).T
        f = np.array([np.dot(x, np.exp(- t / l)) for t in time])
        return f

    def update_model(self, **kwargs):
        """
        This recalculates the decay. The

        :param kwargs:

            `lifetime_spectrum`
            `verbose` a boolean
            `irf` a :py:class:`Curve` object
            `scatter` float
            `background` float
            `lintable` numpy-array

        :return:
        """
        verbose = kwargs.get('verbose', mfm.verbose)
        lifetime_spectrum = kwargs.get('lifetime_spectrum', self.lifetime_spectrum)
        irf = kwargs.get('irf', self.convolve.irf)
        scatter = kwargs.get('scatter', self.generic.scatter)
        background = kwargs.get('background', self.generic.background)
        lintable = kwargs.get('lintable', self.corrections.lintable)

        if verbose:
            print("--------------------")
            print("Calculation of Decay")

        if irf is not None:
            lt = self.anisotropy.get_decay(lifetime_spectrum)
            decay = self.convolve.convolve(lt, verbose=verbose, scatter=scatter)
            if verbose:
                print("Lifetimes: %s" % lt)
                print("Decay: %s" % decay)
                print("Max(Decay) %s" % max(decay))
            if self.corrections.correct_pile_up:
                mfm.fluorescence.tcspc.pile_up(self.fit.data.y, decay, self.convolve.rep_rate,
                                               self.corrections.dead_time, self.corrections.measurement_time)
            decay = self.convolve.scale(decay, self.fit.data, background, **kwargs)
            decay += background
            decay[decay < 0.0] = 0.0
            if lintable is not None:
                decay *= lintable
            self.y_values = decay

        if verbose:
            print "------"
            if isinstance(self, FRETModel):
                print "Transfer-efficency: %s " % self.transfer_efficiency
                print "FRET tauX: %s " % self.fret_species_averaged_lifetime
                print "FRET tauF: %s " % self.fret_fluorescence_averaged_lifetime
                print "Donor tauX: %s " % self.donor_species_averaged_lifetime
                print "Donor tauF: %s " % self.donor_fluorescence_averaged_lifetime
            print "<tau>x: %s " % self.species_averaged_lifetime
            print "<tau>f: %s " % self.fluorescence_averaged_lifetime
            print "------"

    def clean(self, fit):
        generic = self.generic.clean()
        donors = self.donors.clean()
        corrections = self.corrections.clean(fit)
        conv = self.convolve.clean(fit)
        aniso = self.anisotropy.clean()
        new = LifetimeModel(fit, generic=generic, donors=donors, corrections=corrections, convolve=conv,
                            anisotropy=aniso)
        return new


class LifetimeModelWidget(LifetimeModel, ModelWidget):

    plot_classes = [(plots.LinePlot, {'d_scalex': 'lin',
                                                  'd_scaley': 'log',
                                                  'r_scalex': 'lin',
                                                  'r_scaley': 'lin',
    }),
                    (plots.SurfacePlot, {})
    ]

    def update_model(self, **kwargs):
        LifetimeModel.update_model(self, **kwargs)
        self.donors.update()

    def __init__(self, fit, **kwargs):
        ModelWidget.__init__(self)
        self.icon = QtGui.QIcon(":/icons/icons/TCSPC.ico")
        self.fit = fit

        corrections = CorrectionsWidget(fit, model=self, **kwargs)
        if not kwargs.get('disable_fit', False):
            fitting_widget = FittingWidget(fit, **kwargs)
        else:
            fitting_widget = QtGui.QLabel()
        generic = GenericWidget(parent=self, model=self, **kwargs)
        anisotropy = AnisotropyWidget(model=self, short='rL', **kwargs)
        donors = kwargs.get('donors', LifetimeWidget(model=self, parent=self, gtitle='Lifetimes', short='L'))
        error_widget = ErrorWidget(fit, **kwargs)
        convolve = ConvolveWidget(fit=fit, model=self, dt=fit.data.dt, **kwargs)
        convolve.hide_curve_convolution(True)
        LifetimeModel.__init__(self, fit, generic=generic, donors=donors, convolve=convolve,
                               anisotropy=anisotropy, corrections=corrections)
        layout = QtGui.QVBoxLayout(self)
        layout.setMargin(0)
        layout.setSpacing(0)
        layout.setAlignment(QtCore.Qt.AlignTop)
        ## add widgets
        layout.addWidget(fitting_widget)
        layout.addWidget(convolve)
        layout.addWidget(generic)
        layout.addWidget(donors)
        layout.addWidget(anisotropy)
        layout.addWidget(corrections)
        layout.addWidget(error_widget)
        self.setLayout(layout)

        if kwargs.get('hide_corrections', False):
            corrections.hide()
        if kwargs.get('hide_fit', False):
            fitting_widget.hide()
        if kwargs.get('hide_generic', False):
            generic.hide()
        if kwargs.get('hide_convolve', False):
            convolve.hide()
        if kwargs.get('hide_rotation', False):
            anisotropy.hide()
        if kwargs.get('hide_error', False):
            error_widget.hide()


class FRETModel(LifetimeModel):

    @property
    def forster_radius(self):
        """
        The Forster-radius of the FRET-pair
        """
        return self._forster_radius

    @property
    def tau0(self):
        """
        The lifetime of the donor in absence of additional quenching
        """
        return self._tau0

    @property
    def kappa2(self):
        """
        The mean orientation factor
        """
        return self._kappa2

    @property
    def distance_distribution(self):
        """
        The distribution of distances. The distribution should be 3D numpy array of the form

            gets distribution in form: (1,2,3)
            0: number of distribution
            1: amplitude
            2: distance

        """
        pass

    @property
    def donly(self):
        """
        The fractions of donor-only (No-FRET) species. By default no donor-only is assumed. This has to be
        implemented by the model anyway.
        """
        return 0.0

    @property
    def rate_spectrum(self):
        """
        The FRET-rate spectrum. This takes the distance distribution of the model and calculated the resulting
        FRET-rate spectrum (excluding the donor-offset).
        """

        try:
            rs = distribution2rates(self.distance_distribution, self.tau0, self.kappa2,
                                    self.forster_radius)
            r = np.hstack(rs).ravel([-1])
            return r
        except TypeError:
            return np.array([1.0, 1.0], dtype=np.float64)

    @property
    def lifetime_spectrum(self):
        """
        This returns the lifetime-spectrum of the model including the donor-only offset (donor-only fraction)
        """
        return rates2lifetimes(self.rate_spectrum,
                               self.donors.lifetime_spectrum,
                               self.donly)

    @property
    def donor_lifetime_spectrum(self):
        """
        The donor lifetime spectrum in form amplitude, lifetime, amplitude, lifetime
        """
        return self.donors.lifetime_spectrum

    @donor_lifetime_spectrum.setter
    def donor_lifetime_spectrum(self, v):
        self.model.donors.lifetime_spectrum = v

    @property
    def donor_species_averaged_lifetime(self):
        """
        The current species averaged lifetime of the donor sample xi*taui
        """
        return self.donors.species_averaged_lifetime

    @property
    def donor_fluorescence_averaged_lifetime(self):
        """
        The current species averaged lifetime of the donor sample xi*taui
        """
        return self.donors.fluorescence_averaged_lifetime

    @property
    def fret_species_averaged_lifetime(self):
        """
        The current species averages lifetime of the FRET sample xi * taui
        """
        return self.species_averaged_lifetime

    @property
    def fret_fluorescence_averaged_lifetime(self):
        """
        The current fluorescence averaged lifetime of the FRET-sample = xi*taui**2 / species_averaged_lifetime
        """
        return self.fluorescence_averaged_lifetime

    @property
    def transfer_efficiency(self):
        """
        The current transfer efficency of the model (this includes donor-only)
        """
        return 1.0 - self.fret_species_averaged_lifetime / self.donor_species_averaged_lifetime

    @property
    def donors(self):
        return self._donors

    @donors.setter
    def donors(self, v):
        self._donors = v

    def get_transfer_efficency(self, phiD, phiA):
        """ Get the current donor-acceptor fluorescence intensity ratio
        :param phiD: float
            donor quantum yield
        :param phiA:
            acceptor quantum yield
        :return: float, transfer efficency
        """
        return transfer_efficency2fdfa(self.transfer_efficiency, phiD, phiA)

    def __init__(self, fit, **kwargs):
        LifetimeModel.__init__(self, fit, **kwargs)
        self._forster_radius = kwargs.get('forster_radius', 52.0)
        self._tau0 = kwargs.get('tau0', 4.0)
        self._kappa2 = kwargs.get('kappa2', 2./3.)
        self._donors = kwargs.get('donors', Lifetime('D'))
        self._donly = kwargs.get('donly', 0.0)


class SingleDistanceModel(FRETModel):

    name = "Fixed distance distribution"

    @property
    def donly(self):
        return self._donly.value

    @property
    def distance_distribution(self):
        n_points = self.n_points_dist
        r = np.vstack([self.prda, self.rda]).reshape([1, 2,  n_points])
        return r

    @property
    def n_points_dist(self):
        """
        The number of points in the distribution
        """
        return self.prda.shape[0]

    @property
    def rda(self):
        return self._rda

    @rda.setter
    def rda(self, v):
        self._rda = v

    @property
    def prda(self):
        p = self._prda
        p /= sum(p)
        return p

    @prda.setter
    def prda(self, v):
        self._prda = v

    def __init__(self, **kwargs):
        FRETModel.__init__(self, **kwargs)
        self._rda = kwargs.get('rda', np.array([100.0]))
        self._prda = kwargs.get('prda', np.array([100.0]))
        self._donly = kwargs.get('donly', Parameter(0.0))


class SingleDistanceModelWidget(QtGui.QWidget, SingleDistanceModel):

    plot_classes = [(plots.LinePlot, {'d_scalex': 'lin',
                                                  'd_scaley': 'log',
                                                  'r_scalex': 'lin',
                                                  'r_scaley': 'lin',
    }),
                    (plots.SurfacePlot, {})
    ]

    def __init__(self, fit, **kwargs):
        QtGui.QWidget.__init__(self)
        self.icon = QtGui.QIcon(":/icons/icons/TCSPC.ico")
        self.fit = fit

        anisotropy = AnisotropyWidget(model=self, short='rL', **kwargs)
        convolve = ConvolveWidget(fit=fit, model=self, **kwargs)
        donors = LifetimeWidget(parent=self, model=self, gtitle='Donor(0)', short='D')
        generic = GenericWidget(model=self, parent=self, **kwargs)
        fitting = QtGui.QLabel() if kwargs.get('disable_fit', False) else FittingWidget(fit=fit, **kwargs)
        errors = ErrorWidget(fit, **kwargs)
        corrections = CorrectionsWidget(fit, model=self, **kwargs)
        donly = ParameterWidget('dOnly', 0.0, self)

        SingleDistanceModel.__init__(self, fit=fit, convolve=convolve, corrections=corrections,
                                            generic=generic, donors=donors, donly=donly, anisotropy=anisotropy)

        uic.loadUi('mfm/ui/fitting/models/tcspc/load_distance_distibution.ui', self)
        self.icon = QtGui.QIcon(":/icons/icons/TCSPC.ico")
        self.connect(self.actionOpen_distirbution, QtCore.SIGNAL('triggered()'), self.load_distance_distribution)

        self.verticalLayout.addWidget(fitting)
        self.verticalLayout.addWidget(convolve)
        self.verticalLayout.addWidget(generic)
        self.verticalLayout.addWidget(donly)
        self.verticalLayout.addWidget(donors)
        self.verticalLayout.addWidget(anisotropy)
        self.verticalLayout.addWidget(corrections)
        self.verticalLayout.addWidget(errors)

        if kwargs.get('hide_corrections', False):
            corrections.hide()
        if kwargs.get('hide_fit', False):
            fitting.hide()
        if kwargs.get('hide_generic', False):
            generic.hide()
        if kwargs.get('hide_convolve', False):
            convolve.hide()
        if kwargs.get('hide_rotation', False):
            anisotropy.hide()

    def load_distance_distribution(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        print "load_distance_distribution"
        verbose = kwargs.get('verbose', self.verbose)
        filename = kwargs.get('filename', str(QtGui.QFileDialog.getOpenFileName(self, 'Open File')))
        self.lineEdit.setText(filename)
        ar = np.array(pd.read_csv(filename, sep='\t')).T
        if verbose:
            print "Opening distribution"
            print "Filename: %s" % filename
            print "Shape: %s" % ar.shape
        self.rda = ar[0]
        self.prda = ar[1]
        self.update_model()



