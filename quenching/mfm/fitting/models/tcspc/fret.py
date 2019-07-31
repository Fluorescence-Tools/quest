import numpy as np
from PyQt4 import QtCore, QtGui
from guiqwt.plot import CurveDialog
from guiqwt.builder import make

from mfm import plots
from mfm.fitting.parameter import ParameterWidget, Parameter, AggregatedParameters
import mfm.math
from tcspc import Lifetime, LifetimeWidget
from tcspc import ConvolveWidget, AnisotropyWidget, FRETModel, GenericWidget, CorrectionsWidget
from mfm.fitting.fit import ErrorWidget, FittingWidget


class Gaussians(AggregatedParameters):

    nGaussPoints = 64
    nSigma = 2.0

    @property
    def distribution(self):
        d = list()
        for i in range(len(self)):
            g_min = max(1e-9, self.mean[i] - self.nSigma * self.sigma[i])
            g_max = self.mean[i] + self.nSigma * self.sigma[i]
            bins = np.linspace(g_min, g_max, self.nGaussPoints)
            p = np.exp(-(bins - self.mean[i]) ** 2 / (2 * self.sigma[i] ** 2))
            p = p / np.sum(p)
            p *= self.amplitude[i]
            d.append([p, bins])
        d = np.array(d)
        return d

    @property
    def forster_radius(self):
        return self._R0.value

    @forster_radius.setter
    def forster_radius(self, v):
        self._R0.value = v

    @property
    def tau0(self):
        return self._t0.value

    @tau0.setter
    def tau0(self, v):
        self._t0.value = v

    @property
    def kappa2(self):
        return self._kappa2.value

    @property
    def mean(self):
        try:
            a = np.sqrt(np.array([g.value for g in self._gaussianMeans]) ** 2)
            return a
        except AttributeError:
            return np.array([])

    @property
    def sigma(self):
        try:
            return np.array([g.value for g in self._gaussianSigma])
        except AttributeError:
            return np.array([])

    @property
    def amplitude(self):
        try:
            a = np.sqrt(np.array([g.value for g in self._gaussianAmplitudes]) ** 2)
            a /= a.sum()
            return a
        except AttributeError:
            return np.array([])

    @property
    def donly(self):
        return np.sqrt(self._donly.value ** 2)

    @donly.setter
    def donly(self, v):
        self._donly.value = v

    def finalize(self):
        """
        This updates the values of the fitting parameters
        """
        # update amplitudes (sum of amplitudes is one)
        a = self.amplitude
        for i, g in enumerate(self._gaussianAmplitudes):
            g.value = a[i]
        # update means (only positive distances)
        a = self.mean
        for i, g in enumerate(self._gaussianMeans):
            g.value = a[i]

    def append(self, mean, sigma, x):
        """
        Adds/appends a new Gaussian/normal-distribution

        :param mean: float
            Mean of the new normal distribution
        :param sigma: float
            Sigma/width of the normal distribution
        :param x: float
            Amplitude of the normal distribution
        """
        n = len(self)
        m = Parameter(name='R(%s,%i)' % (self.short, n + 1), value=mean)
        x = Parameter(name='x(%s,%i)' % (self.short, n + 1), value=x)
        s = Parameter(name='s(%s,%i)' % (self.short, n + 1), value=sigma)
        self._gaussianMeans.append(m)
        self._gaussianSigma.append(s)
        self._gaussianAmplitudes.append(x)

    def pop(self):
        """
        Removes the last appended Gaussian/normal-distribution
        """
        self._gaussianMeans.pop()
        self._gaussianSigma.pop()
        self._gaussianAmplitudes.pop()

    def __len__(self):
        return len(self._gaussianAmplitudes)

    def __init__(self, forster_radius=52.0, kappa2=0.667, t0=4.1, donor_only=0.5, no_donly=False, **kwargs):
        """
        This class keeps the necessary parameters to perform a fit with Gaussian/Normal-disitributed
        distances. New distance distributions are added using the methods append.

        :param donors: Lifetime
            The donor-only spectrum in form of a `Lifetime` object.
        :param forster_radius: float
            The Forster-radius of the FRET-pair in Angstrom. By default 52.0 Angstrom (FRET-pair Alexa488/Alexa647)
        :param kappa2: float
            Orientation factor. By default 2./3.
        :param t0: float
            Lifetime of the donor-fluorophore in absence of FRET.
        :param donor_only: float
            Donor-only fraction. The fraction of molecules without acceptor.
        :param no_donly: bool
            If this is True the donor-only fraction is not displayed/present.
        """
        self.donors = kwargs.get('donors', Lifetime(**kwargs))
        self.no_donly = no_donly

        self._gaussianMeans = []
        self._gaussianSigma = []
        self._gaussianAmplitudes = []
        self.short = 'G'

        self._t0 = Parameter(name='t0', value=t0, fixed=True)
        self._R0 = Parameter(name='R0', value=forster_radius, fixed=True)
        self._kappa2 = Parameter(name='k2', value=kappa2, fixed=True, lb=0.0, ub=4.0, bounds_on=False)
        self._name = kwargs.get('name', 'Gaussians')
        if not no_donly:
            self._donly = Parameter(name='DOnly', value=donor_only, fixed=False, lb=0.0, ub=1.0, bounds_on=False)


class GaussianWidget(Gaussians, QtGui.QWidget):
    def __init__(self, donors, parent=None, model=None, short='', forster_radius=52.0, kappa2=0.667, t0=4.1,
                 donly=0.5, no_donly=False, **kwargs):
        hide_donly = kwargs.get('hide_donly', False)

        self.parent = parent
        self.model = model
        self.short = short
        Gaussians.__init__(self, donors=donors, forster_radius=forster_radius, kappa2=kappa2, t0=t0, donor_only=donly,
                           no_donly=no_donly)
        QtGui.QWidget.__init__(self)

        self.layout = QtGui.QVBoxLayout(self)
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        self.layout.setSpacing(0)
        self.layout.setMargin(0)

        self.gb = QtGui.QGroupBox()
        self.layout.addWidget(self.gb)
        self.gb.setTitle("Gaussian-rates")
        self.lh = QtGui.QVBoxLayout()
        self.lh.setSpacing(0)
        self.lh.setMargin(0)
        self.gb.setLayout(self.lh)

        # illustrative plot
        splitter1 = QtGui.QSplitter(QtCore.Qt.Horizontal)

        # Plot of the Gaussians
        win = CurveDialog(edit=False, toolbar=False, wintitle='Distribution', parent=self)
        plot = win.get_plot()
        title = make.label("p", "RDA", (0, -40), "RDA")
        plot.add_item(title)
        self.distance_plot = plot
        self.dist_win = win
        #splitter1.addWidget(plot)

        widget = QtGui.QWidget()
        lh = QtGui.QVBoxLayout()
        lh.setMargin(0)
        lh.setSpacing(0)
        widget.setLayout(lh)

        # TODO: replace this by `FRETWidget`
        self._R0 = ParameterWidget('R0', forster_radius, layout=lh, model=self.model, digits=4, fixed=True,
                                  text='R<sub>0</sub>')
        self._t0 = ParameterWidget('t0', t0, layout=lh, model=self.model, digits=4, fixed=True,
                                  text='&tau;<sub>0</sub>')
        self._kappa2 = ParameterWidget('k2', kappa2, layout=lh, model=self.model, digits=4, fixed=True,
                                      lb=0.0, ub=4.0, hide_bounds=True, bounds_on=False,
                                      text='&kappa;<sup>2</sup>')
        l = QtGui.QHBoxLayout()

        addGaussian = QtGui.QPushButton()
        addGaussian.setText("add")
        l.addWidget(addGaussian)

        removeGaussian = QtGui.QPushButton()
        removeGaussian.setText("del")
        l.addWidget(removeGaussian)

        showGaussian = QtGui.QCheckBox()
        showGaussian.setText("show")
        l.addWidget(showGaussian)
        self.connect(showGaussian, QtCore.SIGNAL("toggled(bool)"), self.onShowGaussian)

        splitter1.addWidget(widget)
        lh.addLayout(l)

        self.lh.addWidget(splitter1)

        l = QtGui.QHBoxLayout()
        self._donly = ParameterWidget('x(D0)', donly, model=self.model, digits=4, bounds_on=False,
                                     lb=0.0, ub=1.0, layout=l, text='x<sup>(D,0)</sup>')
        self._donly.setDisabled(self.no_donly)
        if hide_donly:
            self._donly.hide()

        self.lh.addLayout(l)
        self._gb = list()

        self.gaus_grid_layout = QtGui.QGridLayout()
        self.lh.addLayout(self.gaus_grid_layout)

        self.connect(addGaussian, QtCore.SIGNAL("clicked()"), self.append)
        self.connect(removeGaussian, QtCore.SIGNAL("clicked()"), self.pop)
        # add some initial distance
        self.append(1.0, 50.0, 6.0, False)

    def onShowGaussian(self, v):
        print("onShowGaussian %s" % v)
        if v is True:
            self.dist_win.show()
        else:
            self.dist_win.hide()

    def update(self, *__args):
        QtGui.QWidget.update(self, *__args)
        self.distance_plot.del_all_items()
        for dist in self.distribution:
            y, x = dist
            curve = make.curve(x, y, color="r", linewidth=2)
            self.distance_plot.add_item(curve)
        self.distance_plot.do_autoscale()
        self.model.update()
        #self.model.updatePlots()

    def append(self, x=None, mean=None, sigma=None, update=True):
        x = 1.0 if x is None else x
        m = np.random.normal(self._R0.value, self._R0.value * 0.6, 1)[0] if mean is None else mean
        s = 6.0 if sigma is None else sigma
        gb = QtGui.QGroupBox()
        n_gauss = len(self)
        gb.setTitle('G%i' % (n_gauss + 1))
        l = QtGui.QVBoxLayout()
        l.setSpacing(0)
        l.setMargin(0)
        m = ParameterWidget('R(%s,%i)' % (self.short, n_gauss + 1),
                           value=m,
                           layout=l, model=self.model, digits=4,
                           text='R', update_function=self.update)
        s = ParameterWidget('s(%s,%i)' % (self.short, n_gauss + 1), s, layout=l, model=self.model, digits=4,
                           fixed=True, bounds_on=False, lb=0.0, ub=40.0, hide_bounds=True,
                           text='<b>&sigma;</b>', update_function=self.update)
        x = ParameterWidget('x(%s,%i)' % (self.short, n_gauss + 1), x, layout=l, model=self.model, digits=4,
                           bounds_on=False, text='x', update_function=self.update)
        gb.setLayout(l)
        row = n_gauss / 2
        col = n_gauss % 2
        self.gaus_grid_layout.addWidget(gb, row, col)
        self._gb.append(gb)
        self._gaussianMeans.append(m)
        self._gaussianSigma.append(s)
        self._gaussianAmplitudes.append(x)
        if update:
            self.update()

    def pop(self):
        self._gaussianMeans.pop().close()
        self._gaussianSigma.pop().close()
        self._gaussianAmplitudes.pop().close()
        self._gb.pop().close()
        self.update()

    def clean(self):
        donors = self.donors.clean() if self.donors is not None else None
        if self.no_donly:
            new = Gaussians(donors, self.forster_radius, self.kappa2, self.tau0, no_donly=self.no_donly)
        else:
            new = Gaussians(donors, self.forster_radius, self.kappa2, self.tau0, self.donly)
        new._gaussianAmplitudes = [a.clean() for a in self._gaussianAmplitudes]
        new._gaussianMeans = [a.clean() for a in self._gaussianMeans]
        new._gaussianSigma = [a.clean() for a in self._gaussianSigma]
        return new


class FRETrate(object):

    def __init__(self, **kwargs):
        # TODO refactor gaussians - separate FRET-rate related stuff in separate class (tau0, R0, kappa2)
        pass

    @property
    def forster_radius(self):
        return self.gaussians.forster_radius

    @forster_radius.setter
    def forster_radius(self, v):
        self.gaussians.forster_radius = v

    @property
    def tau0(self):
        return self.gaussians.tau0

    @tau0.setter
    def tau0(self, v):
        self.gaussians.tau0 = v

    @property
    def kappa2(self):
        return self.gaussians.kappa2

    @kappa2.setter
    def kappa2(self, v):
        self.gaussians.kappa2 = v


class GaussianModel(FRETModel, FRETrate):
    """
    This fit model is uses multiple Gaussian/normal distributions to fit the FRET-decay. Here the donor lifetime-
    spectrum as well as the distances may be fitted. In this model it is assumed that each donor-species is fitted
    by the same FRET-rate distribution.

    References
    ----------

    .. [1]  Kalinin, S., and Johansson, L.B., Energy Migration and Transfer Rates
            are Invariant to Modeling the Fluorescence Relaxation by Discrete and Continuous
            Distributions of Lifetimes.
            J. Phys. Chem. B, 108 (2004) 3092-3097.

    """

    name = "FD(A): Gaussian"

    @property
    def forster_radius(self):
        return self.gaussians.forster_radius

    @property
    def kappa2(self):
        return self.gaussians.kappa2

    @property
    def tau0(self):
        return self.gaussians.tau0

    @property
    def donly(self):
        return self.gaussians.donly

    @donly.setter
    def donly(self, v):
        self.gaussians.donly = v

    @property
    def distance_distribution(self):
        dist = self.gaussians.distribution
        return dist

    def append(self, mean, sigma, species_fraction):
        self.gaussians.append(mean, sigma, species_fraction)

    def pop(self):
        return self.gaussians.pop()

    def finalize(self):
        super(FRETModel, self).finalize()
        self.gaussians.finalize()

    def __init__(self, fit, **kwargs):
        FRETModel.__init__(self, fit, **kwargs)
        FRETrate.__init__(self, **kwargs)
        self.gaussians = kwargs.get('gaussians', Gaussians(**kwargs))

    def clean(self, fit):
        generic = self.generic.clean()
        donors = self.donors.clean()
        corrections = self.corrections.clean(fit)
        aniso = self.anisotropy.clean()
        gaussians = self.gaussians.clean()
        con = self.convolve.clean(fit)
        return GaussianModel(fit, gaussians=gaussians, donors=donors,
                             corrections=corrections, generic=generic, convolve=con,
                             anisotropy=aniso)


class GaussianModelWidget(GaussianModel, QtGui.QWidget):
    plot_classes = [(plots.LinePlot, {'d_scalex': 'lin',
                                                  'd_scaley': 'log',
                                                  'r_scalex': 'lin',
                                                  'r_scaley': 'lin',
    }),
                    (plots.SurfacePlot, {})
    ]

    def update(self):
        GaussianModel.update(self)
        self.emit(QtCore.SIGNAL('model_update'))

    def __init__(self, fit, **kwargs):
        QtGui.QWidget.__init__(self)
        self.icon = QtGui.QIcon(":/icons/icons/TCSPC.ico")
        self.fit = fit

        convolve = ConvolveWidget(fit=fit, model=self, **kwargs)
        convolve.hide_curve_convolution(True)
        donors = LifetimeWidget(parent=self, model=self, gtitle='Donor(0)', short='D')
        gaussians = GaussianWidget(donors=donors, parent=self, model=self, short='G', **kwargs)
        anisotropy = AnisotropyWidget(model=self, short='rL', **kwargs)
        generic = GenericWidget(model=self, parent=self, **kwargs)
        fitting = QtGui.QLabel() if kwargs.get('disable_fit', False) else FittingWidget(fit=fit, **kwargs)
        errors = ErrorWidget(fit, **kwargs)
        corrections = CorrectionsWidget(fit, model=self, **kwargs)

        self.layout = QtGui.QVBoxLayout(self)
        self.layout.setSpacing(0)
        self.layout.setMargin(0)
        self.layout.setAlignment(QtCore.Qt.AlignTop)

        self.layout.addWidget(fitting)
        self.layout.addWidget(convolve)
        self.layout.addWidget(generic)
        self.layout.addWidget(donors)
        self.layout.addWidget(gaussians)

        self.layout.addWidget(anisotropy)
        self.layout.addWidget(corrections)
        self.layout.addWidget(errors)

        GaussianModel.__init__(self, fit=fit, gaussians=gaussians, donors=donors,
                               generic=generic, corrections=corrections, convolve=convolve,
                               anisotropy=anisotropy)

        self.y_values = np.zeros(fit.data.y.shape[0])
        self.dt = kwargs.get('dt', fit.data.x[1] - fit.data.x[0])


class FretRate(AggregatedParameters):

    @property
    def distribution(self):
        a = np.array(self.amplitude)
        a /= sum(a)
        d = np.array(self.distances)
        n_rates = len(self)
        d = np.vstack([a, d]).reshape([1, 2, n_rates])
        return d

    @property
    def R0(self):
        return self._R0.value

    @R0.setter
    def R0(self, v):
        self._R0.value = v

    @property
    def tau0(self):
        return self._t0.value

    @tau0.setter
    def tau0(self, v):
        self._t0.value = v

    @property
    def kappa2(self):
        return self._kappa2.value

    @property
    def distances(self):
        try:
            a = np.sqrt(np.array([g.value for g in self._distances]) ** 2)
            for i, g in enumerate(self._distances):
                g.value = a[i]
            return a
        except AttributeError:
            return np.array([])

    @property
    def amplitude(self):
        try:
            a = np.sqrt(np.array([g.value for g in self._amplitudes]) ** 2)
            a /= a.sum()
            for i, g in enumerate(self._amplitudes):
                g.value = a[i]
            return a
        except AttributeError:
            return np.array([])

    @property
    def donly(self):
        return np.sqrt(self._donly.value ** 2)

    @donly.setter
    def donly(self, v):
        self._donly.value = v

    def append(self, distance, x):
        """
        Adds/appends a new FRET-rate

        :param distance: float
            Mean of the new normal distribution
        :param x: float
            Amplitude of the normal distribution
        """
        n = len(self)
        m = Parameter(name='R(%s,%i)' % (self.short, n + 1), value=distance)
        self._distances.append(m)
        self._amplitudes.append(x)

    def pop(self):
        """
        Removes the last appended Gaussian/normal-distribution
        """
        self._distances.pop()
        self._amplitudes.pop()

    def __len__(self):
        return len(self._amplitudes)

    def __init__(self, forster_radius=52.0, kappa2=0.667, t0=4.1, donor_only=0.5, no_donly=False, **kwargs):
        """
        This class keeps the necessary parameters to perform a fit with single (discrete) FRET-rates in form of
        distances. New distance distributions are added using the methods append.

        :param donors: Lifetime
            The donor-only spectrum in form of a `Lifetime` object.
        :param forster_radius: float
            The Forster-radius of the FRET-pair in Angstrom. By default 52.0 Angstrom (FRET-pair Alexa488/Alexa647)
        :param kappa2: float
            Orientation factor. By default 2./3.
        :param t0: float
            Lifetime of the donor-fluorophore in absence of FRET.
        :param donor_only: float
            Donor-only fraction. The fraction of molecules without acceptor.
        :param no_donly: bool
            If this is True the donor-only fraction is not displayed/present.
        """
        self.donors = kwargs.get('donors', Lifetime(**kwargs))
        self._name = kwargs.get('name', 'FRET-distance')
        self.no_donly = no_donly

        self._distances = []
        self._amplitudes = []
        self.short = 'F'

        self._t0 = Parameter(name='t0', value=t0, fixed=True)
        self._R0 = Parameter(name='R0', value=forster_radius, fixed=True)
        self._kappa2 = Parameter(name='k2', value=kappa2, fixed=True, lb=0.0, ub=4.0, bounds_on=False)
        if not no_donly:
            self._donly = Parameter(name='DOnly', value=donor_only, fixed=False, lb=0.0, ub=1.0, bounds_on=False)


class FretRateWidget(FretRate, QtGui.QWidget):
    def __init__(self, donors, parent=None, model=None, short='F', forster_radius=52.0, kappa2=0.667, t0=4.1,
                 donly=0.5, no_donly=False, **kwargs):
        hide_donly = kwargs.get('hide_donly', False)

        self.parent = parent
        self.model = model
        self.short = short
        FretRate.__init__(self, donors=donors, forster_radius=forster_radius, kappa2=kappa2, t0=t0, donor_only=donly,
                           no_donly=no_donly)
        QtGui.QWidget.__init__(self)

        self.layout = QtGui.QVBoxLayout(self)
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        self.layout.setSpacing(0)
        self.layout.setMargin(0)

        self.gb = QtGui.QGroupBox()
        self.layout.addWidget(self.gb)
        self.gb.setTitle("FRET-rates")
        self.lh = QtGui.QVBoxLayout()
        self.lh.setSpacing(0)
        self.lh.setMargin(0)
        self.gb.setLayout(self.lh)

        # illustrative plot
        splitter1 = QtGui.QSplitter(QtCore.Qt.Horizontal)

        widget = QtGui.QWidget()
        lh = QtGui.QVBoxLayout()
        lh.setMargin(0)
        lh.setSpacing(0)
        widget.setLayout(lh)
        # TODO: replace this by `FRETWidget`
        self._R0 = ParameterWidget('R0', forster_radius, layout=lh, model=self.model, digits=1, fixed=True,
                                  text='R<sub>0</sub>')
        self._t0 = ParameterWidget('t0', t0, layout=lh, model=self.model, digits=2, fixed=True,
                                  text='&tau;<sub>0</sub>')
        self._kappa2 = ParameterWidget('k2', kappa2, layout=lh, model=self.model, digits=3, fixed=True,
                                      lb=0.0, ub=4.0, hide_bounds=True, bounds_on=False,
                                      text='&kappa;<sup>2</sup>')
        l = QtGui.QHBoxLayout()

        add_FretRate = QtGui.QPushButton()
        add_FretRate.setText("add")
        l.addWidget(add_FretRate)

        remove_FRETrate = QtGui.QPushButton()
        remove_FRETrate.setText("del")
        l.addWidget(remove_FRETrate)

        splitter1.addWidget(widget)
        lh.addLayout(l)

        self.lh.addWidget(splitter1)

        l = QtGui.QHBoxLayout()
        self._donly = ParameterWidget('x(D0)', donly, model=self.model, digits=2, bounds_on=False,
                                     lb=0.0, ub=1.0, layout=l, text='x<sup>(D,0)</sup>')
        self._donly.setDisabled(self.no_donly)
        if hide_donly:
            self._donly.hide()

        self.lh.addLayout(l)
        self._gb = list()

        self.fret_grid_layout = QtGui.QGridLayout()
        self.lh.addLayout(self.fret_grid_layout)

        self.connect(add_FretRate, QtCore.SIGNAL("clicked()"), self.append)
        self.connect(remove_FRETrate, QtCore.SIGNAL("clicked()"), self.pop)
        # add some initial distance
        self.append(1.0, 50.0, False)

    def update(self, *__args):
        QtGui.QWidget.update(self, *__args)
        self.model.update()
        #self.model.updatePlots()

    def append(self, x=None, mean=None, update=True):
        x = 1.0 if x is None else x
        m = np.random.normal(self._R0.value, self._R0.value * 0.6, 1)[0] if mean is None else mean
        gb = QtGui.QGroupBox()
        n_fret = len(self)
        gb.setTitle('G%i' % (n_fret + 1))
        l = QtGui.QVBoxLayout()
        l.setSpacing(0)
        l.setMargin(0)
        m = ParameterWidget('R(%s,%i)' % (self.short, n_fret + 1), value=m, layout=l, model=self.model, digits=1,
                           text='R', update_function=self.update)
        x = ParameterWidget('x(%s,%i)' % (self.short, n_fret + 1), x, layout=l, model=self.model, digits=2,
                           bounds_on=False, text='x', update_function=self.update)
        gb.setLayout(l)
        row = n_fret / 2
        col = n_fret % 2
        self.fret_grid_layout.addWidget(gb, row, col)
        self._gb.append(gb)

        self._distances.append(m)
        self._amplitudes.append(x)

        if update:
            self.update()

    def pop(self):
        self._amplitudes.pop().close()
        self._distances.pop().close()
        self._gb.pop().close()
        self.update()


class FRETrateModel(FRETModel):
    """
    This fit model is uses multiple discrete FRET rates to fit the Donor-decay. Here the donor lifetime-
    spectrum as well as the distances may be fitted. In this model it is assumed that each donor-species is fitted
    by the same FRET-rate distribution.

    References
    ----------

    .. [1]  Kalinin, S., and Johansson, L.B., Energy Migration and Transfer Rates
            are Invariant to Modeling the Fluorescence Relaxation by Discrete and Continuous
            Distributions of Lifetimes.
            J. Phys. Chem. B, 108 (2004) 3092-3097.

    """

    name = "FD(A): Discrete"

    @property
    def forster_radius(self):
        return self.fret_rates.R0

    @forster_radius.setter
    def forster_radius(self, v):
        self.fret_rates.R0 = v

    @property
    def tau0(self):
        return self.fret_rates.tau0

    @tau0.setter
    def tau0(self, v):
        self.fret_rates.tau0 = v

    @property
    def kappa2(self):
        return self.fret_rates.kappa2

    @kappa2.setter
    def kappa2(self, v):
        self.fret_rates.kappa2 = v

    @property
    def donly(self):
        return self.fret_rates.donly

    @donly.setter
    def donly(self, v):
        self.fret_rates.donly = v

    @property
    def distance_distribution(self):
        dist = self.fret_rates.distribution
        return dist

    def append(self, mean, sigma, species_fraction):
        self.fret_rates.append(mean, sigma, species_fraction)

    def pop(self):
        return self.fret_rates.pop()

    def __init__(self, fit, **kwargs):
        FRETModel.__init__(self, fit, **kwargs)
        self.fret_rates = kwargs.get('fret_rates', FretRate(**kwargs))


class FRETrateModelWidget(FRETrateModel, QtGui.QWidget):
    plot_classes = [(plots.LinePlot, {'d_scalex': 'lin',
                                                  'd_scaley': 'log',
                                                  'r_scalex': 'lin',
                                                  'r_scaley': 'lin',
    }),
                    (plots.SurfacePlot, {})
    ]

    def update(self):
        FRETrateModel.update(self)
        self.emit(QtCore.SIGNAL('model_update'))

    def __init__(self, fit, **kwargs):
        QtGui.QWidget.__init__(self)
        self.icon = QtGui.QIcon(":/icons/icons/TCSPC.ico")
        self.fit = fit

        convolve = ConvolveWidget(fit=fit, model=self, **kwargs)
        convolve.hide_curve_convolution(True)
        donors = LifetimeWidget(parent=self, model=self, gtitle='Donor(0)', short='D')
        fret_rates = FretRateWidget(donors=donors, parent=self, model=self, short='G', **kwargs)
        anisotropy = AnisotropyWidget(model=self, short='rL', **kwargs)
        generic = GenericWidget(model=self, parent=self, **kwargs)
        fitting = QtGui.QLabel() if kwargs.get('disable_fit', False) else FittingWidget(fit=fit, **kwargs)
        errors = ErrorWidget(fit, **kwargs)
        corrections = CorrectionsWidget(fit, model=self, **kwargs)

        self.layout = QtGui.QVBoxLayout(self)
        self.layout.setSpacing(0)
        self.layout.setMargin(0)
        self.layout.setAlignment(QtCore.Qt.AlignTop)

        self.layout.addWidget(fitting)
        self.layout.addWidget(convolve)
        self.layout.addWidget(generic)
        self.layout.addWidget(donors)
        self.layout.addWidget(fret_rates)

        self.layout.addWidget(anisotropy)
        self.layout.addWidget(corrections)
        self.layout.addWidget(errors)

        FRETrateModel.__init__(self, fit=fit, fret_rates=fret_rates, donors=donors,
                               generic=generic, corrections=corrections, convolve=convolve,
                               anisotropy=anisotropy)

        self.y_values = np.zeros(fit.data.y.shape[0])
        self.dt = kwargs.get('dt', fit.data.x[1] - fit.data.x[0])


class WormLikeChainModel(FRETModel):
    name = "FD(A): Worm-like chain"

    n_points = 128

    @property
    def r_da_axis(self):
        return np.linspace(0.001, 0.999, self.n_points)

    @property
    def distance_distribution(self):
        r = self.r_da_axis
        prob = mfm.math.functions.rdf.worm_like_chain(r, self.persistence_length)
        r *= self.chain_length
        dist = np.array([prob, r]).reshape([1, 2, self.n_points])
        return dist

    @property
    def donly(self):
        return self._donly.value

    @property
    def chain_length(self):
        return self._chain_length.value

    @property
    def persistence_length(self):
        return self._persistence_length.value

    @property
    def forster_radius(self):
        return self._R0.value

    @property
    def tau0(self):
        return self._tau0.value

    def __init__(self, fit, **kwargs):
        self.donors = kwargs.get('donors', Lifetime())
        self._R0 = kwargs.get('R0', Parameter(52.0))
        self._tau0 = kwargs.get('tau0', Parameter(4.0))
        self._donly = kwargs.get('donly', Parameter(0.0))
        FRETModel.__init__(self, fit, **kwargs)

    def clean(self, fit):
        pass


class WormLikeChainModelWidget(WormLikeChainModel, QtGui.QWidget):
    plot_classes = [(plots.LinePlot, {'d_scalex': 'lin',
                                                  'd_scaley': 'log',
                                                  'r_scalex': 'lin',
                                                  'r_scaley': 'lin',
    }),
                    (plots.SurfacePlot, {})
    ]

    def update(self):
        WormLikeChainModel.update(self)
        self.emit(QtCore.SIGNAL('model_update'))

    def __init__(self, fit, **kwargs):
        QtGui.QWidget.__init__(self)
        self.fit = fit
        self.icon = QtGui.QIcon(":/icons/icons/TCSPC.ico")
        convolve = ConvolveWidget(fit=fit, model=self, hide_curve_convolution=True, **kwargs)
        donors = LifetimeWidget(parent=self, model=self, gtitle='Donor(0)', short='D')
        generic = GenericWidget(model=self, parent=self, **kwargs)

        if not kwargs.get('disable_fit', False):
            fitting = FittingWidget(fit=fit, **kwargs)
        else:
            fitting = QtGui.QLabel()

        corrections = CorrectionsWidget(fit, model=self, **kwargs)
        WormLikeChainModel.__init__(self, fit=fit, donors=donors,
                                    generic=generic, corrections=corrections, convolve=convolve)

        self.y_values = np.zeros(fit.data.y.shape[0])
        self.dt = fit.data.x[1] - fit.data.x[0]

        self.layout = QtGui.QVBoxLayout(self)
        self.layout.setSpacing(0)
        self.layout.setMargin(0)
        self.layout.setAlignment(QtCore.Qt.AlignTop)

        self.layout.addWidget(fitting)
        self.layout.addWidget(convolve)
        self.layout.addWidget(generic)
        self.layout.addWidget(donors)
        self._donly = ParameterWidget('dOnly', 0.2, layout=self.layout, model=self, digits=2, fixed=True, **kwargs)
        self.layout.addWidget(self._donly)

        self._chain_length = ParameterWidget('length [A]', 10.0, layout=self.layout, model=self, digits=1, fixed=False,
                                            text='l', update_function=self.update)
        self._persistence_length = ParameterWidget('persistence', 0.02, layout=self.layout, model=self, digits=4,
                                                  fixed=False,
                                                  text='k', update_function=self.update)

        self._R0 = ParameterWidget('R0 [A]', 52.0, layout=self.layout, model=self, digits=1, fixed=True,
                                  text='R0[A]', update_function=self.update)
        self._tau0 = ParameterWidget('tau0', 4.1, layout=self.layout, model=self, digits=2, fixed=True,
                                    text='tau0', update_function=self.update)

        self.layout.addWidget(self._chain_length)
        self.layout.addWidget(self._persistence_length)
        self.layout.addWidget(self._R0)
        self.layout.addWidget(self._tau0)
        self.layout.addWidget(corrections)

        if kwargs.get('hide_corrections', False):
            corrections.hide()
        if kwargs.get('hide_fit', False):
            fitting.hide()
        if kwargs.get('hide_generic', False):
            generic.hide()
        if kwargs.get('hide_convolve', False):
            convolve.hide()
        if kwargs.get('hide_donor', False):
            donors.hide()


class GaussianChainModel(FRETModel):

    name = "FD(A): Gaussian-chain"

    n_points = 512

    @property
    def r_da_axis(self):
        r_max = 150.0
        n_points = self.n_points
        r = np.linspace(5, r_max, n_points)
        return r

    @property
    def distance_distribution(self):
        """
        The rate spectrum of the donor-decay in presence of an acceptor
        """
        r = self.r_da_axis
        prob = mfm.math.functions.rdf.gaussian_chain(r, self.chain_bond_length, self.n_bonds)
        dist = np.array([prob, r])
        dist = dist.reshape([1, 2, self.n_points])
        return dist

    @property
    def donly(self):
        """
        The fraction of donor-only (no-FRET)
        """
        return self._donly.value

    @property
    def n_bonds(self):
        """
        The number of bonds
        """
        return self._n_bonds.value

    @property
    def chain_bond_length(self):
        """
        The length per bond
        """
        return self._chain_bond_length.value

    @property
    def forster_radius(self):
        """
        The Forster-radius of the dye-pair
        """
        return self._R0.value

    @property
    def tau0(self):
        return self._tau0.value

    def __init__(self, fit, **kwargs):
        FRETModel.__init__(self, fit, **kwargs)
        self.donors = kwargs.get('donors', Lifetime())
        self._R0 = kwargs.get('R0', Parameter(52.0))
        self._tau0 = kwargs.get('tau0', Parameter(4.1))

    def clean(self, fit):
        pass


class GaussianChainModelWidget(GaussianChainModel, QtGui.QWidget):

    plot_classes = [(plots.LinePlot, {'d_scalex': 'lin',
                                                  'd_scaley': 'log',
                                                  'r_scalex': 'lin',
                                                  'r_scaley': 'lin',
    }),
                    (plots.SurfacePlot, {})
    ]

    def update(self):
        GaussianChainModel.update(self)
        self.emit(QtCore.SIGNAL('model_update'))

    def __init__(self, fit, **kwargs):
        QtGui.QWidget.__init__(self)
        self.fit = fit
        self.icon = QtGui.QIcon(":/icons/icons/TCSPC.ico")
        self.layout = QtGui.QVBoxLayout(self)

        convolve = ConvolveWidget(fit=fit, model=self, hide_curve_convolution=True, **kwargs)
        donors = LifetimeWidget(parent=self, model=self, gtitle='Donor(0)', short='D')
        generic = GenericWidget(model=self, parent=self, **kwargs)

        if not kwargs.get('disable_fit', False):
            fitting = FittingWidget(fit=fit, **kwargs)
        else:
            fitting = QtGui.QLabel()

        self.layout.addWidget(fitting)
        self.layout.addWidget(convolve)
        self.layout.addWidget(generic)
        self.layout.addWidget(donors)

        corrections = CorrectionsWidget(fit, model=self, **kwargs)

        GaussianChainModel.__init__(self, fit=fit, donors=donors, generic=generic,
                                    corrections=corrections, convolve=convolve)

        self.y_values = np.zeros(fit.data.y.shape[0])
        self.dt = fit.data.x[1] - fit.data.x[0]

        self.layout.setSpacing(0)
        self.layout.setMargin(0)
        self.layout.setAlignment(QtCore.Qt.AlignTop)

        self._donly = ParameterWidget('dOnly', 0.2, layout=self.layout, model=self, digits=2, fixed=True, **kwargs)
        self.layout.addWidget(self._donly)

        self._n_bonds = ParameterWidget('n', 300, layout=self.layout, model=self, digits=1, fixed=False,
                                       text='n bond', update_function=self.update)
        self._chain_bond_length = ParameterWidget('l', 3.5, layout=self.layout, model=self, digits=4,
                                                  fixed=False, text='bond length', update_function=self.update,
                                                  tooltip='The length of a bond')

        self._R0 = ParameterWidget('R0 [A]', 52.0, layout=self.layout, model=self, digits=1, fixed=True,
                                  text='R0[A]', update_function=self.update)
        self._tau0 = ParameterWidget('tau0', 4.1, layout=self.layout, model=self, digits=2, fixed=True,
                                    text='tau0', update_function=self.update)

        self.layout.addWidget(self._chain_bond_length)
        self.layout.addWidget(self._n_bonds)
        self.layout.addWidget(self._R0)
        self.layout.addWidget(self._tau0)
        self.layout.addWidget(corrections)

        print kwargs

        if kwargs.get('hide_corrections', False):
            corrections.hide()
        if kwargs.get('hide_fit', False):
            fitting.hide()
        if kwargs.get('hide_generic', False):
            generic.hide()
        if kwargs.get('hide_convolve', False):
            convolve.hide()
        if kwargs.get('hide_donor', False):
            print "GC-hid d"
            donors.hide()