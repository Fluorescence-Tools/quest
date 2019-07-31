from PyQt4 import QtGui, uic, QtCore
import numpy as np
from mfm.fitting.parameter import ParameterWidget, AggregatedParameters

from .tcspc import GenericWidget, CorrectionsWidget
from mfm.fitting.models.tcspc.fret import GaussianWidget
from .tcspc import LifetimeWidget, LifetimeModel
from .tcspc import ConvolveWidget, AnisotropyWidget
from mfm import FittingWidget, ErrorWidget, plots
from mfm.fluorescence import gaussian2rates, tcspc


class PDDEM(AggregatedParameters):

    @property
    def pxA(self):
        """
        :return: float
            Excitation probability of fluorphore A
        """
        return self._pxA.value

    @property
    def pxB(self):
        """
        :return: float
            Excitation probability of fluorphore B
        """
        return self._pxB.value

    @property
    def px(self):
        """
        :return: numpy-array
            Exciation probabilities of flurophore (A, B)
        """
        return np.array([self.pxA, self.pxB], dtype=np.float64)

    @property
    def pmA(self):
        """
        :return: float
            Emission probability of flurophore A
        """
        return self._pmA.value

    @property
    def pmB(self):
        """
        :return: float
            Emission probability of flurophore B
        """
        return self._pmB.value

    @property
    def pm(self):
        """
        :return: array
            Emission probability of flurophore (A, B)
        """
        return np.array([self.pmA, self.pmB], dtype=np.float64)

    @property
    def pureA(self):
        """
        :return: float
            fraction of decay A in total decay
        """
        return self._pA.value

    @property
    def pureB(self):
        """
        :return: float
            fraction of decay B in total decay
        """
        return self._pB.value

    @property
    def pureAB(self):
        """
        :return: array
            fraction of decay (A, B) in total decay
        """
        return np.array([self.pureA, self.pureB], dtype=np.float64)

    @property
    def fAB(self):
        """
        :return: float
            probability of energy-transfer from A to B
        """
        return self._fAB.value

    @property
    def fBA(self):
        """
        :return: float
            probability of energy-transfer from B to A
        """
        return self._fBA.value

    @property
    def fABBA(self):
        """
        :return: array
            probability of energy-transfer from (A to B), (B to A)
        """
        return np.array([self.fAB, self.fBA], dtype=np.float64)



class PDDEMWidget(QtGui.QWidget, PDDEM):
    def __init__(self, parent=None, model=None, short='', forster_radius=52.0, kappa2=0.667, t0=4.1):
        QtGui.QWidget.__init__(self)
        PDDEM.__init__(self)
        uic.loadUi('mfm/ui/fitting/models/tcspc/pddem.ui', self)
        self.model = model
        self.parent = parent
        self.A = LifetimeWidget(gtitle='Lifetimes-A', model=model, short='A')
        self.B = LifetimeWidget(gtitle='Lifetimes-B', model=model, short='B')
        self.gaussians = GaussianWidget(donors=None, parent=self, model=model, short='G', no_donly=True)

        l = QtGui.QHBoxLayout()
        self._fAB = ParameterWidget('A>B', 1.0, layout=l, model=self.parent, digits=2, fixed=True)
        self._fBA = ParameterWidget('B>A', 0.0, layout=l, model=self.parent, digits=2, fixed=True)
        self.verticalLayout_3.addLayout(l)

        l = QtGui.QHBoxLayout()
        self._pA = ParameterWidget('pureA', 0.0, layout=l, model=self.parent, digits=2, fixed=True)
        self._pB = ParameterWidget('pureB', 0.0, layout=l, model=self.parent, digits=2, fixed=True)
        self.verticalLayout_3.addLayout(l)

        l = QtGui.QHBoxLayout()
        self._pxA = ParameterWidget('xA', 0.98, layout=l, model=self.parent, digits=2, fixed=True, text='Ex<sub>A</sub>')
        self._pxB = ParameterWidget('xB', 0.02, layout=l, model=self.parent, digits=2, fixed=True, text='Ex<sub>B</sub>')
        self.verticalLayout_3.addLayout(l)

        l = QtGui.QHBoxLayout()
        self._pmA = ParameterWidget('mA', 0.02, layout=l, model=self.parent, digits=2, fixed=True, text='Em<sub>A</sub>')
        self._pmB = ParameterWidget('mB', 0.98, layout=l, model=self.parent, digits=2, fixed=True, text='Em<sub>B</sub>')
        self.verticalLayout_3.addLayout(l)

        self.verticalLayout_3.addWidget(self.A)
        self.verticalLayout_3.addWidget(self.B)

        self.verticalLayout.addWidget(self.gaussians)

    def clean(self):
        new = PDDEM()
        new.gaussians = self.gaussians.clean()

        new.A = self.A.clean()
        new.B = self.B.clean()

        new._fAB = self._fAB.clean()
        new._fBA = self._fBA.clean()

        new._pxA = self._pxA.clean()
        new._pxB = self._pxB.clean()

        new._pmA = self._pmA.clean()
        new._pmB = self._pmB.clean()

        new._pB = self._pB.clean()
        new._pA = self._pA.clean()

        return new


class PDDEMModel(LifetimeModel):
    """
    Kalinin, S., and Johansson, L.B.
    Energy Migration and Transfer Rates are Invariant to Modeling the
    Fluorescence Relaxation by Discrete and Continuous Distributions of
    Lifetimes.
    J. Phys. Chem. B, 108 (2004) 3092-3097.
    """

    name = "PDDEM"

    def __init__(self, fit, **kwargs):
        LifetimeModel.__init__(self, fit, **kwargs)
        self.pddem = kwargs.get('pddem', [])

    @property
    def rate_spectrum(self):
        if len(self.pddem.gaussians) > 0:
            gaussians = self.pddem.gaussians
            return gaussian2rates(gaussians.mean, gaussians.sigma, gaussians.amplitude,
                                  gaussians.tau0, gaussians.kappa2, gaussians.forster_radius)
        else:
            return np.array([1.0, 1.0], np.float64)

    @property
    def lifetime_spectrum(self):
        decayA = self.pddem.A.lifetime_spectrum
        decayB = self.pddem.B.lifetime_spectrum
        rate_spectrum = self.rate_spectrum
        if len(decayA) > 0 and len(decayB) > 0 and len(rate_spectrum) > 0:
            p, rates = rate_spectrum[::2], rate_spectrum[1::2]
            decays = []
            for i, r in enumerate(rates):
                tmp = tcspc.pddem(decayA, decayB, self.pddem.fABBA * r,
                                  self.pddem.px, self.pddem.pm, self.pddem.pureAB)
                tmp[0::2] *= p[i]
                decays.append(tmp)
            ds = np.concatenate(decays)
            return ds
        else:
            return np.array([0.0, 0.0], dtype=np.float64)

    def clean(self, fit):
        generic = self.generic.clean()
        corrections = self.corrections.clean(fit)
        pddem = self.pddem.clean()
        convolve = self.convolve.clean(fit)
        aniso = self.anisotropy.clean()
        return PDDEMModel(self.fit, pddem=pddem, corrections=corrections, generic=generic, convolve=convolve,
                          anisotropy=aniso)


class PDDEMModelWidget(PDDEMModel, QtGui.QWidget):

    plot_classes = [(plots.LinePlot, {'d_scalex': 'lin',
                                                  'd_scaley': 'log',
                                                  'r_scalex': 'lin',
                                                  'r_scaley': 'lin',
    }),
                    (plots.SurfacePlot, {})
    ]

    def __init__(self, fit, **kwargs):
        QtGui.QWidget.__init__(self)
        self.fit = fit

        fitting = FittingWidget(fit=fit, **kwargs)
        convolve = ConvolveWidget(fit=fit, model=self, dt=fit.data.dt, **kwargs)
        convolve.hide_curve_convolution(True)
        corrections = CorrectionsWidget(fit, self, **kwargs)
        generic = GenericWidget(model=self, parent=self, **kwargs)
        anisotropy = AnisotropyWidget(model=self, short='rL', **kwargs)
        pddem = PDDEMWidget(parent=self, model=self, short='P')
        error_widget = ErrorWidget(fit, **kwargs)

        PDDEMModel.__init__(self, fit=fit, pddem=pddem,
                            generic=generic, corrections=corrections, convolve=convolve,
                            anisotropy=anisotropy)

        self.layout = QtGui.QVBoxLayout(self)
        self.layout.setSpacing(0)
        self.layout.setMargin(0)
        self.layout.setAlignment(QtCore.Qt.AlignTop)

        self.layout.addWidget(fitting)
        self.layout.addWidget(convolve)
        self.layout.addWidget(generic)
        self.layout.addWidget(pddem)
        self.layout.addWidget(anisotropy)

        self.layout.addWidget(corrections)
        self.layout.addWidget(error_widget)