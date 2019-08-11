import numpy as np
from PyQt4 import QtGui, QtCore, uic

import mfm
from mfm import plots
from mfm.fitting.models import Model
import mfm.fluorescence
import mfm.math
from mfm.fitting.optimization import solve_nnls, solve_richardson_lucy, maxent
from .tcspc import LifetimeModel
import mfm.fluorescence.general


class Phasor(object):

    @property
    def phasor_siwD0(self):
        """Phasor plot si(w)
        The phasor approach to fluorescence lifetime page 236
        :return:
        """
        return mfm.fluorescence.general.phasor_siw(self.fd0, self.phasor_n, self.phasor_omega, self.times)

    @property
    def phasor_giwD0(self):
        """Phasor plot gi(w)
        The phasor approach to fluorescence lifetime page 236
        :return:
        """
        return mfm.fluorescence.general.phasor_giw(self.fd0, self.phasor_n, self.phasor_omega, self.times)

    @property
    def phasor_siwDA(self):
        """Phasor plot si(w)
        The phasor approach to fluorescence lifetime page 236
        :return:
        """
        return mfm.fluorescence.general.phasor_siw(self.fda, self.phasor_n, self.phasor_omega, self.times)

    @property
    def phasor_giwDA(self):
        """Phasor plot gi(w)
        The phasor approach to fluorescence lifetime page 236
        :return:
        """
        return mfm.fluorescence.general.phasor_giw(self.fda, self.phasor_n, self.phasor_omega, self.times)

    @property
    def phasor_siwE(self):
        """Phasor plot si(w)
        The phasor approach to fluorescence lifetime page 236
        :return:
        """
        return mfm.fluorescence.general.phasor_siw(self.et, self.phasor_n, self.phasor_omega, self.times)

    @property
    def phasor_giwE(self):
        """Phasor plot gi(w)
        The phasor approach to fluorescence lifetime page 236
        :return:
        """
        return mfm.fluorescence.general.phasor_giw(self.et, self.phasor_n, self.phasor_omega, self.times)

    def set_fd0_fda_et(self, fd0, fda, et):
        self.fd0 = fd0
        self.fda = fda
        self.et = et

    def __init__(self, **kwargs):
        self._phasor_n = kwargs.get('phasor_n', 1.0)
        self._phasor_omega = kwargs.get('phasor_omega', 31.25)


class PhasorWidget(Phasor, QtGui.QWidget):

    @property
    def phasor_omega(self):
        return float(self.doubleSpinBox_12.value()) / 1000.0 * np.pi * 2.0

    @phasor_omega.setter
    def phasor_omega(self, v):
        self.doubleSpinBox_12.setValue(v)

    @property
    def phasor_n(self):
        return int(self.spinBox_5.value())

    @phasor_n.setter
    def phasor_n(self, v):
        self.spinBox_5.setValue(v)

    @property
    def phasor_omega(self):
        return self._phasor_omega / 1000.0 * np.pi * 2.0

    @phasor_omega.setter
    def phasor_omega(self, v):
        self._phasor_omega = v

    @property
    def phasor_n(self):
        return self._phasor_n

    @phasor_n.setter
    def phasor_n(self, v):
        self._phasor_n = float(v)

    def __init__(self, **kwargs):
        QtGui.QWidget.__init__(self)
        Phasor.__init__(self, **kwargs)
        uic.loadUi('mfm/ui/fitting/models/tcspc/phasor_widget.ui', self)
        #self.connect(self.actionUpdate_phasor, QtCore.SIGNAL('triggered()'), self.onUpdatePhasor)
        #self.connect(self.actionUpdate_phasor, QtCore.SIGNAL('triggered()'), self.onUpdatePhasor)

    def onUpdatePhasor(self):
        self.lineEdit.setText(str(self.phasor_siwD0))
        self.lineEdit_3.setText(str(self.phasor_giwD0))

        self.lineEdit_4.setText(str(self.phasor_siwDA))
        self.lineEdit_5.setText(str(self.phasor_giwDA))

        self.lineEdit_6.setText(str(self.phasor_siwE))
        self.lineEdit_7.setText(str(self.phasor_giwE))


class LCurve(object):

    def __init__(self, **kwargs):
        self._l_curve_start = 0.0
        self._l_curve_stop = 1.0
        self._l_curve_steps = 512
        self._l_curve_chi2 = np.array([1.0], dtype=np.float64)
        self._l_curve_sol_norm = np.array([1.0], dtype=np.float64)


    @property
    def l_curve_start(self):
        """
        The smallest regularization value
        """
        return self._l_curve_start

    def l_curve_start(self, v):
        """
        The smallest regularization value
        """
        self._l_curve_start = v

    @property
    def l_curve_stop(self):
        """
        The largest regularization value
        """
        return self._l_curve_stop

    @l_curve_stop.setter
    def l_curve_stop(self, v):
        self._l_curve_stop = v

    @property
    def l_curve_steps(self):
        """
        The number of points of the l-curve
        """
        return self._l_curve_steps

    @l_curve_steps.setter
    def l_curve_steps(self, v):
        self._l_curve_steps = v

    @property
    def l_curve_reg(self):
        """
        The regulariation parameters of the calculated l-curve
        """
        return np.linspace(self.l_curve_start, self.l_curve_stop, self.l_curve_steps)

    @property
    def l_curve_chi2(self):
        """
        The chi2r of the l-curve
        """
        return self._l_curve_chi2

    @property
    def l_curve_solution_norm(self):
        """
        The solution norm calculated for a set of regularization parameters
        """
        return self._l_curve_sol_norm

    def update_l_curve(self):
        """
        Calling this method recalcultes the values of the l-curve
        """
        regus = self.l_curve_reg
        chi2s = np.zeros_like(regus)
        sol_norm = np.zeros_like(regus)
        rda = self.r_DA
        if isinstance(self.fda_model, LifetimeModel):
            d0_lifetime_spectrum = self.fd0_model.lifetime_spectrum
            for i, r in enumerate(regus):
                prda = self.get_pRDA(regularization_factor=r)
                lifetime_spectrum = self.get_lifetime_spectrum(rda, prda, d0_lifetime_spectrum)
                chi2s[i] = self.get_chi2r(lifetime_spectrum)
                sol_norm[i] = np.linalg.norm(prda)
        self._l_curve_chi2 = chi2s
        self._l_curve_sol_norm = sol_norm
        self.update_plots()


class DistanceDistribution(object):

    @property
    def r_DA_min(self):
        return self._r_DA_min

    @r_DA_min.setter
    def r_DA_min(self, v):
        self._r_DA_min = v

    @property
    def r_DA_max(self):
        return self._r_DA_max

    @r_DA_max.setter
    def r_DA_max(self, v):
        self._r_DA_max = v

    @property
    def r_DA_npoints(self):
        return self._r_DA_npoints

    @r_DA_npoints.setter
    def r_DA_npoints(self, v):
        self._r_DA_npoints = v

    @property
    def kappa2(self):
        return self._kappa2

    @kappa2.setter
    def kappa2(self, v):
        self._kappa2 = v

    @property
    def tau0(self):
        return self._tau0

    @tau0.setter
    def tau0(self, v):
        self._tau0 = v

    @property
    def R0(self):
        return self._R0

    @R0.setter
    def R0(self, v):
        self._R0 = v

    @property
    def p_rDA(self):
        """
        The the probability distribution of the distances
        """
        return self._p_rDA

    @p_rDA.setter
    def p_rDA(self, v):
        self._p_rDA = v

    @property
    def r_DA(self):
        """
        The array of donor acceptor distances
        """
        return np.linspace(self.r_DA_min, self.r_DA_max, self.r_DA_npoints)

    @r_DA.setter
    def r_DA(self, v):
        self._r_DA = v

    def __init__(self, **kwargs):
        self._r_DA_min = 0.0
        self._r_DA_max = 150.0
        self._r_DA_npoints = 512
        self._p_rDA = np.ones(512)
        self._kappa2 = 0.667
        self._tau0 = 4.0
        self._R0 = 52.0


class EtModelFree(Model, Phasor, LCurve, DistanceDistribution):
    """
    This model deconvolutes the distance distribution given the two fits of the type
    :py:class:`~.mfm.models.tcspc.tcspc.LifetimeModel`.
    """

    name = "Et-Model free"

    def __init__(self, fit, **kwargs):
        Model.__init__(self, fit=fit)
        Phasor.__init__(self)
        LCurve.__init__(self, **kwargs)
        DistanceDistribution.__init__(self)

        self.verbose = kwargs.get('verbose', mfm.verbose)

        self.fd0_index = 0
        self.fda_index = 0

        self._t_points = 4096
        self._t_min = 0.0
        self._t_max = 200.0
        self._t_mode = 'log'
        self._inversion_method = 'nnls'
        self._regularization_factor = 0.01

        self._chi2r = 1000.0
        self.t_matrix = None


    @property
    def fda_model(self):
        if 0 <= self.fda_index < len(self.fits):
            return self.fits[self.fda_index].model
        else:
            return None

    @property
    def fd0_model(self):
        if 0 <= self.fd0_index < len(self.fits):
            return self.fits[self.fd0_index].model
        else:
            return None

    @property
    def data(self):
        return self.fda.data

    @property
    def t_points(self):
        """
        The number of points used in the calculation of the time axis
        """
        return self._t_points

    @t_points.setter
    def t_points(self, v):
        self._t_points = v

    @property
    def t_min(self):
        """
        The smallest time in the time-axis
        """
        return self._t_min

    @t_min.setter
    def t_min(self, v):
        self._t_min = v

    @property
    def t_max(self):
        """
        The largest time of the time-axis
        """
        return self._t_max

    @t_max.setter
    def t_max(self, v):
        self._t_max = v

    @property
    def t_mode(self):
        """
        Calculation mode of the time-axis (either 'lin' or 'log')
        """
        return self._t_mode

    @t_max.setter
    def t_mode(self, v):
        self._t_mode = v

    @property
    def inversion_method(self):
        return self._inversion_method

    @inversion_method.setter
    def inversion_method(self, v):
        self._inversion_method = v

    @property
    def regularization_factor(self):
        return self._regularization_factor

    @regularization_factor.setter
    def regularization_factor(self, v):
        self._regularization_factor = v

    @property
    def chi2r(self):
        return self._chi2r

    @chi2r.setter
    def chi2r(self, v):
        self._chi2r = float(v)

    @property
    def times(self):
        if self.t_mode == 'log':
            t_min = np.log10(max(self.t_min, 0.001))
            t_max = np.log10(self.t_max)
            ts = np.logspace(t_min, t_max, self.t_points)
        elif self.t_mode == 'lin':
            ts = np.linspace(self.t_min, self.t_max, self.t_points)
        return ts

    @property
    def fd0(self):
        if len(self.fits) > 0 and self.fd0_model is not None:
            return self.fd0_model.decay(self.times)
        else:
            return np.ones(self.t_matrix.shape[1], dtype=np.float64)

    @property
    def fda(self):
        if len(self.fits) > 0 and self.fda_model is not None:
            return self.fda_model.decay(self.times)
        else:
            return np.ones(self.t_matrix.shape[1], dtype=np.float64)

    @property
    def et(self):
        return self.fda / self.fd0

    @property
    def fits(self):
        fits = [f for f in mfm.fits if isinstance(f.model, LifetimeModel) and not isinstance(f.model, EtModelFree)]
        return fits

    @fits.setter
    def fits(self, v):
        pass

    @property
    def lifetime_spectrum(self):
        """
        Lifetime-spectrum of the calculated distance-distribution and the lifetime-spectrum
        of the donor reference sample
        """
        rda = self.r_DA
        p_rda = self.p_rDA
        if isinstance(self.fd0_model, LifetimeModel):
            d0_spectrum = self.fd0_model.lifetime_spectrum
            return self.get_lifetime_spectrum(rda, p_rda, d0_spectrum)
        else:
            return np.array([1.0, 1.0], dtype=np.float64)

    def get_pRDA(self, **kwargs):
        """
        This re-calculates the probability distribution of the distances

        :param kwargs: optional parameters
            If an optional parameters is not provided the parameters value is taken form the attribute of the
            class-instance

            inversion_method: string
                The applied inversion method: either 'nnls' for non-negativity least squares with Thikonov
                regularization or 'lstsq' for truncated singulart value decomposition
            regularization_factor: float
                The used regularization factor

        """
        inversion_method = kwargs.get('inversion_method', self.inversion_method)
        regularization_factor = kwargs.get('regularization_factor', self.regularization_factor)
        if self.t_matrix is not None:
            n, m = self.t_matrix.shape
            if n > 2 and m > 2:
                if inversion_method == 'nnls':
                    x = solve_nnls(self.t_matrix, self.et, regularization_factor)
                    #x = mfm.math.linalg.solve_richardson_lucy(self.t_matrix, self.et, x0=x, max_iter=100)
                elif inversion_method == 'mem':
                    x = maxent(self.t_matrix.T, self.et, regularization_factor)
                elif inversion_method == 'lstsq':
                    x = np.linalg.lstsq(self.t_matrix.T, self.et, self.regularization_factor)[0]
                elif inversion_method == 'richardson-lucy':
                    x = solve_richardson_lucy(self.t_matrix, self.et, 1000)
        return x

    def weighted_residuals(self, **kwargs):
        """
        The current weighted residuals given a lifetime distribution
        :param data:
        :return:
        """
        lifetime_spectrum = kwargs.get('lifetime_spectrum', self.lifetime_spectrum)
        if self.fda_model is not None:
            self.fda_model.update_model(lifetime_spectrum=lifetime_spectrum, verbose=self.verbose, autoscale=True)
            wres = self.fda_model.weighted_residuals()
            self.fda_model.update_model()
            return wres

    def get_lifetime_spectrum(self, rDA, pRDA, d0_lifetime_spectrum):
        """
        Get the lifetime-spectrum of a distance-distribution given the the lifetime-spectrum a donor reference sample

        :param rDA: numpy-array
            A 1D array of distances
        :param pRDA: numpy-array
            A 1D array of probabilities
        :param d0_lifetime_spectrum: numpy-array
            A interleaved lifetime-spectrum

        :return: numpy-array
            A interleaved lifetime-spectrum given the distance distribution
        """
        lt_d = d0_lifetime_spectrum
        xd, ld = lt_d.reshape((lt_d.shape[0]/2, 2)).T

        ekFRET = np.exp(mfm.fluorescence.general.distance2rate(rDA, self.kappa2, self.tau0, self.R0))
        ekD = np.exp(1. / ld)

        rates = np.log(np.einsum('i,j', ekD, ekFRET))
        species = np.einsum('i,j', xd, pRDA)

        lifetimes = 1. / rates
        re = np.vstack([species.ravel(), lifetimes.ravel()]).ravel(-1)
        return re

    def update(self, update_matrix=True):
        if update_matrix:
            self.update_matrix()
            self.p_rDA = self.get_pRDA()
            self.chi2r = self.get_chi2r(self.lifetime_spectrum)
        if isinstance(self.fda_model, Model):
            self.update_widgets()
            self.update_plots()

    def update_matrix(self):
        self.t_matrix, x = mfm.fluorescence.calc_transfer_matrix(self.times, r_DA=self.r_DA, kappa2=self.kappa2,
                                                                tau0=self.tau0, R0=self.R0)

    def get_chi2r(self, lifetime_spectrum):
        if self.fda_model is not None:
            self.fda_model.update_model(lifetime_spectrum=lifetime_spectrum, verbose=self.verbose, autoscale=True)
            chi2r = self.fda_model.chi2r(self.fda_model.fit.data)
            self.fda_model.update_model()
            return chi2r


class EtModelFreeWidget(EtModelFree, QtGui.QWidget):
    plot_classes = [(plots.GlobalEt, {'f_scalex': 'log',
                                                  'f_scaley': 'lin',
                                                  'e_scalex': 'log',
                                                  'e_scaley': 'lin'}),
                    (plots.SurfacePlot, {})
    ]

    def __init__(self, fit, **kwargs):
        # TODO, refactor L-Curve (make L-Curve widget)
        # TODO, refactor Phasor (make Phasor widget)
        QtGui.QWidget.__init__(self)
        self.icon = QtGui.QIcon(":/icons/icons/TCSPC.ico")

        EtModelFree.__init__(self, fit)
        uic.loadUi('mfm/ui/fitting/models/tcspc/et_model_free.ui', self)
        phasor = PhasorWidget()
        self.verticalLayout_2.addWidget(phasor)
        self.fits = []
        self.connect(self.actionUpdate_decay_list, QtCore.SIGNAL('triggered()'), self.onUpdateDecays)
        self.connect(self.actionUpdate_plots, QtCore.SIGNAL('triggered()'), self.update)
        self.connect(self.actionUpdate_regularization, QtCore.SIGNAL('triggered()'), self.update)
        self.connect(self.actionUpdate_L_Curve, QtCore.SIGNAL('triggered()'), self.update_l_curve)
        self.connect(self, QtCore.SIGNAL('add_fit'), self.onUpdateDecays)

    @property
    def l_curve_start(self):
        return float(self.doubleSpinBox_5.value())

    @l_curve_start.setter
    def l_curve_start(self, v):
        self.doubleSpinBox_5.setValue(v)

    @property
    def l_curve_stop(self):
        return float(self.doubleSpinBox_9.value())

    @l_curve_stop.setter
    def l_curve_stop(self, v):
        self.doubleSpinBox_9.setValue(v)

    @property
    def l_curve_steps(self):
        return int(self.spinBox_6.value())

    @l_curve_steps.setter
    def l_curve_steps(self, v):
        self.spinBox_6.setValue(int(v))

    @property
    def entropy_weight(self):
        return float(self.doubleSpinBox_11.value())


    @property
    def t_points(self):
        return int(self.spinBox.value())

    @t_points.setter
    def t_points(self, v):
        self.spinBox.setValue(int(v))

    @property
    def t_min(self):
        return float(self.doubleSpinBox_3.value())

    @t_min.setter
    def t_min(self, v):
        self.doubleSpinBox_3.setValue(v)

    @property
    def t_max(self):
        return float(self.doubleSpinBox_4.value())

    @t_max.setter
    def t_max(self, v):
        self.doubleSpinBox_4.setValue(v)

    @property
    def t_mode(self):
        if self.radioButton_3.isChecked():
            return 'lin'
        else:
            return 'log'

    @t_mode.setter
    def t_mode(self, v):
        if v == 'lin':
            self.radioButton_3.setChecked(True)
        else:
            self.radioButton_4.setChecked(True)

    @property
    def inversion_method(self):
        if self.radioButton_2.isChecked():
            return 'nnls'
        elif self.radioButton_4.isChecked():
            return 'mem'
        else:
            return 'lstsq'

    @property
    def regularization_factor(self):
        return float(self.doubleSpinBox_10.value())

    @regularization_factor.setter
    def regularization_factor(self, v):
        self.doubleSpinBox_10.setValue(float(v))

    @property
    def r_DA_min(self):
        return float(self.doubleSpinBox.value())

    @r_DA_min.setter
    def r_DA_min(self, v):
        self.doubleSpinBox.setValue(v)

    @property
    def r_DA_max(self):
        return float(self.doubleSpinBox_2.value())

    @r_DA_max.setter
    def r_DA_max(self, v):
        self.doubleSpinBox_2.setValue(v)

    @property
    def r_DA_npoints(self):
        return float(self.spinBox_2.value())

    @r_DA_npoints.setter
    def r_DA_npoints(self, v):
        self.spinBox_2.setValue(v)

    @property
    def kappa2(self):
        return float(self.doubleSpinBox_6.value())

    @kappa2.setter
    def kappa2(self, v):
        self.doubleSpinBox_6.setValue(v)

    @property
    def tau0(self):
        return float(self.doubleSpinBox_8.value())

    @tau0.setter
    def tau0(self, v):
        self.doubleSpinBox_8.setValue(v)

    @property
    def R0(self):
        return float(self.doubleSpinBox_7.value())

    @R0.setter
    def R0(self, v):
        self.doubleSpinBox_7.setValue(v)

    @property
    def fda_index(self):
        return int(self.comboBox_2.currentIndex())

    @fda_index.setter
    def fda_index(self, v):
        pass

    @property
    def fd0_index(self):
        return int(self.comboBox.currentIndex())

    @fd0_index.setter
    def fd0_index(self, v):
        pass

    @property
    def chi2r(self):
        return float(self.lineEdit_2.text())

    @chi2r.setter
    def chi2r(self, v):
        self.lineEdit_2.setText(str(v))

    def update(self):
        EtModelFree.update(self)
        self.emit(QtCore.SIGNAL('model_update'))

    def onUpdateDecays(self):
        fit_names = [f.name for f in self.fits]
        self.comboBox.clear()
        self.comboBox_2.clear()
        self.comboBox.addItems(fit_names)
        self.comboBox_2.addItems(fit_names)
        #self.onUpdatePhasor()
        # phasor update

