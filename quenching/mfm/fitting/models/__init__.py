"""
This module is responsible contains all fitting modules for experimental data

The :py:mod:`.models`

1. :py:mod:`.models.tcspc`
2. :py:mod:`.models.fcs`
3. :py:mod:`.models.gloablfit`
4. :py:mod:`.models.parse`
5. :py:mod:`.models.proteinMC`
6. :py:mod:`.models.stopped_flow`


"""
from mfm import Data
from mfm.curve import Genealogy
from mfm.fitting.parameter import Parameter
from PyQt4 import QtGui

class Model(Genealogy):
    """
    This class provides
    """

    plot_classes = []

    def __init__(self, **kwargs):
        Genealogy.__init__(self)
        self.fit = kwargs.get('fit', None)
        self._y_values = []
        if isinstance(self.fit, mfm.fitting.fit.Fit):
            if isinstance(self.fit.data, Data):
                try:
                    self.y_values = np.zeros_like(self.fit.data.y)
                except AttributeError:
                    pass

    @property
    def y_values(self):
        return self._y_values

    @y_values.setter
    def y_values(self, v):
        self._y_values = v

    @property
    def x_axis(self):
        return self.fit.data.x

    @property
    def xmin(self):
        """
        The minimum x-value. This is only for convenience and returns :py:attr:`mfm.fit.xmin`
        """
        return self.fit.xmin

    @xmin.setter
    def xmin(self, v):
        self.fit.xmin = v

    @property
    def xmax(self):
        """
        The minimum x-value. This is only for convenience and returns :py:attr:`mfm.fit.xmax`
        """
        return self.fit.xmax

    @xmax.setter
    def xmax(self, v):
        self.fit.xmax = v

    @property
    def model(self):
        """
        The x-values and y-values of the model within the model-range defined by :py:attr:`.xmin` and :py:attr:`.xmax`
        """
        xmin, xmax = self.xmin, self.xmax
        x, m = self[xmin:xmax]
        return x, m

    @property
    def data(self):
        """
        The measurement data in the fitting range
        """
        xmin, xmax = self.xmin, self.xmax
        data = self.fit.data
        return data[xmin:xmax]

    def weighted_residuals(self, **kwargs):
        """
        This returns the weighted residuals of the model with respect to the data.

        If no data is provided the data of the fit is used. Otherwise the data passed
        by the argument `data` is used.

        :param data:
        :return: a numpy array of the weighted residuals
        """
        data = kwargs.get('data', self.fit.data)
        xmin = kwargs.get('xmin', self.xmin)
        xmax = kwargs.get('xmax', self.xmax)

        xmin = max(xmin, 0)
        xmax = max(xmax, 1)

        x, m = self[xmin:xmax]
        x, d, w = data[xmin:xmax]

        ml = min([len(m), len(d)])
        re = np.array((d[:ml] - m[:ml]) * w[:ml], dtype=np.float64)
        return re

    def chi2r(self, data=None):
        """
        The reduced chi2 of the model with respect to the passed data. If no data is passed
        the reduced chi2 of `fit.data` is returned

        :param data:
        :return: reduced chi2
        """
        data = self.fit.data if data is None else data
        return mfm.fitting.fit.get_chi2(self.parameters, self, data, True)

    @property
    def n_free(self):
        """
        The number of free fitting parameters
        """
        return len(self.parameters)

    @property
    def n_points(self):
        """
        The number of measurement points used in the fit
        """
        return self.xmax - self.xmin

    @property
    def parameters_all(self):
        """
        All model parameters irrespectively if parameter is fixed or linked
        """
        return mfm.find_object_type(self.__dict__.values(), Parameter)

    @property
    def parameters(self):
        """
        The parameters of the model which are not fixed or linked to another variable
        """
        return [p for p in self.parameters_all if not (p.isFixed or p.linkEnabled)]

    @property
    def parameter_linked_all(self):
        """
        An list of booleans of all parameters True if the respective parameter is linked otherwise False
        """
        return [p.linkEnabled for p in self.parameters_all]

    @property
    def parameter_dict(self):
        """
        A dictionary of the parameters (not the fixed of the linked parameters) here the keys are determined by
        the names of the parameters
        """
        re = dict()
        for p in self.parameters:
            re[p.name] = p
        return re

    @property
    def parameters_all_dict(self):
        """
        A dictionary of all parameters here the keys are determined by the names of the parameters
        """
        return dict([(p.name, p) for p in self.parameters_all])

    @property
    def parameter_names(self):
        """
        A list of the parameter names of the free parameters (not linked or fixed parameters)
        """
        return [p.name for p in self.parameters]

    @property
    def parameter_values(self):
        """
        A list of the parameter values of the free parameters (not linked or fixed parameters)
        """
        return [p.value for p in self.parameters]

    @parameter_values.setter
    def parameter_values(self, vs):
        ps = self.parameters
        for i, v in enumerate(vs):
            ps[i].value = v
        self.update_model()

    @property
    def parameterValues_all(self):
        """
        A list of the parameter values of all parameters (including linked and fixed parameters)
        """
        return [p.value for p in self.parameters_all]

    @property
    def parameter_fixed_all(self):
        """
        A list of booleans representing all parameters: True if fixed False if not Fixed
        """
        return [p.isFixed for p in self.parameters_all]

    @parameter_names.setter
    def parameter_names(self, v):
        self._parameterNames = v

    @property
    def parameter_names_all(self):
        """
        A list of all parameter names
        """
        return [p.name for p in self.parameters_all]

    @property
    def parameter_bounds_all(self):
        """
        The bounds of all parameters
        """
        return [pi.bounds for pi in self.parameters_all]

    @property
    def parameter_bounds(self):
        """
        The bounds of all free parameters
        """
        return [pi.bounds for pi in self.parameters]

    @parameter_bounds.setter
    def parameter_bounds(self, bounds):
        for i, pi in enumerate(self.parameters):
            pi.bounds = bounds[i]

    @property
    def plots(self):
        """
        The plots of the model
        """
        return [p for p in self.get_children() if isinstance(p, Plot)]

    @plots.setter
    def plots(self, v):
        self._plots = v

    def finalize(self):
        """
        This updates the values of the fitting parameters
        """
        pass

    def update_plots(self):
        for p in self.plots:
            p.update_all()
            p.update_widget()

    def update_widgets(self):
        [p.update_widget() for p in self.parameters_all if isinstance(p, ParameterWidget)]

    def update(self):
        if mfm.verbose:
            print "Update model: %s" % self.name
        self.update_model()
        self.update_widgets()
        self.update_plots()

    def update_model(self):
        pass

    def __getitem__(self, key):
        if isinstance(self.y_values, np.ndarray):
            xmin, xmax = 0, len(self.y_values)
            start = xmin if key.start is None else key.start
            stop = xmax if key.stop is None else key.stop
            step = 1 if key.step is None else key.step
            x, y = self.x_axis[start:stop:step], self.y_values[start:stop:step]
            return x, y
        else:
            return np.array([0]), np.array([0])

    def __str__(self):
        s = ""
        s += "Model: %s\n" % str(self.name)
        p0 = list(
            zip(self.parameter_names_all, self.parameterValues_all,
                self.parameter_bounds_all, self.parameter_fixed_all,
                self.parameter_linked_all))
        s += "Parameter\tValue\tBounds\tFixed\tLinked\n"
        for p in p0:
            s += "%s\t%.4e\t%s\t%s\t%s\n" % p
        return s


class ModelWidget(QtGui.QWidget, Model):

    def update(self, *__args):
        QtGui.QWidget.update(self, *__args)
        Model.update(self)

    def __init__(self, **kwargs):
        QtGui.QWidget.__init__(self, **kwargs)
        Model.__init__(self, **kwargs)


import fcs
from mfm.plots import Plot
import tcspc
import stopped_flow
import proteinMC
import parse
from globalfit import *


