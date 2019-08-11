"""
This module is responsible for all experiments/fits

The :py:mod:`experiments` module contains the fitting models and the setups (assembled reading routines) for
different experimental setups. Furthermore, it contains a set of plotting libraries.


"""
from mfm.curve import Genealogy


class Experiment(Genealogy):
    """
    All information contained within `ChiSurf` is associated to an experiment. Each experiment
    is associated with a list of models and a list of setups. The list of models and the list
    of setups determine the applicable models and loadable data-types respectively.
    """

    @property
    def setups(self):
        return self.get_setups()

    @property
    def setup_names(self):
        return self.get_setup_names()

    @property
    def models(self):
        return self.model_classes

    @property
    def model_names(self):
        return self.get_model_names()

    def __init__(self, name):
        Genealogy.__init__(self)
        self.name = name
        self.model_classes = list()
        self._setups = list()

    def add_model(self, model):
        self.model_classes.append(model)

    def add_models(self, models):
        for model in models:
            self.model_classes.append(model)

    def add_setup(self, setup):
        self.setups.append(setup)
        setup.link(self)

    def get_setups(self):
        return self._setups

    def get_setup_names(self):
        names = list()
        for s in self.setups:
            names.append(s.name)
        return names

    def get_model_names(self):
        names = list()
        for s in self.model_classes:
            names.append(s.name)
        return names


class Setup(object):

    def autofitrange(self, data, **kwargs):
        xmin = 0
        xmax = len(data.y) - 1
        return xmin, xmax

    def __init__(self):
        pass

    def load_data(self, filename):
        pass

    def get_dataset(self, experiment, filename=None):
        data = self.load_data(filename=filename)
        data.experiment = experiment
        data.setup = self
        data.link(experiment)
        return data

from .fcs import *
from .tcspc import *
from .globalFit import *
from . import modelling
