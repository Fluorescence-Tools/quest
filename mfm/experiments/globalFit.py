from PyQt4 import QtGui
from collections import OrderedDict

from mfm.curve import Genealogy
from mfm.experiments import Setup
import mfm


class GlobalFitSetup(QtGui.QWidget, Genealogy, Setup):

    name = 'GlobalFitSetup'

    def __init__(self, name, parent=None):
        QtGui.QWidget.__init__(self)
        Genealogy.__init__(self)
        self.hide()

        self.parameterWidgets = []
        self.parameters = OrderedDict([])
        self.parent = parent
        self.name = name

    def autofitrange(self, fit, threshold=10.0, area=0.999):
        return 0, 0

    def load_data(self, filename=None):
        d = mfm.DataCurve()
        d.name = "Global-fit"
        return d

    def __str__(self):
        s = 'Global-Fit\n'
        s += 'Name: \t%s \n' % self.name
        return s


