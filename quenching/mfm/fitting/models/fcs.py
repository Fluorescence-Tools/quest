from PyQt4 import QtGui

from mfm import plots
from mfm.fitting.models.parse import ParseModelWidget


class ParseFCSWidget(ParseModelWidget):

    """
    FCS
    """

    plot_classes = [(plots.LinePlot, {'d_scalex': 'log',
                                                   'd_scaley': 'lin',
                                                   'r_scalex': 'log',
                                                   'r_scaley': 'lin',
                                                   }),
                    (plots.SurfacePlot, {})
    ]

    def __init__(self, fit):
        self.icon = QtGui.QIcon(":/icons/icons/FCS.ico")
        ParseModelWidget.__init__(self, fit, model_file='./settings/fcs.model.json')

