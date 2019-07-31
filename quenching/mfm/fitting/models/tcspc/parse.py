from PyQt4 import QtCore, QtGui

from mfm import plots
from mfm.fitting.fit import FittingWidget
from .. import parse
from mfm.fitting.models.tcspc import tcspc


class ParseDecayModel(parse.ParseModel):

    def __init__(self, fit, **kwargs):
        parse.ParseModel.__init__(self, fit, **kwargs)
        self.convolve = kwargs.get('convolve', [])
        self.corrections = kwargs.get('corrections', [])
        self.generic = kwargs.get('generic', [])

    def update_model(self, **kwargs):
        parse.ParseModel.update_model(self, **kwargs)
        decay = self.y_values
        if self.convolve.irf is not None:
            decay = self.convolve.convolve(self.y_values, mode='full')[:self.y_values.shape[0]]
            decay += (self.generic.scatter * self.convolve.irf.y)
        self.convolve.scale(decay, self.fit.data, self.generic.background)
        decay += self.generic.background
        decay[decay < 0.0] = 0.0
        if self.corrections.lintable is not None:
            decay *= self.corrections.lintable
        self.y_values = decay

    def clean(self, fit):
        convolve = self.convolve.clean(fit)
        corrections = self.corrections.clean(fit)
        generic = self.generic.clean()
        parse = self.parse.clean()

        new = ParseDecayModel(fit, convolve=convolve, corrections=corrections, generic=generic,
                              parse=parse)
        new.parse.parameters = [p.clean() for p in self.parse.parameters]
        return new


class ParseDecayModelWidget(ParseDecayModel, QtGui.QWidget):

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

        self.convolve = tcspc.ConvolveWidget(fit=fit, model=self, show_convolution_mode=False, dt=fit.data.dt, **kwargs)
        generic = tcspc.GenericWidget(parent=self, model=self, **kwargs)
        error_widget = tcspc.ErrorWidget(fit, **kwargs)
        pw = parse.ParseWidget(self, model_file='./settings/tcspc.model.json')
        corrections = tcspc.CorrectionsWidget(fit, model=self, **kwargs)

        self.fit = fit
        ParseDecayModel.__init__(self, fit=fit, parse=pw, convolve=self.convolve,
                                 generic=generic, corrections=corrections)
        fitting_widget = FittingWidget(fit=fit, **kwargs)

        layout = QtGui.QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setMargin(0)
        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.addWidget(fitting_widget)
        layout.addWidget(self.convolve)
        layout.addWidget(generic)
        layout.addWidget(pw)
        layout.addWidget(error_widget)
        layout.addWidget(corrections)
        self.setLayout(layout)

