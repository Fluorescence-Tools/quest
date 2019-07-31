from PyQt4 import QtGui

from mfm.curve import Genealogy


class Plot(QtGui.QWidget, Genealogy):

    def __init__(self, parent=None):
        Genealogy.__init__(self)
        QtGui.QWidget.__init__(self, parent)
        parent = parent
        self.pltControl = None
        self.widgets = []

    def hide(self):
        fit = self.fit
        self.hide()
        if isinstance(self.pltControl, QtGui.QWidget):
            self.pltControl.hide()

    def update_widget(self):
        for w in self.widgets:
            w.update()

    def update_all(self):
        pass

    def close(self):
        QtGui.QWidget.close(self)
        if isinstance(self.pltControl, QtGui.QWidget):
            self.pltControl.close()