from PyQt4 import QtGui

from lib import Genealogy


class Plot(QtGui.QWidget, Genealogy):
    def __init__(self, parent=None):
        Genealogy.__init__(self)
        QtGui.QWidget.__init__(self, parent)
        parent = parent
        self.widgets = []

    def hide(self):
        fit = self.fit
        self.hide()
        try:
            self.pltControl.hide()
        except AttributeError:
            print("No plot-control widget to hide.")

    def updateWidget(self):
        for w in self.widgets:
            w.update()

    def updateAll(self):
        pass

    def close(self):
        QtGui.QWidget.close(self)
        try:
            self.pltControl.close()
        except AttributeError:
            print(self)
            print("No plot-control widget to close.")
