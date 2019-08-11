from PyQt4 import QtGui

from mfm.curve import Genealogy
from mfm.experiments import Setup
from mfm.io.widgets import PDBLoad
import mfm.widgets as widgets


class LoadStructure(QtGui.QWidget, Genealogy, Setup):

    name = 'Protein-PDB'

    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self)
        Genealogy.__init__(self)
        Setup.__init__(self)
        self.parent = parent

        layout = QtGui.QVBoxLayout(self)
        self.layout = layout
        self.layout.setMargin(0)
        self.layout.setSpacing(0)
        self.pdbWidget = PDBLoad(self)
        self.layout.addWidget(self.pdbWidget)

    def load_data(self, filename=None):
        print("LoadStructure:loadData")
        self.pdbWidget.onLoadStructure()
        return self.pdbWidget.structure

    def __str__(self):
        s = 'ProteinMC\n'
        return s

    def autofitrange(self, fit):
        return None, None


class LoadStructureFolder(QtGui.QWidget, Genealogy, Setup):

    name = 'Trajectory'

    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self)
        Genealogy.__init__(self)
        Setup.__init__(self)
        self.parent = parent

        layout = QtGui.QVBoxLayout(self)
        self.layout = layout
        self.layout.setMargin(0)
        self.layout.setSpacing(0)
        self.pdbWidget = widgets.PDBFolderLoad(self)
        self.layout.addWidget(self.pdbWidget)

    def load_data(self):
        return self.pdbWidget.trajectory

    def __str__(self):
        s = 'ProteinMC\n'
        return s

    def autofitrange(self, fit):
        return None, None

