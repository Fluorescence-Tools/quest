import tempfile

from qtpy import QtGui, QtWidgets
import lib
import lib.ui.rescource_rc
import sys
sys.tracebacklimit = 500

# After updating of icon run:
# pyrcc4 -o rescource_rc.py rescource.qrc
# encoding=utf8

def start_gui():

    class Main(QtWidgets.QMainWindow):
        """ Main window
        The program is structured in a tree
        self.rootNode -> n * Experiment ->  setup -> datasets -> Fit -> Model
        """

        def __init__(self):
            QtWidgets.QMainWindow.__init__(self)
            dg = lib.tools.TransientDecayGenerator(tempfile.gettempdir())
            self.dg = dg
            self.setCentralWidget(dg)
            self._create_menus()

        def _create_menus(self):
            mbar = self.menuBar()
            file_menu = mbar.addMenu("&File")

            load_pdb_action = file_menu.addAction("Load PDB…")
            load_pdb_action.triggered.connect(self.dg.onLoadPDB)

            file_menu.addSeparator()
            save_project_action = file_menu.addAction("Save project…")
            save_project_action.triggered.connect(self.dg.onSaveProject)
            load_project_action = file_menu.addAction("Load project…")
            load_project_action.triggered.connect(self.dg.onLoadProject)

            file_menu.addSeparator()
            exit_action = file_menu.addAction("Exit")
            exit_action.triggered.connect(self.close)


    app = QtWidgets.QApplication(sys.argv)
    win = Main()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    start_gui()
