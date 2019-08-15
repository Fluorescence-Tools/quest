import tempfile
import sip
sip.setapi('QDate', 2)
sip.setapi('QDateTime', 2)
sip.setapi('QString', 2)
sip.setapi('QTextStream', 2)
sip.setapi('QTime', 2)
sip.setapi('QUrl', 2)
sip.setapi('QVariant', 2)

from PyQt5 import QtGui, QtWidgets
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
            self.setCentralWidget(dg)


    app = QtWidgets.QApplication(sys.argv)
    win = Main()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    start_gui()
