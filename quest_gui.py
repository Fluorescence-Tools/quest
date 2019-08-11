#!/usr/bin/python
#import gc
#gc.enable()
#gc.set_debug(gc.DEBUG_LEAK|gc.DEBUG_STATS)

import tempfile
import sip
sip.setapi('QDate', 2)
sip.setapi('QDateTime', 2)
sip.setapi('QString', 2)
sip.setapi('QTextStream', 2)
sip.setapi('QTime', 2)
sip.setapi('QUrl', 2)
sip.setapi('QVariant', 2)

from PyQt4 import QtGui, QtCore, uic

import lib
import sys
sys.tracebacklimit = 500

# After updating of icon run:
# pyrcc4 -o rescource_rc.py rescource.qrc
import lib.ui.rescource_rc

# encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')
from multiprocessing import freeze_support


class Main(QtGui.QMainWindow):
    """ Main window
    The program is structured in a tree
    self.rootNode -> n * Experiment ->  setup -> datasets -> Fit -> Model
    """

    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        dg = lib.tools.TransientDecayGenerator(tempfile.gettempdir())
        self.setCentralWidget(dg)
        

if __name__ == "__main__":
    freeze_support()
    app = QtGui.QApplication(sys.argv)
    win = Main()
    win.show()
    sys.exit(app.exec_())


