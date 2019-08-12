__author__ = 'thomas'
import os
import sys

from PyQt5 import QtGui, QtCore, uic, QtWidgets
from OpenGL.GL import *
from PyQt5.QtOpenGL import *
from PyQt5.Qt import Qt

from lib.plots.plotbase import Plot

try:
    import pymol2
except ImportError:
    pass



"""
Id Atom

id_atom returns the original source id of a single atom, or raises and exception if the atom does not exist or if the selection corresponds to multiple atoms.
PYMOL API

list = cmd.id_atom(string selection)
"""


class EmittingStream(QtCore.QObject):

    textWritten = QtCore.pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))



class MolQtWidget(QGLWidget):
    """
    http://www.mail-archive.com/pymol-users@lists.sourceforge.net/msg09609.html
    maybe later use this...: http://www.plosone.org/article/info:doi/10.1371/journal.pone.0021931
    """
    _buttonMap = {Qt.LeftButton:0,
                  Qt.MidButton:1,
                  Qt.RightButton:2}

    def __init__(self, parent, enableUi=True, File="", play=False, sequence=False):
        f = QGLFormat()
        f.setStencil(True)
        f.setRgba(True)
        f.setDepth(True)
        f.setDoubleBuffer(True)
        self.play = play
        self.nFrames = 0
        QGLWidget.__init__(self, f, parent=parent)
        self.setMinimumSize(200, 150)
        self._enableUi = enableUi
        self.pymol = pymol2.PyMOL()# _pymolPool.getInstance()
        self.pymol.start()
        self.cmd = self.pymol.cmd
        # self.toPymolName = self.pymol.toPymolName ### Attribute Error
        self._pymolProcess()

        if not self._enableUi:
            self.pymol.cmd.set("internal_gui", 0)
            self.pymol.cmd.set("internal_feedback", 1)
            self.pymol.cmd.button("double_left", "None", "None")
            self.pymol.cmd.button("single_right", "None", "None")

        self.pymol.cmd.set("internal_gui_mode", "0")
        self.pymol.cmd.set("internal_feedback", "0")
        if sequence:
            self.pymol.cmd.set("seq_view", "1")
        if File is not "":
            self.nFrames += 1
            self.pymol.cmd.load(File, 'p', self.nFrames)

        self.pymol.reshape(self.width(),self.height())
        self._timer = QtCore.QTimer()
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._pymolProcess)
        self.resizeGL(self.width(),self.height())
        #globalSettings.settingsChanged.connect(self._updateGlobalSettings)
        self._updateGlobalSettings()

    def openFile(self, File, frame=None, mode=None, verbose=True, object_name=None):
        if self.play is True:
            self.pymol.cmd.mplay()

        if frame is None:
            self.nFrames += 1
            frame = self.nFrames

        object_name = object_name if object_name is not None else os.path.basename(File)
        if verbose:
            print("Pymol opening file: %s" % File)
            print("Object name: %s" % object_name)
        self.pymol.cmd.load(File, object_name, frame)
        if self.nFrames == 1:
            if mode is not None:
                if mode == 'coarse':
                    self.pymolWidget.pymol.cmd.hide('all')
                    self.pymolWidget.pymol.cmd.do('show ribbon')
                    self.pymolWidget.pymol.cmd.do('show spheres, name cb')
                else:
                    self.pymol.cmd.hide('all')
                    self.pymol.cmd.show(mode)

        self.pymol.cmd.orient()
        #self.pymol.cmd.iterate_state()

    def __del__(self):
        pass

    def _updateGlobalSettings(self):
        #for k,v in globalSettings.settings.iteritems():
        #    self.pymol.cmd.set(k, v)
        #self.update()
        return

    def redoSizing(self):
        self.resizeGL(self.width(), self.height())

    def paintGL(self):
        glViewport(0, 0, self.width(), self.height())
        bottom = self.mapToGlobal(QtCore.QPoint(0,self.height())).y()
        #self.pymol.cmd.set("_stencil_parity", bottom & 0x1)
        self._doIdle()
        self.pymol.draw()

    def mouseMoveEvent(self, ev):
        self.pymol.drag(ev.x(), self.height()-ev.y(),0)
        self._pymolProcess()

    def mousePressEvent(self, ev):
        if not self._enableUi:
            self.pymol.cmd.button("double_left","None","None")
            self.pymol.cmd.button("single_right","None","None")
        self.pymol.button(self._buttonMap[ev.button()], 0, ev.x(),
                          self.height() - ev.y(),0)
        self._pymolProcess()

    def mouseReleaseEvent(self, ev):
        self.pymol.button(self._buttonMap[ev.button()], 1, ev.x(),
                          self.height()-ev.y(),0)
        self._pymolProcess()
        self._timer.start(0)

    def resizeGL(self, w, h):
        self.pymol.reshape(w,h, True)
        self._pymolProcess()

    def initializeGL(self):
        pass

    def _pymolProcess(self):
        self._doIdle()
        self.update()

    def _doIdle(self):
        if self.pymol.idle():
            self._timer.start(0)

    def strPDB(self, str):
        self.cmd.read_pdbstr(str)

    def reset(self):
        self.nFrames = 0
        self.pymol.cmd.reinitialize()




class ControlWidget(QtWidgets.QWidget):

    history = ['spectrum count, rainbow_rev, all, byres=1',
               'intra_fit all']

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self)
        uic.loadUi('experiments/plots/ui/molViewControlWidget.ui', self)
        self.parent = parent


        self.connect(self.pushButton_4, QtCore.SIGNAL("clicked()"), self.onReset)
        self.connect(self.pushButton_6, QtCore.SIGNAL("clicked()"), self.onIntrafit)
        self.connect(self.pushButton_5, QtCore.SIGNAL("clicked()"), self.onExeCommand)

        self.connect(self.actionNext_frame, QtCore.SIGNAL('triggered()'), self.onNextFrame)
        self.connect(self.actionPrevious_frame, QtCore.SIGNAL('triggered()'), self.onPreviousFrame)
        self.connect(self.actionPause, QtCore.SIGNAL('triggered()'), self.onStopPymol)
        self.connect(self.actionPlay, QtCore.SIGNAL('triggered()'), self.onPlayPymol)
        self.connect(self.actionTo_last_frame, QtCore.SIGNAL('triggered()'), self.onLastState)
        self.connect(self.actionTo_first_frame, QtCore.SIGNAL('triggered()'), self.onFirstState)


        self.connect(self.lineEdit, QtCore.SIGNAL("returnPressed ()"), self.onExeCommand)
        self.connect(self.radioButton, QtCore.SIGNAL("clicked()"), self.onUpdateSpectrum)
        self.connect(self.radioButton_2, QtCore.SIGNAL("clicked()"), self.onUpdateSpectrum)
        self.connect(self.actionCurrentStateChanged, QtCore.SIGNAL('triggered()'),  self.onCurrentStateChanged)

    def onPreviousFrame(self):
        s = self.current_state
        s -= 1
        s = self.n_states if s < 1 else s
        self.current_state = (s % self.n_states) + 1

    def onNextFrame(self):
        self.current_state = ((self.current_state + 1) % self.n_states) + 1

    def onCurrentStateChanged(self):
        self.parent.pymolWidget.pymol.cmd.do("set state, %i" % self.current_state)

    @property
    def current_state(self):
        return int(self.spinBox.value())

    @current_state.setter
    def current_state(self, v):
        self.spinBox.setValue(v)
        self.parent.pymolWidget.pymol.cmd.do("set state, %i" % v)
        self.onCurrentStateChanged()

    @property
    def n_states(self):
        return int(self.spinBox_2.value())

    @n_states.setter
    def n_states(self, v):
        self.spinBox_2.setValue(v)
        self.horizontalSlider.setMaximum(v)
        self.spinBox.setMaximum(v)

    @property
    def spectrum_type(self):
        if self.radioButton.isChecked():
            return 'movemap'
        elif self.radioButton_2.isChecked():
            return 'index'

    @property
    def pymol(self):
        return self.parent.pymolWidget.pymol

    def onIntrafit(self):
        self.pymol.cmd.do("intra_fit all")

    @property
    def first_frame(self):
        return int(self.spinBox_3.value())

    @property
    def last_frame(self):
        return int(self.spinBox_2.value())

    @property
    def step(self):
        return int(self.spinBox.value())

    def onLastState(self):
        self.parent.pymolWidget.pymol.cmd.do("set state, %i" % self.n_states)
        self.current_state = self.n_states

    def onFirstState(self):
        self.parent.pymolWidget.pymol.cmd.do("set state, %i" % 1)
        self.current_state = 1

    def onExeCommand(self):
        print("onExeCommand")
        sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)
        c = str(self.lineEdit.text())
        print("%s" % c)
        self.parent.pymolWidget.pymol.cmd.do(c)
        self.lineEdit.clear()
        sys.stdout = sys.__stdout__

    def normalOutputWritten(self, text):
        """Append text to the QTextEdit."""
        # Maybe QTextEdit.append() works as well, but this is how I do it:
        cursor = self.plainTextEdit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.plainTextEdit.setTextCursor(cursor)
        self.plainTextEdit.ensureCursorVisible()

    def onUpdateSpectrum(self):
        if self.spectrum_type == 'movemap':
            self.parent.pymolWidget.pymol.cmd.do("spectrum b")
        elif self.spectrum_type == 'index':
            self.parent.pymolWidget.pymol.cmd.do('spectrum count, rainbow_rev, all, byres=1')

    def onStopPymol(self):
        print("onStopPymol")
        self.parent.pymolWidget.pymol.cmd.mstop()

    def onPlayPymol(self):
        print("onPlayPymol")
        self.parent.pymolWidget.pymol.cmd.mplay()

    def onReset(self):
        self.plainTextEdit.clear()
        self.parent.reset()


class MolView(Plot):

    name = "MolView"

    def __init__(self, fit=None, enableUi=False, mode='cartoon', sequence=True):
        Plot.__init__(self)
        self.fit = fit
        self.mode = mode
        self.pltControl = ControlWidget(self)
        self.layout = QtGui.QVBoxLayout(self)

        self.pymolWidget = MolQtWidget(self, play=False, sequence=sequence, enableUi=enableUi)
        self.layout.addWidget(self.pymolWidget)
        self.open_structure(fit.data)

    def open_file(self, filename, bfact=None):
        self.pymolWidget.openFile(filename, mode=self.mode)
        self.set_bfactor(bfact)

    def set_bfactor(self, bfact=None):
        if bfact is not None:
            self.pymolWidget.pymol.bfact = list(bfact)
            self.pymolWidget.pymol.cmd.alter("all and n. CA", "b=bfact.pop(0)")
        self.pymolWidget.pymol.cmd.do("orient")

    def append_structure(self, structure):
        self.pymolWidget.pymol.cmd.read_pdbstr(str(structure), 'p')
        self.pltControl.n_states = self.pymolWidget.pymol.cmd.count_states(selection="(all)")

    def open_structure(self, structure, bfact=None, mode=None):
        mode = mode if mode is not None else self.mode
        self.pymolWidget.pymol.cmd.read_pdbstr(str(structure), 'p')
        self.set_bfactor(bfact)

        if mode is not None:
            if mode == 'coarse':
                self.pymolWidget.pymol.cmd.hide('all')
                self.pymolWidget.pymol.cmd.do('show ribbon')
                self.pymolWidget.pymol.cmd.do('show spheres, name cb')
            else:
                self.pymol.cmd.hide('all')
                self.pymol.cmd.show(mode)

    def replace_structure(self, structure, bfact, stateNbr=0):
        print("MolView:replace_structure")
        self.open_structure(structure, stateNbr, mode=self.mode)
        self.set_bfactor(bfact)

    def setState(self, nbr):
        print("set state: % s" % nbr)
        self.pymolWidget.pymol.cmd.set("state", nbr)

    def reset(self):
        self.pymolWidget.reset()

    def colorStates(self, rgbColors):
        print("colorStates") # TODO
        for i, r in enumerate(rgbColors):
            color = "[%s, %s, %s]" % (r[0] / 255., r[1] / 255., r[2] / 255.)
            self.pymolWidget.pymol.cmd.set_color("tmpcolor", color)
            self.pymolWidget.pymol.cmd.set("cartoon_color", "tmpcolor", "p", (i + 1))

