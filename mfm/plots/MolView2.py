__author__ = 'thomas'
import os
from PyQt4 import QtGui, QtCore, uic
from OpenGL.GL import *
from PyQt4.QtOpenGL import *
from PyQt4.Qt import Qt

from mfm.plots.plotbase import Plot



class EmittingStream(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))


class MolQtWidget(QGLWidget):
    """
    http://www.mail-archive.com/pymol-users@lists.sourceforge.net/msg09609.html
    maybe later use this...: http://www.plosone.org/article/info:doi/10.1371/journal.pone.0021931
    """
    _buttonMap = {Qt.LeftButton: 0,
                  Qt.MidButton: 1,
                  Qt.RightButton: 2}

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
        self.pymol = pymol2.PyMOL()  # _pymolPool.getInstance()
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

        self.pymol.reshape(self.width(), self.height())
        self._timer = QtCore.QTimer()
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._pymolProcess)
        self.resizeGL(self.width(), self.height())
        # globalSettings.settingsChanged.connect(self._updateGlobalSettings)
        self._updateGlobalSettings()

    def openFile(self, File, frame=None, mode=None, verbose=True, object_name=None):
        if isinstance(File, str):
            if os.path.isfile(File):
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
                            self.pymol.cmd.hide('all')
                            self.pymol.cmd.do('show ribbon')
                            self.pymol.cmd.do('show spheres, name cb')
                        else:
                            self.pymol.cmd.hide('all')
                            self.pymol.cmd.show(mode)

                self.pymol.cmd.orient()
                # self.pymol.cmd.iterate_state()

    def __del__(self):
        pass

    def _updateGlobalSettings(self):
        # for k,v in globalSettings.settings.iteritems():
        #    self.pymol.cmd.set(k, v)
        #self.update()
        return

    def redoSizing(self):
        self.resizeGL(self.width(), self.height())

    def paintGL(self):
        glViewport(0, 0, self.width(), self.height())
        bottom = self.mapToGlobal(QtCore.QPoint(0, self.height())).y()
        # self.pymol.cmd.set("_stencil_parity", bottom & 0x1)
        self._doIdle()
        self.pymol.draw()

    def mouseMoveEvent(self, ev):
        self.pymol.drag(ev.x(), self.height() - ev.y(), 0)
        self._pymolProcess()

    def mousePressEvent(self, ev):
        if not self._enableUi:
            self.pymol.cmd.button("double_left", "None", "None")
            self.pymol.cmd.button("single_right", "None", "None")
        self.pymol.button(self._buttonMap[ev.button()], 0, ev.x(),
                          self.height() - ev.y(), 0)
        self._pymolProcess()

    def mouseReleaseEvent(self, ev):
        self.pymol.button(self._buttonMap[ev.button()], 1, ev.x(),
                          self.height() - ev.y(), 0)
        self._pymolProcess()
        self._timer.start(0)

    def resizeGL(self, w, h):
        self.pymol.reshape(w, h, True)
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


class ControlWidget(QtGui.QWidget):
    history = ['spectrum count, rainbow_rev, all, byres=1',
               'intra_fit all']

    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self)
        uic.loadUi('mfm/ui/plots/molViewControlWidget.ui', self)
        self.parent = parent
        """
        self.connect(self.pushButton_4, QtCore.SIGNAL("clicked()"), self.onReset)
        self.connect(self.pushButton_6, QtCore.SIGNAL("clicked()"), self.onIntrafit)
        self.connect(self.pushButton_5, QtCore.SIGNAL("clicked()"), self.onExeCommand)

        self.connect(self.actionNext_frame, QtCore.SIGNAL('triggered()'), self.onNextFrame)
        self.connect(self.actionPrevious_frame, QtCore.SIGNAL('triggered()'), self.onPreviousFrame)
        self.connect(self.actionPause, QtCore.SIGNAL('triggered()'), self.onStopPymol)
        self.connect(self.actionPlay, QtCore.SIGNAL('triggered()'), self.onPlayPymol)
        self.connect(self.actionTo_last_frame, QtCore.SIGNAL('triggered()'), self.onLastState)
        self.connect(self.actionTo_first_frame, QtCore.SIGNAL('triggered()'), self.onFirstState)
        self.connect(self.actionShow_quencher, QtCore.SIGNAL('triggered()'), self.onHighlightQuencher)

        self.connect(self.lineEdit, QtCore.SIGNAL("returnPressed ()"), self.onExeCommand)
        self.connect(self.radioButton, QtCore.SIGNAL("clicked()"), self.onUpdateSpectrum)
        self.connect(self.radioButton_2, QtCore.SIGNAL("clicked()"), self.onUpdateSpectrum)
        self.connect(self.actionCurrentStateChanged, QtCore.SIGNAL('triggered()'), self.onCurrentStateChanged)
        """

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

    def onHighlightQuencher(self):
        quencher = self.parent.quencher
        pymol = self.parent.pymol
        for res_name in quencher:
            pymol.cmd.do("hide lines, resn %s" % res_name)
            pymol.cmd.do("show sticks, resn %s" % res_name)
            pymol.cmd.do("color red, resn %s" % res_name)


class MolView(Plot):

    @property
    def pymol(self):
        return self.pymolWidget.pymol

    def __init__(self, fit=None, enableUi=False, mode='cartoon', sequence=True, **kwargs):
        Plot.__init__(self)
        self.quencher = kwargs.get('quencher', )
        self.name = kwargs.get('name', 'MolView')
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
            self.pymol.bfact = list(bfact)
            self.pymol.cmd.alter("all and n. CA", "b=bfact.pop(0)")
        self.pymol.cmd.do("orient")

    def append_structure(self, structure):
        self.pymol.cmd.read_pdbstr(str(structure), 'p')
        self.pltControl.n_states = self.pymol.cmd.count_states(selection="(all)")

    def open_structure(self, structure, bfact=None, mode=None):
        mode = mode if mode is not None else self.mode
        self.pymol.cmd.read_pdbstr(str(structure), 'p')
        self.set_bfactor(bfact)

        if mode is not None:
            if mode == 'coarse':
                self.pymol.cmd.hide('all')
                self.pymol.cmd.do('show ribbon')
                self.pymol.cmd.do('show spheres, name cb')
            else:
                self.pymol.cmd.hide('all')
                self.pymol.cmd.show(mode)

    def replace_structure(self, structure, bfact, stateNbr=0):
        print("MolView:replace_structure")
        self.open_structure(structure, stateNbr, mode=self.mode)
        self.set_bfactor(bfact)

    def setState(self, nbr):
        print("set state: % s" % nbr)
        self.pymol.cmd.set("state", nbr)

    def reset(self):
        self.pymolWidget.reset()

    def colorStates(self, rgbColors):
        print("colorStates")  # TODO
        for i, r in enumerate(rgbColors):
            color = "[%s, %s, %s]" % (r[0] / 255., r[1] / 255., r[2] / 255.)
            self.pymol.cmd.set_color("tmpcolor", color)
            self.pymol.cmd.set("cartoon_color", "tmpcolor", "p", (i + 1))


#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vispy: testskip
# -----------------------------------------------------------------------------
# Copyright (c) 2014, Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------
# Abstract: show mesh primitive
# Keywords: cone, arrow, sphere, cylinder, qt
# -----------------------------------------------------------------------------

try:
    from sip import setapi

    setapi("QVariant", 2)
    setapi("QString", 2)
except ImportError:
    pass

from PyQt4 import QtGui, QtCore
import sys
import OpenGL.GL as gl

import numpy as np
from vispy import app, gloo
from vispy.util.transforms import perspective, translate, rotate
from vispy.geometry import meshdata as md
from vispy.geometry import generation as gen

vertex = """
#version 120

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform vec3 u_light_position;
uniform vec3 u_light_spec_position;

attribute vec3  a_position;
attribute vec3  a_color;
attribute float a_radius;

varying vec3  v_color;
varying vec4  v_eye_position;
varying float v_radius;
varying vec3  v_light_direction;

void main (void) {
    v_radius = a_radius;
    v_color = a_color;

    v_eye_position = u_view * u_model * vec4(a_position,1.0);
    v_light_direction = normalize(u_light_position);
    float dist = length(v_eye_position.xyz);

    gl_Position = u_projection * v_eye_position;

    // stackoverflow.com/questions/8608844/...
    //  ... resizing-point-sprites-based-on-distance-from-the-camera
    vec4  proj_corner = u_projection * vec4(a_radius, a_radius, v_eye_position.z, v_eye_position.w);
    gl_PointSize = 512.0 * proj_corner.x / proj_corner.w;
}
"""

fragment = """
#version 120

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform vec3 u_light_position;
uniform vec3 u_light_spec_position;

varying vec3  v_color;
varying vec4  v_eye_position;
varying float v_radius;
varying vec3  v_light_direction;
void main()
{
    // r^2 = (x - x0)^2 + (y - y0)^2 + (z - z0)^2
    vec2 texcoord = gl_PointCoord* 2.0 - vec2(1.0);
    float x = texcoord.x;
    float y = texcoord.y;
    float d = 1.0 - x*x - y*y;
    if (d <= 0.0)
        discard;

    float z = sqrt(d);
    vec4 pos = v_eye_position;
    pos.z += v_radius*z;
    vec3 pos2 = pos.xyz;
    pos = u_projection * pos;
    gl_FragDepth = 0.5*(pos.z / pos.w)+0.5;
    vec3 normal = vec3(x,y,z);
    float diffuse = clamp(dot(normal, v_light_direction), 0.0, 1.0);

    // Specular lighting.
    vec3 M = pos2.xyz;
    vec3 O = v_eye_position.xyz;
    vec3 L = u_light_spec_position;
    vec3 K = normalize(normalize(L - M) + normalize(O - M));
    // WARNING: abs() is necessary, otherwise weird bugs may appear with some
    // GPU drivers...
    float specular = clamp(pow(abs(dot(normal, K)), 40.), 0.0, 1.0);
    vec3 v_light = vec3(1., 1., 1.);
    gl_FragColor.rgb = .15*v_color +  .55*diffuse * v_color + .35*specular * v_light;
}
"""


class MolecularViewerCanvas(app.Canvas):
    def __init__(self, fname):
        app.Canvas.__init__(self, title='Molecular viewer')
        self.size = 800, 600

        self.program = gloo.Program(vertex, fragment)
        self.view = np.eye(4, dtype=np.float32)
        self.model = np.eye(4, dtype=np.float32)
        self.projection = np.eye(4, dtype=np.float32)
        self.translate = 5
        translate(self.view, 0, 0, -self.translate)

        self.load_molecule(fname)
        self.load_data()

        self.theta = 0
        self.phi = 0

        self.timer = app.Timer(1.0 / 30)  # change rendering speed here
        self.timer.connect(self.on_timer)
        self.timer.start()


    def load_molecule(self, fname):

        molecule = np.load(fname)
        self._nAtoms = molecule.shape[0]

        # The x,y,z values store in one array
        self.coords = molecule[:, :3]

        # The array that will store the color and alpha scale for all the atoms.
        self.atomsColours = molecule[:, 3:6]

        # The array that will store the scale for all the atoms.
        self.atomsScales = molecule[:, 6]

    # ---------------------------------
    def set_data(self, vertices, filled, outline):
        self.filled_buf = gloo.IndexBuffer(filled)
        self.outline_buf = gloo.IndexBuffer(outline)
        self.vertices_buff = gloo.VertexBuffer(vertices)
        self.program.bind(self.vertices_buff)
        self.update()

    def load_data(self):
        n = self._nAtoms

        data = np.zeros(n, [('a_position', np.float32, 3),
                            ('a_color', np.float32, 3),
                            ('a_radius', np.float32, 1)])

        data['a_position'] = self.coords
        data['a_color'] = self.atomsColours
        data['a_radius'] = self.atomsScales

        self.program.bind(gloo.VertexBuffer(data))

        self.program['u_model'] = self.model
        self.program['u_view'] = self.view
        self.program['u_light_position'] = 0., 0., 2.
        self.program['u_light_spec_position'] = -5., 5., -5.


    def on_initialize(self, event):
        gl.glClearColor(0, 0, 0, 1)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_VERTEX_PROGRAM_POINT_SIZE)
        gl.glEnable(gl.GL_POINT_SPRITE)


    def on_key_press(self, event):
        if event.text == ' ':
            if self.timer.running:
                self.timer.stop()
            else:
                self.timer.start()
                # if event.text == 'A':
                # self.

    def on_timer(self, event):
        self.theta += .5
        self.phi += .5
        self.model = np.eye(4, dtype=np.float32)

        rotate(self.model, self.theta, 0, 0, 1)
        rotate(self.model, self.phi, 0, 1, 0)

        self.program['u_model'] = self.model
        self.update()

    def on_resize(self, event):
        width, height = event.size
        gl.glViewport(0, 0, width, height)
        self.projection = perspective(25.0, width / float(height), 2.0, 100.0)
        self.program['u_projection'] = self.projection

    def on_mouse_wheel(self, event):
        self.translate -= event.delta[1]
        self.translate = max(-1, self.translate)
        self.view = np.eye(4, dtype=np.float32)

        translate(self.view, 0, 0, -self.translate)

        self.program['u_view'] = self.view
        self.update()

    def on_draw(self, event):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        self.program.draw(gl.GL_POINTS)


class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)

        self.resize(700, 500)
        self.setWindowTitle('vispy example ...')

        self.splitter_v = QtGui.QSplitter(QtCore.Qt.Vertical)
        self.splitter_v.addWidget(QtGui.QLabel('test'))

        self.canvas = MolecularViewerCanvas('protein.npy')
        self.canvas.create_native()
        self.canvas.native.setParent(self)

        # Central Widget
        splitter1 = QtGui.QSplitter(QtCore.Qt.Horizontal)
        splitter1.addWidget(self.splitter_v)
        splitter1.addWidget(self.canvas.native)

        self.setCentralWidget(splitter1)

        # FPS message in statusbar:
        self.status = self.statusBar()
        self.status.showMessage("...")
