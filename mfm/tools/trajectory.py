import os
import glob
from itertools import izip

import mfm
import numpy as np
from PyQt4 import QtCore, QtGui, uic
import mdtraj as md
import tables
import mdtraj.scripts.mdconvert as mdconvert

from mfm.structure.cStructure import translate, rotate, below_min_distance
from mfm.widgets import MyMessageBox


class AlignTrajectoryWidget(QtGui.QWidget):
    # WORKS

    @property
    def stride(self):
        return int(self.spinBox.value())

    @property
    def atom_list(self):
        txt = str(self.plainTextEdit.toPlainText())
        atom_list = np.fromstring(txt, dtype=np.int32, sep=",")
        return atom_list

    @property
    def trajectory_filename(self):
        return str(self.lineEdit.text())

    @trajectory_filename.setter
    def trajectory_filename(self, v):
        self.lineEdit.setText(str(v))

    def __init__(self, **kwargs):
        self.trajectory = None

        QtGui.QWidget.__init__(self)
        uic.loadUi('mfm/ui/tools/align_trajectory.ui', self)
        self.connect(self.actionOpen_trajectory, QtCore.SIGNAL('triggered()'), self.onOpenTrajectory)
        self.connect(self.actionSave_aligned_trajectory, QtCore.SIGNAL('triggered()'), self.onSaveTrajectory)

    def onOpenTrajectory(self, filename=None):
        print "onOpenTrajectory"
        self.trajectory_filename = str(QtGui.QFileDialog.getOpenFileName(None, 'Open H5-Model file', '', 'H5-files (*.h5)'))

    def onSaveTrajectory(self, target_filename=None):
        if target_filename is None:
            target_filename = str(QtGui.QFileDialog.getSaveFileName(None, 'Save H5-Model file', '', 'H5-files (*.h5)'))
        filename = self.trajectory_filename
        atom_indices = self.atom_list
        stride = self.stride

        # Make empty trajectory
        frame_0 = md.load_frame(filename, 0)
        target_traj = md.Trajectory(xyz=np.empty((0, frame_0.n_atoms, 3)), topology=frame_0.topology)
        target_traj.save(target_filename)

        chunk_size = 1000
        table = tables.open_file(target_filename, 'a')
        for i, chunk in enumerate(md.iterload(filename, chunk=chunk_size, stride=stride)):
            chunk = chunk.superpose(frame_0, frame=0, atom_indices=atom_indices)
            xyz = chunk.xyz.copy()
            table.root.xyz.append(xyz)
            table.root.time.append(np.arange(i * chunk_size, i * chunk_size + xyz.shape[0], dtype=np.float32))
        table.close()


class JoinTrajectoriesWidget(QtGui.QWidget):
    #WORKS

    @property
    def stride(self):
        return

    @property
    def chunk_size(self):
        return int(self.spinBox.value())

    @property
    def reverse_traj_1(self):
        return bool(self.checkBox_2.isChecked())

    @property
    def reverse_traj_2(self):
        return bool(self.checkBox.isChecked())

    @property
    def trajectory_filename_1(self):
        return str(self.lineEdit.text())

    @trajectory_filename_1.setter
    def trajectory_filename_1(self, v):
        self.lineEdit.setText(str(v))

    @property
    def trajectory_filename_2(self):
        return str(self.lineEdit_2.text())

    @trajectory_filename_2.setter
    def trajectory_filename_2(self, v):
        self.lineEdit_2.setText(str(v))

    @property
    def join_mode(self):
        if self.radioButton_2.isChecked():
            return 'time'
        else:
            return 'atoms'

    def __init__(self, **kwargs):
        QtGui.QWidget.__init__(self)
        uic.loadUi('mfm/ui/tools/join_traj.ui', self)
        self.connect(self.actionOpen_first_trajectory, QtCore.SIGNAL('triggered()'), self.onOpenTrajectory_1)
        self.connect(self.actionOpen_second_trajectory, QtCore.SIGNAL('triggered()'), self.onOpenTrajectory_2)
        self.connect(self.actionSave_joined_trajectory, QtCore.SIGNAL('triggered()'), self.onJoinTrajectories)

    def onJoinTrajectories(self):
        print "onJoinTrajectories"
        target_filename = str(QtGui.QFileDialog.getSaveFileName(None, 'Save H5-Model file', '', 'H5-files (*.h5)'))

        fn1 = self.trajectory_filename_1
        fn2 = self.trajectory_filename_2

        r1 = self.reverse_traj_1
        r2 = self.reverse_traj_2

        traj_1 = md.load_frame(fn1, index=0)
        traj_2 = md.load_frame(fn2, index=0)

        # Create empty trajectory
        if self.join_mode == 'time':
            traj_join = traj_1.join(traj_2)
            axis = 0
        elif self.join_mode == 'atoms':
            traj_join = traj_1.stack(traj_2)
            axis = 1

        target_traj = md.Trajectory(xyz=np.empty((0, traj_join.n_atoms, 3)), topology=traj_join.topology)
        target_traj.save(target_filename)

        chunk_size = self.chunk_size
        table = tables.open_file(target_filename, 'a')
        for i, (c1, c2) in enumerate(izip(md.iterload(fn1, chunk=chunk_size), md.iterload(fn2, chunk=chunk_size))):
            xyz_1 = c1.xyz[::-1] if r1 else c1.xyz
            xyz_2 = c2.xyz[::-1] if r2 else c2.xyz
            xyz = np.concatenate((xyz_1, xyz_2), axis=axis)

            table.root.xyz.append(xyz)
            table.root.time.append(np.arange(i * chunk_size, i * chunk_size + xyz.shape[0], dtype=np.float32))

        table.close()

    def onOpenTrajectory_1(self, filename=None):
        print "onOpenTrajectory 1"
        if filename is None:
            self.trajectory_filename_1 = str(QtGui.QFileDialog.getOpenFileName(None, 'Open H5-Model file', '', 'H5-files (*.h5)'))

    def onOpenTrajectory_2(self, filename=None):
        print "onOpenTrajectory 2"
        if filename is None:
            self.trajectory_filename_2 = str(QtGui.QFileDialog.getOpenFileName(None, 'Open H5-Model file', '', 'H5-files (*.h5)'))


class SaveTopology(QtGui.QWidget):

    @property
    def trajectory_filename(self):
        return str(self.lineEdit.text())

    @trajectory_filename.setter
    def trajectory_filename(self, v):
        self.lineEdit.setText(str(v))

    def __init__(self, **kwargs):
        QtGui.QWidget.__init__(self)
        uic.loadUi('mfm/ui/tools/save_topology.ui', self)
        self.connect(self.actionOpen_trajectory, QtCore.SIGNAL('triggered()'), self.onOpenTrajectory)
        self.connect(self.actionSave_clash_free_trajectory, QtCore.SIGNAL('triggered()'), self.onSaveTopology)

    def onSaveTopology(self):
        print "onRemoveClashes"
        target_filename = str(QtGui.QFileDialog.getSaveFileName(None, 'Save PDB-file', '', 'PDB-files (*.pdb)'))
        filename = self.trajectory_filename
        # Make empty trajectory
        frame_0 = md.load_frame(filename, 0)
        frame_0.save(target_filename)

    def onOpenTrajectory(self, filename=None):
        print "onOpenTrajectory 1"
        if filename is None:
            self.trajectory_filename = str(QtGui.QFileDialog.getOpenFileName(None, 'Open H5-Model file', '', 'H5-files (*.h5)'))


class RotateTranslateTrajectoryWidget(QtGui.QWidget):
    # WORKS

    @property
    def stride(self):
        return int(self.spinBox.value())

    @property
    def rotation_matrix(self):
        r = np.array([[float(self.lineEdit_3.text()), float(self.lineEdit_6.text()), float(self.lineEdit_9.text())],
                      [float(self.lineEdit_4.text()), float(self.lineEdit_7.text()), float(self.lineEdit_10.text())],
                      [float(self.lineEdit_5.text()), float(self.lineEdit_8.text()), float(self.lineEdit_11.text())]],
                     dtype=np.float32)
        return r

    @rotation_matrix.setter
    def rotation_matrix(self, v):
        self.lineEdit_3.setText(str(v[0, 0]))
        self.lineEdit_6.setText(str(v[0, 1]))
        self.lineEdit_9.setText(str(v[0, 2]))

        self.lineEdit_4.setText(str(v[1, 0]))
        self.lineEdit_7.setText(str(v[1, 1]))
        self.lineEdit_10.setText(str(v[1, 2]))

        self.lineEdit_5.setText(str(v[2, 0]))
        self.lineEdit_8.setText(str(v[2, 1]))
        self.lineEdit_11.setText(str(v[2, 2]))

    @property
    def translation_vector(self):
        r = np.array([
            float(self.lineEdit_12.text()),
            float(self.lineEdit_13.text()),
            float(self.lineEdit_14.text())
        ], dtype=np.float32)
        return r / 10.0

    @translation_vector.setter
    def translation_vector(self, v):
        self.lineEdit_12.setText(str(v[0]))
        self.lineEdit_13.setText(str(v[1]))
        self.lineEdit_14.setText(str(v[2]))

    @property
    def trajectory_filename(self):
        return str(self.lineEdit.text())

    @trajectory_filename.setter
    def trajectory_filename(self, v):
        self.lineEdit.setText(str(v))

    def __init__(self, **kwargs):
        self.trajectory = None
        self.verbose = kwargs.get('verbose', mfm.verbose)

        QtGui.QWidget.__init__(self)
        uic.loadUi('mfm/ui/tools/rotate_translate_traj.ui', self)
        self.connect(self.actionOpen_trajectory, QtCore.SIGNAL('triggered()'), self.onOpenTrajectory)
        self.connect(self.actionSave_trajectory, QtCore.SIGNAL('triggered()'), self.onSaveTrajectory)

    def onOpenTrajectory(self, filename=None):
        print "onOpenTrajectory"
        self.trajectory_filename = str(QtGui.QFileDialog.getOpenFileName(None, 'Open H5-Model file', '', 'H5-files (*.h5)'))

    def onSaveTrajectory(self, target_filename=None):
        if target_filename is None:
            target_filename = str(QtGui.QFileDialog.getSaveFileName(None, 'Save H5-Model file', '', 'H5-files (*.h5)'))

        translation_vector = self.translation_vector
        rotation_matrix = self.rotation_matrix
        stride = self.stride

        if self.verbose:
            print "Stride: %s" % stride
            print "\nRotation Matrix"
            print rotation_matrix
            print "\nTranslation vector"
            print translation_vector

        first_frame = md.load_frame(self.trajectory_filename, 0)
        traj_new = md.Trajectory(xyz=np.empty((0, first_frame.n_atoms, 3)), topology=first_frame.topology)
        traj_new.save(target_filename)

        chunk_size = 1000
        table = tables.open_file(target_filename, 'a')
        for i, chunk in enumerate(md.iterload(self.trajectory_filename, chunk=chunk_size, stride=stride)):
            xyz = chunk.xyz.copy()
            rotate(xyz, rotation_matrix)
            translate(xyz, translation_vector)
            table.root.xyz.append(xyz)
            table.root.time.append(np.arange(i * chunk_size, i * chunk_size + xyz.shape[0], dtype=np.float32))
        table.close()



class Object(object):
    # This is only used to pass the arguments
    pass


class MDConverter(QtGui.QWidget):


    name = "MC-Converter"

    def __init__(self, parent=None, **kwargs):
        QtGui.QWidget.__init__(self, parent=parent)
        uic.loadUi('./mfm/ui/convert_structures.ui', self)
        self.connect(self.toolButton, QtCore.SIGNAL("clicked()"), self.onLoadHDFFile)
        self.connect(self.toolButton_2, QtCore.SIGNAL("clicked()"), self.onSelecteTargetDir)
        self.connect(self.pushButton_3, QtCore.SIGNAL("clicked()"), self.onConvert)
        self.connect(self.actionOpen_Topology, QtCore.SIGNAL("triggered()"), self.onOpenTopology)
        self.verbose = kwargs.get('verbose', mfm.verbose)

    def onOpenTopology(self):
        self.topology_file = str(QtGui.QFileDialog.getOpenFileName(self, 'Open PDB-File', '.', 'PDB-File (*.pdb)'))

    def onLoadHDFFile(self):
        if not self.use_folder:
            self.trajectory = str(QtGui.QFileDialog.getOpenFileName(self, 'Open HDF-File', '.', 'H5-File (*.h5)'))
        else:
            self.trajectory = str(QtGui.QFileDialog.getExistingDirectory(self, 'Open PDB-Files', '.'))

    def onSelecteTargetDir(self):
        self.target_directory = str(QtGui.QFileDialog.getExistingDirectory(self, 'Choose Target-Folder', '.'))

    def onConvert(self, **kwargs):
        verbose = kwargs.get('verbose', self.verbose)
        if verbose:
            print "Converting trajectory"
        args = Object()
        args.topology = self.topology_file
        args.input = [self.trajectory] if not self.use_folder else glob.glob(os.path.join(self.trajectory, '*.pdb'))
        args.index = None
        args.chunk = 1000
        args.stride = self.stride

        if self.first_frame != 0 or self.last_frame != -1:
            args.index = slice(self.first_frame, self.last_frame, self.stride)
            args.stride = None
            args.chunk = None

        args.force = True
        args.atom_indices = None

        if self.split:
            i = 0
            for chunk in md.iterload(self.trajectory, chunk=args.chunk):
                for s in chunk:
                    try:
                        fn = os.path.join(self.target_directory, self.filename + '_%0*d' % (8, i) + self.ending)
                        if verbose:
                            print fn
                        s.save(fn)
                    except:
                        pass
                    i += 1
        else:
            args.output = os.path.join(self.target_directory, self.filename + self.ending)
            mdconvert.main(args)
        MyMessageBox('Conversion done!')

    @property
    def first_frame(self):
        return int(self.spinBox_2.value())

    @property
    def last_frame(self):
        return int(self.spinBox_3.value())

    @property
    def stride(self):
        return int(self.spinBox.value())

    @property
    def topology_file(self):
        s = str(self.lineEdit_4.text())
        if os.path.isfile(s):
            return s
        else:
            return None

    @topology_file.setter
    def topology_file(self, v):
        return self.lineEdit_4.setText(v)

    @property
    def ending(self):
        return str(self.comboBox.currentText())

    @property
    def split(self):
        return bool(self.checkBox.isChecked())

    @property
    def filename(self):
        return str(self.lineEdit_3.text())

    @property
    def trajectory(self):
        return str(self.lineEdit.text())

    @trajectory.setter
    def trajectory(self, v):
        self.lineEdit.setText(str(v))

    @property
    def target_directory(self):
        return str(self.lineEdit_2.text())

    @target_directory.setter
    def target_directory(self, v):
        self.lineEdit_2.setText(str(v))

    @property
    def use_folder(self):
        return bool(self.radioButton.isChecked())


class RemoveClashedFrames(QtGui.QWidget):

    @property
    def stride(self):
        return int(self.spinBox.value())

    @property
    def atom_list(self):
        txt = str(self.plainTextEdit.toPlainText())
        atom_list = np.fromstring(txt, dtype=np.int32, sep=",")
        return atom_list

    @property
    def trajectory_filename(self):
        return str(self.lineEdit.text())

    @trajectory_filename.setter
    def trajectory_filename(self, v):
        self.lineEdit.setText(str(v))

    @property
    def min_distance(self):
        return float(self.doubleSpinBox.value()) / 10.0

    def onRemoveClashes(self):
        print "onRemoveClashes"
        target_filename = str(QtGui.QFileDialog.getSaveFileName(None, 'Save H5-Model file', '', 'H5-files (*.h5)'))

        filename = self.trajectory_filename
        atom_indices = self.atom_list
        stride = self.stride
        min_distance = self.min_distance

        # Make empty trajectory
        frame_0 = md.load_frame(filename, 0)
        target_traj = md.Trajectory(xyz=np.empty((0, frame_0.n_atoms, 3)), topology=frame_0.topology)
        target_traj.save(target_filename)

        chunk_size = 1000
        table = tables.open_file(target_filename, 'a')
        for i, chunk in enumerate(md.iterload(filename, chunk=chunk_size, stride=stride)):
            xyz = chunk.xyz.copy()

            frames_below = below_min_distance(xyz, min_distance, atom_selection=atom_indices)
            selection = np.where(frames_below < 1)[0]
            xyz_clash_free = np.take(xyz, selection)

            table.root.xyz.append(xyz_clash_free)
            times = np.arange(table.root.time.shape[0],
                              table.root.time.shape[0] + xyz_clash_free.shape[0], dtype=np.float32)
            table.root.time.append(times)
        table.close()

    def onOpenTrajectory(self, filename=None):
        print "onOpenTrajectory 1"
        if filename is None:
            self.trajectory_filename = str(QtGui.QFileDialog.getOpenFileName(None, 'Open H5-Model file', '', 'H5-files (*.h5)'))

    def __init__(self, **kwargs):
        QtGui.QWidget.__init__(self)
        uic.loadUi('./mfm/ui/tools/remove_clashes.ui', self)
        self.connect(self.actionOpen_trajectory, QtCore.SIGNAL('triggered()'), self.onOpenTrajectory)
        self.connect(self.actionSave_clash_free_trajectory, QtCore.SIGNAL('triggered()'), self.onRemoveClashes)

