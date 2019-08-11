# -*- coding: utf-8 -*-
# After updating of icon run:
# pyrcc4 -o rescource_rc.py rescource.qrc
from multiprocessing import freeze_support
import sip
try:
    sip.setapi('QDate', 2)
    sip.setapi('QDateTime', 2)
    sip.setapi('QString', 2)
    sip.setapi('QTextStream', 2)
    sip.setapi('QTime', 2)
    sip.setapi('QUrl', 2)
    sip.setapi('QVariant', 2)
except ValueError, e:
    raise RuntimeError('Could not set API version (%s): did you import PyQt4 directly?' % e)
from PyQt4 import QtGui, QtCore, uic
import sys
import numpy as np
import mfm
import mfm.ui.rescource_rc
from mfm import experiments, FitSubWindow
import time
from mfm.io.txt_csv import CSVFileWidget


def excepthook(excType, excValue, tracebackobj):
    """
    http://www.riverbankcomputing.com/pipermail/pyqt/2009-May/022961.html
    Global function to catch unhandled exceptions.

    @param excType exception type
    @param excValue exception value
    @param tracebackobj traceback object
    """
    separator = '-' * 80
    logFile = "error_log.log"
    email = mfm.settings["email-contact"]
    notice = \
        """An unhandled exception occurred. Please report the problem\n"""\
        """using the error reporting dialog or via email to <%s>.\n"""\
        """A log has been written to "%s".\n\nError information:\n""" % \
        (email, logFile)
    versionInfo = mfm.__version__
    timeString = time.strftime("%Y-%m-%d, %H:%M:%S")

    errmsg = '%s: \n%s' % (str(excType), str(excValue))
    sections = [separator, timeString, separator, errmsg, separator]
    msg = '\n'.join(sections)
    try:
        f = open(logFile, "w")
        f.write(msg)
        f.write(versionInfo)
        f.close()
    except IOError:
        pass

    if not mfm.settings['quiet_exceptions']:
        errorbox = QtGui.QMessageBox()
        errorbox.setText(str(notice)+str(msg)+str(versionInfo))
        errorbox.exec_()


sys.excepthook = excepthook


class Main(QtGui.QMainWindow):
    """ Main window
    The program is structured in a tree
    self.rootNode -> n * Experiment ->  setup -> datasets -> Fit -> Model
    """

    def doTileMdiWindows(self):
        self.mdiarea.setViewMode(QtGui.QMdiArea.SubWindowView)
        self.mdiarea.tileSubWindows()

    def doTabMdiWindows(self):
        self.mdiarea.setViewMode(QtGui.QMdiArea.TabbedView)

    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        uic.loadUi("mfm/ui/mainwindow.ui", self)
        self.setCentralWidget(self.mdiarea)

        ## Last directory used to save fits
        self.last_save_directory = None

        ## Helping Widgets
        self.calculate_potential = mfm.tools.PotentialEnergyWidget()
        self.lifetime_calc = mfm.tools.LifetimeCalculator()
        self.fret_lines = mfm.tools.fret_lines.FRETLineGeneratorWidget()
        self.kappa2_dist = mfm.tools.kappa2dist.Kappa2Dist()
        self.structure2transfer = mfm.tools.Structure2Transfer()
        self.decay_generator = mfm.tools.dye_diffusion.TransientDecayGenerator()
        #self.decay_fret_generator = mfm.fluorescence.dye_diffusion.TransientFRETDecayGenerator()
        self.hdf2pdb = mfm.tools.MDConverter()
        self.pdb2label = mfm.tools.PDB2Label()
        self.trajectory_rot_trans = mfm.tools.RotateTranslateTrajectoryWidget()
        self.join_trajectories = mfm.tools.JoinTrajectoriesWidget()
        self.align_trajectory = mfm.tools.AlignTrajectoryWidget()
        self.remove_clashes = mfm.tools.RemoveClashedFrames()
        self.traj_save_topol = mfm.tools.SaveTopology()

        self.about = uic.loadUi("mfm/ui/about.ui")
        self.about.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.about.hide()

        #add_console
        if mfm.settings['hide_console']:
            self.dockWidget_console.hide()
        self.console = mfm.widgets.QIPythonWidget()
        self.console.pushVariables({'mfm': mfm})
        self.console.pushVariables({'np': np})
        self.console.pushVariables({'cs': self})

        self.console.width = 50
        self.verticalLayout_4.addWidget(self.console)

        ## Arrange Docks and window positions
        self.tabifyDockWidget(self.dockWidget_load, self.dockWidget_fit)
        self.tabifyDockWidget(self.dockWidget_fit, self.dockWidget_plot)
        self.tabifyDockWidget(self.dockWidget_plot, self.dockWidget_console)
        self.modelLayout.setAlignment(QtCore.Qt.AlignTop)
        self.plotOptionsLayout.setAlignment(QtCore.Qt.AlignTop)
        self.dockWidget_load.raise_()

        ## Center windows in the middle of the screen
        screen = QtGui.QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width()-size.width())/2, (screen.height()-size.height())/2)

        ##### Slots and Actions #####
        self.mdiarea.subWindowActivated.connect(self.on_fit_changed)
        self.connect(self.comboBox_experimentSelect, QtCore.SIGNAL("currentIndexChanged(int)"), self.on_experiment_changed)
        self.connect(self.comboBox_setupSelect, QtCore.SIGNAL("currentIndexChanged(int)"), self.on_setup_changed)
        self.connect(self.actionChange_current_dataset, QtCore.SIGNAL('triggered()'), self.onCurrentDatasetChanged)
        self.connect(self.actionAdd_fit, QtCore.SIGNAL('triggered()'), self.on_add_fit)
        self.connect(self.actionSaveAllFits, QtCore.SIGNAL('triggered()'), self.onSaveAllFits)
        self.connect(self.actionSaveCurrentFit, QtCore.SIGNAL('triggered()'), self.onSaveCurrentFit)
        self.connect(self.actionClose_Fit, QtCore.SIGNAL('triggered()'), self.on_close_current_fit)
        self.connect(self.actionClose_all_fits, QtCore.SIGNAL('triggered()'), self.on_close_all_fits)
        self.connect(self.actionLoad_Data, QtCore.SIGNAL('triggered()'), self.on_add_dataset)
        self.connect(self.actionDelete_dataset, QtCore.SIGNAL('triggered()'), self.onDeleteDataset)
        # Help/About
        self.connect(self.actionHelp_2, QtCore.SIGNAL('triggered()'), self.about.show)
        self.connect(self.actionAbout, QtCore.SIGNAL('triggered()'), self.about.show)
        # Window-controls tile, stack etc.
        self.connect(self.actionTile_windows, QtCore.SIGNAL('triggered()'), self.doTileMdiWindows)
        self.connect(self.actionTab_windows, QtCore.SIGNAL('triggered()'), self.doTabMdiWindows)
        self.mdiarea.subWindowActivated.connect(self.subWindowActivated)

        # Tools
        self.connect(self.actionRotate_Translate_trajectory, QtCore.SIGNAL('triggered()'), self.trajectory_rot_trans.show)
        self.connect(self.actionTrajectory_converter, QtCore.SIGNAL('triggered()'), self.hdf2pdb.show)
        self.connect(self.actionCalculate_Potential, QtCore.SIGNAL('triggered()'), self.calculate_potential.show)
        self.connect(self.actionDye_Diffusion, QtCore.SIGNAL('triggered()'), self.decay_generator.show)
        self.connect(self.actionPDB2Label, QtCore.SIGNAL('triggered()'), self.pdb2label.show)
        self.connect(self.actionStructure2Transfer, QtCore.SIGNAL('triggered()'), self.structure2transfer.show)
        self.connect(self.actionKappa2_Distribution, QtCore.SIGNAL('triggered()'), self.kappa2_dist.show)
        self.connect(self.actionFRET_Lines, QtCore.SIGNAL('triggered()'), self.fret_lines.show)
        self.connect(self.actionCalculator, QtCore.SIGNAL('triggered()'), self.lifetime_calc.show)
        self.connect(self.actionJoin_trajectories, QtCore.SIGNAL('triggered()'), self.join_trajectories.show)
        self.connect(self.actionRemove_clashes, QtCore.SIGNAL('triggered()'), self.remove_clashes.show)
        self.connect(self.actionAlign_trajectory, QtCore.SIGNAL('triggered()'), self.align_trajectory.show)
        self.connect(self.actionSave_topology, QtCore.SIGNAL('triggered()'), self.traj_save_topol.show)

        tcspc = experiments.Experiment('TCSPC')
        tcspc.link(mfm.rootNode)
        tcspc.add_setup(mfm.experiments.TCSPCSetupWidget())
        tcspc.add_setup(mfm.experiments.TCSPCSetupTTTRWidget())
        tcspc.add_setup(mfm.experiments.TCSPCSetupSDTWidget())
        tcspc.add_setup(mfm.experiments.TCSPCSetupDummyWidget())
        tcspc.add_models(mfm.fitting.models.tcspc.models)

        fcs = experiments.Experiment('FCS')
        fcs.link(mfm.rootNode)
        fcs.add_setup(mfm.experiments.FCSCsv(name='csv-files', experiment=fcs))
        fcs.add_setup(mfm.experiments.FCStttr(name='tttr-files', experiment=fcs))
        fcs.add_model(mfm.fitting.models.fcs.ParseFCSWidget)

        stopped_flow = experiments.Experiment('Stopped-Flow')
        stopped_flow.link(mfm.rootNode)
        stopped_flow.add_setup(CSVFileWidget(name='Stopped-Flow', weight_calculation=np.sqrt,
                                                           skiprows=0, use_header=True))
        stopped_flow.add_model(mfm.fitting.models.stopped_flow.ParseStoppedFlowWidget)
        stopped_flow.add_model(mfm.fitting.models.stopped_flow.ReactionWidget)

        modelling = experiments.Experiment('Modelling')
        modelling.link(mfm.rootNode)
        # modelling.addSetup(e.setups.modelling.LoadStructureFolder())
        modelling.add_setup(mfm.experiments.modelling.LoadStructure())
        modelling.add_model(mfm.fitting.models.proteinMC.ProteinMCWidget)

        globalFit = experiments.Experiment('Global')
        globalFit.link(mfm.rootNode)
        globalFit.add_model(mfm.fitting.models.GlobalFitModelWidget)
        #globalFit.add_model(e.models.tcspc.GlobalLifetimeAnisotropyWidget)
        globalFit.add_model(mfm.fitting.models.tcspc.et.EtModelFreeWidget)
        globalFit.add_setup(mfm.experiments.GlobalFitSetup('Global-Fit'))

        self.fit = None
        self.experiment_names = [b.name for b in mfm.rootNode if b.name is not 'Global']
        self.data_sets = []
        self.fit_windows = []
        self.comboBox_experimentSelect.addItems(self.experiment_names)

        # Add Global-Dataset by default
        self.on_add_dataset(experiment=globalFit, setup=globalFit.get_setups()[0])

    def subWindowActivated(self):
        """
        This gets called if a fit (a subwindow) is activated.
        Right now this:
            1) updated the window title of the main application
        """
        filename = "" if self.current_fit is None else self.current_fit.name
        window_title = "ChiSurf: " + filename
        self.setWindowTitle(window_title)

    def closeEvent(self, event):
        reply = QtGui.QMessageBox.question(self, 'Message',
            "Are you sure to quit?", QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)
        if reply == QtGui.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def onSaveAllFits(self, **kwargs):
        print("onSaveAllFits")
        last_directory = kwargs.get('directory', self.last_save_directory)
        if isinstance(last_directory, str):
            directory = str(QtGui.QFileDialog.getExistingDirectory(self, "Select Directory", last_directory))
        else:
            directory = str(QtGui.QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.last_save_directory = directory

        for i, fit in enumerate(mfm.fits):
            name = "fit_%s" % i
            fit.save(directory, name)

    def onSaveCurrentFit(self):
        directory = str(QtGui.QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.current_fit.save(directory, 'fit')

    @property
    def fits(self):
        return [w.widget().fit for w in self.mdiarea.subWindowList()]

    @property
    def current_model_nrb(self):
        return self.comboBox_Model.currentIndex()

    @current_model_nrb.setter
    def current_model_nrb(self, v):
        self.comboBox_Model.setCurrentIndex(v)

    @property
    def current_dataset_index(self):
        return int(self.tableWidget.currentRow())

    @current_dataset_index.setter
    def current_dataset_index(self, v):
        self.tableWidget.setCurrentCell(v, 0)
        self.actionChange_current_dataset.trigger()

    @property
    def current_dataset(self):
        return self.data_sets[self.current_dataset_index]

    @property
    def current_model_class(self):
        return self.current_dataset.experiment.model_classes[self.current_model_nrb]

    @property
    def current_fit(self):
        try:
            fit = self.current_fit_widget.fit
        except AttributeError:
            fit = None
        return fit

    @property
    def current_fit_widget(self):
        try:
            return self.current_sub_window.widget()
        except AttributeError:
            return None

    @property
    def current_sub_window(self):
        return self.mdiarea.currentSubWindow()

    @property
    def current_plot_tab(self):
        return self.mdiarea.currentSubWindow().widget()

    @property
    def current_plot(self):
        idx = self.current_plot_tab.currentIndex()
        plots = self.current_fit.model.plots
        return plots[idx]

    @property
    def current_plot_control(self):
        return self.current_plot.pltControl

    @property
    def current_experiment(self):
        """
        The currently selected experiment type. This can also be set via the console

        Examples
        --------

        In the interactive console of the program the experiment can be set like this

        >>> chisurf.current_experiment = 'FCS'

        """
        return mfm.rootNode[self.comboBox_experimentSelect.currentIndex()]

    @current_experiment.setter
    def current_experiment(self, v):
        index = self.experiment_names.index(v)
        self.comboBox_experimentSelect.setCurrentIndex(index)

    @property
    def current_setup(self):
        return self.current_experiment.get_setups()[self.comboBox_setupSelect.currentIndex()]

    def onDeleteDataset(self, **kwargs):
        index = kwargs.get('index', self.current_dataset_index)
        dataSets = self.data_sets
        if len(dataSets) > 0:
            ds = dataSets[index]
            if ds.name != 'Global-fit':
                dataSets.pop(index)
                # delete all fits of dataset
                dsFits = [f for f in ds.get_descendants() if isinstance(f, mfm.Fit)]
                for idx, f in enumerate(mfm.fits):
                    if f in dsFits:
                        self.on_close_current_fit(idx)
                ds.unlink()
                self.tableWidget.removeRow(index)

    def onCurrentDatasetChanged(self):
        row = self.tableWidget.currentRow()
        # color/highlight clicked row
        for r in range(self.tableWidget.rowCount()):
            if r == row:
                brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
            else:
                brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
            brush.setStyle(QtCore.Qt.SolidPattern)
            self.tableWidget.item(r, 0).setBackground(brush)
            self.tableWidget.item(r, 1).setBackground(brush)

        self.comboBox_Model.clear()
        if len(self.data_sets) > 0:
            ds = self.data_sets[row]
            modelNames = ds.experiment.get_model_names()
            self.comboBox_Model.addItems(modelNames)

    def on_add_fit(self):
        fit = mfm.FitQtThread(data=self.current_dataset)
        fit.link(self.current_dataset)
        mfm.fits.append(fit)
        # make new mode
        fit.model = self.current_model_class(fit=fit)
        self.modelLayout.addWidget(fit.model)

        fit_widget = FitSubWindow(fit, self)
        fit.parent = fit_widget
        fit_window = self.mdiarea.addSubWindow(fit_widget)

        try:
            print type(fit.model.icon)
            fit_window.setWindowIcon(fit.model.icon)
        except AttributeError:
            pass

        self.fit_windows.append(fit_window)
        fit_window.show()

    def onPlotChanged(self):
        mfm.widgets.hide_items_in_layout(self.plotOptionsLayout)
        try:
            self.current_plot_control.show()
        except AttributeError("Plot %s does not implement control widget in attribute pltControl."):
            pass

    def on_experiment_changed(self):
        self.comboBox_setupSelect.blockSignals(True)
        self.comboBox_setupSelect.clear()
        self.comboBox_setupSelect.addItems(self.current_experiment.setup_names)
        self.comboBox_setupSelect.blockSignals(False)
        self.on_setup_changed()

    def on_close_all_fits(self):
        for sub_window in self.fit_windows:
            sub_window.widget().confirm = False
            sub_window.widget().fit.unlink()
            sub_window.close()
        mfm.fits = []

    def on_close_current_fit(self, idx=None):
        """
        This method removes either the currently selected fit or the fit with the index provided by the
        parameter `idx`

        :param idx: int
        """
        if idx is None:
            sub_window = self.current_sub_window
        else:
            sub_window = self.fit_windows[idx]
        if not None:
            sub_window.widget().fit.unlink()
            sub_window.close()
            mfm.fits = [w.widget().fit for w in self.mdiarea.subWindowList()]

    def on_fit_changed(self):
        if self.fit is not None:
            self.fit.model.hide()
        if self.current_fit is not None:
            self.fit = self.current_fit
            self.fit.model.show()
        self.onPlotChanged()

    def on_setup_changed(self):
        mfm.widgets.hide_items_in_layout(self.verticalLayout_5)
        self.verticalLayout_5.addWidget(self.current_setup)
        self.current_setup.show()

    def on_add_dataset(self, experiment=None, setup=None, filename=None):
        if setup is None:
            setup = self.current_setup
        if experiment is None:
            experiment = self.current_experiment

        data_set = setup.get_dataset(experiment=experiment, filename=filename)

        self.data_sets.append(data_set)

        row = self.tableWidget.rowCount()
        self.tableWidget.setRowCount(row + 1)

        tmp = QtGui.QTableWidgetItem()
        tmp.setText(setup.name)
        tmp.setFlags(QtCore.Qt.ItemIsEnabled)
        self.tableWidget.setItem(row, 0, tmp)

        tmp = QtGui.QTableWidgetItem()

        fn = data_set.name
        fn = fn.replace('/', '/ ')
        fn = fn.replace('\\', '\\ ')
        tmp.setText(fn)
        tmp.setFlags(QtCore.Qt.ItemIsEnabled)
        self.tableWidget.setItem(row, 1, tmp)

        self.tableWidget.resizeRowsToContents()
        self.tableWidget.setWordWrap(True)
        header = self.tableWidget.horizontalHeader()
        header.setStretchLastSection(True)

if __name__ == "__main__":
    freeze_support()
    app = QtGui.QApplication(sys.argv)

    win = Main()
    win.show()
    sys.exit(app.exec_())


