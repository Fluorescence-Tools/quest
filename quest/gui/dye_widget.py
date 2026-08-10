"""QuEst's desktop form.

Connects AutoForm controls, pyqtgraph visualizations, interactive quenching tables,
and simulation actions into a unified desktop workspace.
"""

from __future__ import annotations

import json
from pathlib import Path

from qtpy import QtCore, QtWidgets

from ..i18n import ui
from .form_model import ProjectFormModel
from .plots import QuEstPlotCanvas
from .quencher_widget import QuencherTableWidget

__all__ = ["TransientDecayGenerator"]

try:
    from ..hosts import enable_chisurf

    enable_chisurf()
except Exception:  # pragma: no cover
    pass


class TransientDecayGenerator(QtWidgets.QWidget):
    """The QuEst desktop workspace: AutoForm parameter controls + pyqtgraph visualizations.

    Parameters
    ----------
    parent : QWidget, optional
    project : dict, optional
        Initial project dict. Defaults to template_project().
    locale : str, optional
        Language of the form. Defaults to resolve_locale().
    """

    def __init__(
        self, parent=None, project: dict | None = None, locale: str | None = None
    ) -> None:
        super().__init__(parent)
        self.model = ProjectFormModel(project, locale=locale)
        self.form = None

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # Splitter: Left = AutoForm Controls, Right = Visualization Center
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self)
        self.splitter.setHandleWidth(6)
        self.splitter.setChildrenCollapsible(False)
        main_layout.addWidget(self.splitter)

        # Left AutoForm Panel
        left_container = QtWidgets.QWidget(self)
        left_layout = QtWidgets.QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Action Header
        header_box = QtWidgets.QHBoxLayout()
        self.btn_run = QtWidgets.QPushButton("Run Simulation", left_container)
        self.btn_run.clicked.connect(self.update_all)
        header_box.addWidget(self.btn_run)

        self.btn_cancel = QtWidgets.QPushButton(self._t("action.cancel"), left_container)
        self.btn_cancel.clicked.connect(self.cancel_run)
        self.btn_cancel.setEnabled(False)
        header_box.addWidget(self.btn_cancel)

        btn_pdb = QtWidgets.QPushButton("Load PDB...", left_container)
        btn_pdb.clicked.connect(self.onLoadPDB)
        header_box.addWidget(btn_pdb)

        btn_guide = QtWidgets.QPushButton("Guide", left_container)
        btn_guide.clicked.connect(self.onGuide)
        header_box.addWidget(btn_guide)

        btn_help = QtWidgets.QPushButton("Help", left_container)
        btn_help.clicked.connect(self.onHelp)
        header_box.addWidget(btn_help)

        left_layout.addLayout(header_box)



        try:
            from chisurf.gui.autoform.auto_form import AutoForm

            self.form = AutoForm(self.model)
            left_layout.addWidget(self.form)
        except Exception:
            from .autoform_panel import InTreePyQtAutoForm

            self.form = InTreePyQtAutoForm(self.model)
            left_layout.addWidget(self.form)

        self.splitter.addWidget(left_container)

        # Right Visualization Center Tabs
        self.tabs = QtWidgets.QTabWidget(self)

        # Tab 1: Decay & Dynamics Plots (Separated by QSplitters for flexible scaling)
        plots_widget = QtWidgets.QWidget(self)
        plots_layout = QtWidgets.QVBoxLayout(plots_widget)
        plots_layout.setContentsMargins(2, 2, 2, 2)

        self.plot_vsplitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, plots_widget)
        self.plot_vsplitter.setHandleWidth(6)
        self.plot_vsplitter.setChildrenCollapsible(False)

        self.decay_plot = QuEstPlotCanvas(
            title="Fluorescence Decay (Donor & FRET)",
            x_label="Time (ns)",
            y_label="Counts",
            log_y=True,
        )
        self.plot_vsplitter.addWidget(self.decay_plot)

        self.plot_hsplitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, plots_widget)
        self.plot_hsplitter.setHandleWidth(6)
        self.plot_hsplitter.setChildrenCollapsible(False)

        self.traj_plot = QuEstPlotCanvas(
            title="Dye Trajectory", x_label="Time (ns)", y_label="|r − ⟨r⟩| (Å)"
        )
        self.acf_plot = QuEstPlotCanvas(
            title="Position Autocorrelation (ACF)",
            x_label="Lag (ns)",
            y_label="ACF",
            log_x=True,
        )
        self.plot_hsplitter.addWidget(self.traj_plot)
        self.plot_hsplitter.addWidget(self.acf_plot)

        self.plot_vsplitter.addWidget(self.plot_hsplitter)
        self.plot_vsplitter.setSizes([450, 350])
        self.plot_hsplitter.setSizes([400, 400])

        plots_layout.addWidget(self.plot_vsplitter)
        self.tabs.addTab(plots_widget, "Plots & Dynamics")



        # Tab 2: 3D Molecular Structure (ChiMol)
        from .chimol_widget import ChiMolStructureWidget

        self.chimol_widget = ChiMolStructureWidget(self.model, self)
        self.chimol_widget.set_selection_callback(self._on_residue_picked)
        self.tabs.addTab(self.chimol_widget, "3D Structure (ChiMol)")

        # Tab 3: Quenching Table
        self.quencher_widget = QuencherTableWidget(self.model, self)
        self.tabs.addTab(self.quencher_widget, "Quenching Chemistry")

        # Tab 4: Project JSON State — ChiSurf's SimpleCodeEditor (JSON highlight)
        json_widget = QtWidgets.QWidget(self)
        json_layout = QtWidgets.QVBoxLayout(json_widget)
        json_layout.setContentsMargins(0, 0, 0, 0)
        try:
            from chisurf.plugins.core.code_editor import SimpleCodeEditor
            self.json_edit = SimpleCodeEditor(parent=json_widget, language="JSON")
            self.json_edit.setReadOnly(True)
        except Exception:
            self.json_edit = QtWidgets.QPlainTextEdit(json_widget)
            self.json_edit.setReadOnly(True)
            self.json_edit.setStyleSheet("font-family: monospace; font-size: 11px;")
        json_layout.addWidget(self.json_edit)
        self.tabs.addTab(json_widget, "Project JSON")

        self.splitter.addWidget(self.tabs)

        # Set initial splitter sizes (40% form, 60% visualization)
        self.splitter.setSizes([500, 800])

        self.refresh()

    def _t(self, key: str) -> str:
        return ui(key, self.locale)

    # -- the project, in and out -------------------------------------------

    def to_project_dict(self) -> dict:
        """The current project — the same JSON the CLI and web UI consume."""
        return self.model.to_project_dict()

    def load_project_dict(self, project: dict) -> None:
        """Replace the state, then redraw the controls to match."""
        self.model.load_project_dict(project)
        self.refresh()

    def refresh(self) -> None:
        """Re-read the model into controls, plots and visualization tabs."""
        if self.form is not None:
            self.form.sync_fields()
            self.form.refresh_plots()

        # Refresh right panel plots
        self.decay_plot.set_data(self.model.decay_series())
        self.traj_plot.set_data(self.model.trajectory_series())
        self.acf_plot.set_data(self.model.autocorrelation_series())

        # Refresh quencher table
        if hasattr(self, "quencher_widget"):
            self.quencher_widget.refresh()

        # Refresh ChiMol 3D structure viewer
        if hasattr(self, "chimol_widget"):
            self.chimol_widget.reload()


        # Refresh JSON view
        if hasattr(self, "json_edit"):
            self.json_edit.setPlainText(
                json.dumps(self.model.to_project_dict(), indent=2)
            )

    # -- menu actions ------------------------------------------------------

    def onLoadPDB(self) -> str | None:
        """Pick a structure, then auto-select a valid attachment site."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self._t("dialog.open_structure"),
            "",
            f"{self._t('filter.structures')};;{self._t('filter.all')}",
        )
        if not path:
            return None
        self.model.pdb = path
        self.model.auto_attach_to_structure()
        self.model.status = self._t("status.loaded").format(name=Path(path).name)
        self.refresh()
        return path

    def onSaveProject(self):
        result = self.model.save_project_dialog()
        self.refresh()
        return result

    def onLoadProject(self):
        result = self.model.load_project_dialog()
        self.refresh()
        return result

    def onGuide(self):
        """Start interactive guided tour."""
        from .guided_tour import start_guided_tour
        return start_guided_tour(self, model=self.model)

    def onHelp(self):
        """Show parameter catalog / help dialog."""
        QtWidgets.QMessageBox.information(
            self,
            "QuEst Help",
            "<b>QuEst Help & Parameters</b><br/><br/>"
            "• Configure dye parameters, attachment chain/residue/atom.<br/>"
            "• Quenching chemistry table defines kQ and contact radii per residue.<br/>"
            "• Press <b>Run Simulation</b> to calculate fluorescence decay."
        )


    def update_all(self):
        """Simulate off the GUI thread, then show the result.

        The run used to be a plain call on the GUI thread, which froze the
        window for its whole duration and could not be stopped. It is a job
        now ([`quest.tasks`](../tasks.py)); this method pumps the event queue
        while waiting, so the window keeps painting and the Cancel button
        keeps responding, and still returns the result to its caller.

        **What this does and does not buy**: the trajectory kernels are
        `njit(nogil=True)` — deliberately, so whole trajectories can run
        concurrently — so the GUI thread really does get the interpreter back
        during the dominant cost. The AV grid kernels are `parallel=True`
        without `nogil` and still hold it, so a very large volume can stutter.
        That is a smaller freeze than the whole run, not none.

        Re-entrant clicks are ignored rather than queued: pumping events means
        this method can be re-entered from the button that started it.
        """
        if self.model.simulation_running:
            return None

        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        try:
            self.model.start_simulation()
            while not self.model.wait_simulation(timeout=0.05):
                QtWidgets.QApplication.processEvents()
            self.model.poll_simulation()
        finally:
            self.btn_run.setEnabled(True)
            self.btn_cancel.setEnabled(False)

        self.refresh()
        return self.model.results

    def cancel_run(self):
        """Ask the running simulation to stop.

        Cooperative, and a single simulation has essentially one checkpoint —
        the result is discarded when the kernel returns rather than the
        computation being interrupted. `ProjectFormModel.cancel_simulation`
        says why.
        """
        return self.model.cancel_simulation()

    def _on_residue_picked(self):
        """Called when a residue is selected in the ChiMol viewer or sequence bar.

        Syncs the form controls to match the new attachment without reloading
        the 3D structure (the viewer is already showing it).
        """
        if self.form is not None:
            self.form.sync_fields()
        if hasattr(self, "json_edit"):
            self.json_edit.setPlainText(
                json.dumps(self.model.to_project_dict(), indent=2)
            )

