"""The standalone desktop application.

Standard Qt desktop GUI application for QuEst with native system controls.
"""

from __future__ import annotations

import sys
import webbrowser
from pathlib import Path

from qtpy import QtGui, QtWidgets

from ..i18n import resolve_locale, ui
from .dye_widget import TransientDecayGenerator
from .theme import apply_theme

__all__ = ["ICON_PATH", "CITATION", "QuEstWindow", "start_gui"]

ICON_PATH = Path(__file__).resolve().parent / "icons" / "dye-diffusion.ico"

CITATION = (
    "<p><b>Combining Graphical and Analytical Methods with Molecular Simulations "
    "To Analyze Time-Resolved FRET Measurements of Labeled Macromolecules "
    "Accurately.</b><br/>"
    "Peulen, T.O., Opanasyuk, O., and Seidel, C.A.M., 2017.<br/>"
    "<i>The Journal of Physical Chemistry B</i> 121 (35), 8211–8241.<br/>"
    "DOI: <a href='https://doi.org/10.1021/acs.jpcb.7b03441'>"
    "10.1021/acs.jpcb.7b03441</a></p>"
)


class QuEstWindow(QtWidgets.QMainWindow):
    """Standard Qt Desktop Application Window for QuEst."""

    def __init__(self, parent=None, locale: str | None = None) -> None:
        super().__init__(parent)
        self.locale = resolve_locale(locale)

        self.dg = TransientDecayGenerator(locale=self.locale)
        self.setCentralWidget(self.dg)
        self.setWindowTitle(f"{self._t('app.title')} — Desktop Workspace")

        if ICON_PATH.is_file():
            self.setWindowIcon(QtGui.QIcon(str(ICON_PATH)))
        self.resize(1280, 850)

        # Apply standard native Qt styling
        apply_theme(self)

        self._create_toolbar()
        self._create_menus()
        self._create_statusbar()

    def _t(self, key: str) -> str:
        return ui(key, self.locale)

    def _create_toolbar(self) -> None:
        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)

        action_run = toolbar.addAction("Run Simulation")
        action_run.setShortcut("Ctrl+R")
        action_run.triggered.connect(self.dg.update_all)

        toolbar.addSeparator()

        action_pdb = toolbar.addAction("Load Structure...")
        action_pdb.triggered.connect(self.dg.onLoadPDB)

        action_save = toolbar.addAction("Save Project...")
        action_save.triggered.connect(self.dg.onSaveProject)

        action_load = toolbar.addAction("Load Project...")
        action_load.triggered.connect(self.dg.onLoadProject)

        toolbar.addSeparator()

        action_guide = toolbar.addAction("Guide")
        action_guide.triggered.connect(self._run_guide)

        action_help = toolbar.addAction("Help")
        action_help.triggered.connect(self._show_help)

        action_web = toolbar.addAction("Open Web UI")
        action_web.triggered.connect(self._open_web_ui)

    def _create_menus(self) -> None:
        mb = self.menuBar()

        # File Menu
        file_menu = mb.addMenu(self._t("menu.file"))
        file_menu.addAction(self._t("menu.load_structure") + "...").triggered.connect(
            self.dg.onLoadPDB
        )
        file_menu.addSeparator()
        file_menu.addAction(self._t("menu.save_project") + "...").triggered.connect(
            self.dg.onSaveProject
        )
        file_menu.addAction(self._t("menu.load_project") + "...").triggered.connect(
            self.dg.onLoadProject
        )
        file_menu.addSeparator()
        file_menu.addAction(self._t("menu.exit")).triggered.connect(self.close)

        # Simulation Menu
        sim_menu = mb.addMenu("Simulation")
        action_run = sim_menu.addAction("Run Simulation")
        action_run.setShortcut("Ctrl+R")
        action_run.triggered.connect(self.dg.update_all)

        # Help Menu
        help_menu = mb.addMenu(self._t("menu.help"))
        help_menu.addAction("Guide").triggered.connect(self._run_guide)
        help_menu.addAction("Help / Documentation").triggered.connect(self._show_help)
        help_menu.addSeparator()
        help_menu.addAction("Open Web Interface").triggered.connect(self._open_web_ui)
        help_menu.addSeparator()
        help_menu.addAction(self._t("menu.about")).triggered.connect(self._show_about)

    def _run_guide(self) -> None:
        from .guided_tour import start_guided_tour
        start_guided_tour(self.dg, model=self.dg.model)

    def _show_help(self) -> None:
        from qtpy import QtWidgets
        QtWidgets.QMessageBox.information(
            self,
            "QuEst Help & Documentation",
            "<b>QuEst Tethered Dye Brownian Dynamics Simulator</b><br/><br/>"
            "• <b>Guide</b>: Starts the step-by-step interactive workspace tour.<br/>"
            "• <b>Load Structure</b>: Select a PDB macromolecular structure.<br/>"
            "• <b>Run Simulation</b>: Compute dye diffusion, PET quenching, and decay histograms.<br/>"
            "• <b>Web UI</b>: Open the local web interface on http://localhost:3000."
        )



    def _create_statusbar(self) -> None:
        sb = self.statusBar()
        sb.showMessage("QuEst Ready — Brownian dynamics tethered dye simulation")

    def _open_web_ui(self) -> None:
        webbrowser.open("http://localhost:3000")

    def _show_about(self) -> None:
        body = (
            f"<h3>{self._t('app.title')} — {self._t('app.subtitle')}</h3>"
            f"<p>{self._t('about.cite')}</p>{CITATION}"
        )
        QtWidgets.QMessageBox.about(self, self._t("about.title"), body)


def start_gui() -> int:
    """Entry point for `quest gui` and `cs.quest`."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = QuEstWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(start_gui())
