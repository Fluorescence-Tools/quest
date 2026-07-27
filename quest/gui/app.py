"""The standalone desktop application.

QuEst is normally a ChiSurf plugin; this is the same widget in a window of its
own, for people who do not run ChiSurf. The menu bar is the only thing the
plugin does not provide.
"""

from __future__ import annotations

import sys
from pathlib import Path

from qtpy import QtGui, QtWidgets

from ..i18n import resolve_locale, ui
from .dye_widget import TransientDecayGenerator

__all__ = ["ICON_PATH", "CITATION", "QuEstWindow", "start_gui"]

#: The window icon. All that survived the Qt resource module (`rescource_rc.py`,
#: 98 kB of generated base64) — one file loaded from disk needs no build step.
ICON_PATH = Path(__file__).resolve().parent / "icons" / "dye-diffusion.ico"

#: The reference itself is not translated — a citation is copied verbatim into a
#: manuscript, so changing it per language would be wrong.
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
    """A window around :class:`TransientDecayGenerator`."""

    def __init__(self, parent=None, locale: str | None = None) -> None:
        super().__init__(parent)
        self.locale = resolve_locale(locale)
        self.dg = TransientDecayGenerator(locale=self.locale)
        self.setCentralWidget(self.dg)
        self.setWindowTitle(self._t("app.title"))
        if ICON_PATH.is_file():
            self.setWindowIcon(QtGui.QIcon(str(ICON_PATH)))
        self.resize(1400, 900)
        self._create_menus()

    def _t(self, key: str) -> str:
        return ui(key, self.locale)

    def _create_menus(self) -> None:
        file_menu = self.menuBar().addMenu(self._t("menu.file"))
        file_menu.addAction(self._t("menu.load_structure")).triggered.connect(
            self.dg.onLoadPDB
        )
        file_menu.addSeparator()
        file_menu.addAction(self._t("menu.save_project")).triggered.connect(
            self.dg.onSaveProject
        )
        file_menu.addAction(self._t("menu.load_project")).triggered.connect(
            self.dg.onLoadProject
        )
        file_menu.addSeparator()
        file_menu.addAction(self._t("menu.exit")).triggered.connect(self.close)

        help_menu = self.menuBar().addMenu(self._t("menu.help"))
        help_menu.addAction(self._t("menu.about")).triggered.connect(self._show_about)

    def _show_about(self) -> None:
        body = (
            f"<h3>{self._t('app.title')} — {self._t('app.subtitle')}</h3>"
            f"<p>{self._t('about.cite')}</p>{CITATION}"
        )
        QtWidgets.QMessageBox.about(self, self._t("about.title"), body)


def start_gui() -> int:
    """Entry point for `quest gui` and the `cs.quest` script."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = QuEstWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(start_gui())
