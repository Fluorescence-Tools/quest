"""Standard Native Qt styling for QuEst desktop application.

Uses native system Qt styles (Fusion/native OS) with clean, standard desktop layout.
"""

from __future__ import annotations

from qtpy import QtWidgets


def apply_theme(app_or_widget, dark: bool = False) -> None:
    """Apply standard clean Qt desktop styling."""
    if isinstance(app_or_widget, QtWidgets.QApplication):
        app_or_widget.setStyle("Fusion")
    elif app_or_widget.window() and app_or_widget.window().inherits("QMainWindow"):
        QtWidgets.QApplication.setStyle("Fusion")

    # Clear custom QSS overrides to restore native Qt desktop controls
    app_or_widget.setStyleSheet("")
