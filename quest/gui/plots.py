"""Interactive pyqtgraph plot widgets for QuEst desktop GUI.

Renders decay histograms, dye trajectories, autocorrelation curves, and accessible volume
profiles using fast C++ pyqtgraph rendering with theme integration.
"""

from __future__ import annotations

from typing import Any
from qtpy import QtCore, QtWidgets

try:
    import pyqtgraph as pg

    # Configure pyqtgraph global options for dark mode aesthetics
    pg.setConfigOption("background", "#0f172a")
    pg.setConfigOption("foreground", "#f8fafc")
    pg.setConfigOption("antialias", True)
    HAS_PYQTGRAPH = True
except Exception:  # pragma: no cover
    HAS_PYQTGRAPH = False


class QuEstPlotCanvas(QtWidgets.QWidget):
    """Container wrapping a pyqtgraph PlotWidget or fallback canvas."""

    def __init__(
        self,
        title: str = "",
        x_label: str = "Time (ns)",
        y_label: str = "Counts",
        log_y: bool = False,
        log_x: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.title_text = title
        self.x_label = x_label
        self.y_label = y_label
        self.log_y = log_y
        self.log_x = log_x

        if HAS_PYQTGRAPH:
            self.plot_widget = pg.PlotWidget(self)
            self.plot_widget.setTitle(title, color="#e2e8f0", size="11pt")
            self.plot_widget.setLabel("bottom", x_label, color="#94a3b8")
            self.plot_widget.setLabel("left", y_label, color="#94a3b8")
            self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
            self.plot_widget.setLogMode(x=self.log_x, y=self.log_y)
            self.legend = self.plot_widget.addLegend(offset=(-10, 10))
            self.layout.addWidget(self.plot_widget)
        else:
            self.plot_widget = None
            lbl = QtWidgets.QLabel(f"{title}\n({x_label} vs {y_label})", self)
            lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.layout.addWidget(lbl)

        # Predefined color palette for plot series
        self.colors = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4"]

    def set_data(self, series_list: list[dict[str, Any]]):
        """Update plot curves from a list of series dicts ({x, y, name})."""
        if not HAS_PYQTGRAPH or self.plot_widget is None:
            return

        self.plot_widget.clear()
        if self.legend:
            self.legend.clear()

        for idx, series in enumerate(series_list):
            x = series.get("x")
            y = series.get("y")
            name = series.get("name", f"Series {idx+1}")
            if x is None or y is None or len(x) == 0:
                continue

            color = self.colors[idx % len(self.colors)]
            pen = pg.mkPen(color=color, width=2)
            self.plot_widget.plot(x, y, pen=pen, name=name)

    def set_log_scale(self, log_x: bool = False, log_y: bool = False):
        self.log_x = log_x
        self.log_y = log_y
        if HAS_PYQTGRAPH and self.plot_widget is not None:
            self.plot_widget.setLogMode(x=log_x, y=log_y)
