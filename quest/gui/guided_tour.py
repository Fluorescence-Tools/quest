"""Guided Tour & Demo Mode Controller for QuEst PyQt Desktop Application.

Loads shared `quest/settings/tour.json` tour definitions and walks the user through
the desktop workspace with spotlight highlights, step explanations, and 148L demo mode loading.
Uses ChiSurf's `chisurf.gui.widgets.tools.guided_tour` when available, or `InTreeGuidedTour` standalone.
"""

from __future__ import annotations

import logging
from typing import Any
from qtpy import QtCore, QtGui, QtWidgets

from ..hosts import enable_chisurf
from ..tour import get_tour_data, create_demo_project, TOUR_JSON_PATH

logger = logging.getLogger(__name__)

__all__ = ["start_guided_tour", "InTreeGuidedTour"]


def start_guided_tour(host: QtWidgets.QWidget, model: Any = None) -> Any:
    """Start the guided tour on *host*, loading demo 148L structure & parameters."""
    # Apply demo project parameters to model
    if hasattr(host, "model") and hasattr(host, "refresh"):
        demo_project = create_demo_project()
        host.model.load_project_dict(demo_project)
        host.refresh()

    enable_chisurf()
    try:
        from chisurf.gui.widgets.tools.guided_tour import GuidedTour, load_tour
        steps = load_tour(TOUR_JSON_PATH)
        if steps:
            tour = GuidedTour(host, steps, model=model or getattr(host, "model", None))
            tour.start()
            return tour
    except Exception as exc:
        logger.debug("Falling back to InTreeGuidedTour: %s", exc)

    tour = InTreeGuidedTour(host)
    tour.start()
    return tour


class _Spotlight(QtWidgets.QWidget):
    """Translucent overlay that dims the window except for the target widget."""

    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self._rect = QtCore.QRect()

    def set_target_rect(self, rect: QtCore.QRect) -> None:
        self._rect = QtCore.QRect(rect)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        shade = QtGui.QColor(0, 0, 0, 110)
        full = self.rect()
        hole = self._rect.adjusted(-6, -6, 6, 6) if (
            self._rect.isValid() and not self._rect.isEmpty()
        ) else QtCore.QRect()

        if hole.isEmpty():
            painter.fillRect(full, shade)
        else:
            hole = hole.intersected(full)
            painter.fillRect(QtCore.QRect(full.left(), full.top(), full.width(), hole.top() - full.top()), shade)
            painter.fillRect(QtCore.QRect(full.left(), hole.bottom() + 1, full.width(), full.bottom() - hole.bottom()), shade)
            painter.fillRect(QtCore.QRect(full.left(), hole.top(), hole.left() - full.left(), hole.height()), shade)
            painter.fillRect(QtCore.QRect(hole.right() + 1, hole.top(), full.right() - hole.right(), hole.height()), shade)
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 190, 60), 2))
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawRoundedRect(hole.adjusted(1, 1, -1, -1), 6, 6)
        painter.end()


class InTreeGuidedTour(QtCore.QObject):
    """In-tree PyQt GuidedTour implementation for standalone desktop execution."""

    def __init__(self, host: QtWidgets.QWidget):
        super().__init__(host)
        self.host = host
        self.tour_data = get_tour_data()
        self.steps = self.tour_data.get("steps", [])
        self.index = 0

        self.spotlight: _Spotlight | None = None
        self.bubble: QtWidgets.QFrame | None = None

    def start(self, index: int = 0):
        if not self.steps:
            return
        self.index = index
        self._build_ui()
        self._show_step()

    def _build_ui(self):
        if self.spotlight is None:
            self.spotlight = _Spotlight(self.host)
        if self.bubble is None:
            self.bubble = QtWidgets.QFrame(self.host)
            self.bubble.setFrameShape(QtWidgets.QFrame.StyledPanel)
            self.bubble.setStyleSheet(
                "QFrame { background-color: #ffffff; border: 2px solid #ffbe3c; border-radius: 6px; padding: 8px; }"
            )
            self.bubble.setMaximumWidth(400)

            layout = QtWidgets.QVBoxLayout(self.bubble)
            self.lbl_title = QtWidgets.QLabel()
            font = self.lbl_title.font()
            font.setBold(True)
            font.setPointSize(11)
            self.lbl_title.setFont(font)
            layout.addWidget(self.lbl_title)

            self.lbl_text = QtWidgets.QLabel()
            self.lbl_text.setWordWrap(True)
            self.lbl_text.setTextFormat(QtCore.Qt.RichText)
            layout.addWidget(self.lbl_text)

            btn_row = QtWidgets.QHBoxLayout()
            self.lbl_count = QtWidgets.QLabel()
            btn_row.addWidget(self.lbl_count)
            btn_row.addStretch()

            btn_close = QtWidgets.QPushButton("Close")
            btn_close.clicked.connect(self.stop)
            btn_row.addWidget(btn_close)

            btn_back = QtWidgets.QPushButton("◀ Back")
            btn_back.clicked.connect(self.back)
            btn_row.addWidget(btn_back)

            btn_next = QtWidgets.QPushButton("Next ▶")
            btn_next.clicked.connect(self.next)
            btn_row.addWidget(btn_next)

            layout.addLayout(btn_row)

        self.spotlight.setGeometry(self.host.rect())
        self.spotlight.show()
        self.spotlight.raise_()
        self.bubble.show()
        self.bubble.raise_()

    def _show_step(self):
        if self.index < 0 or self.index >= len(self.steps):
            self.stop()
            return

        step = self.steps[self.index]
        self.lbl_title.setText(step.get("title", ""))
        self.lbl_text.setText(step.get("text", ""))
        self.lbl_count.setText(f"{self.index + 1} / {len(self.steps)}")

        target_widget = self._resolve_target(step.get("target", {}))
        if target_widget and target_widget.isVisible():
            # Get geometry relative to host
            pos = target_widget.mapTo(self.host, QtCore.QPoint(0, 0))
            rect = QtCore.QRect(pos, target_widget.size())
            self.spotlight.set_target_rect(rect)

            # Position bubble near target
            bubble_x = min(pos.x(), self.host.width() - 420)
            bubble_y = min(pos.y() + rect.height() + 10, self.host.height() - 200)
            self.bubble.move(max(10, bubble_x), max(10, bubble_y))
        else:
            self.spotlight.set_target_rect(QtCore.QRect())
            self.bubble.move((self.host.width() - 400) // 2, (self.host.height() - 200) // 2)

        self.bubble.adjustSize()

    def _resolve_target(self, target: dict[str, Any]) -> QtWidgets.QWidget | None:
        action = target.get("action")
        title = target.get("title")
        tab = target.get("tab")

        if action:
            for btn in self.host.findChildren(QtWidgets.QAbstractButton):
                if action.lower() in btn.text().lower():
                    return btn
        if tab:
            for tab_widget in self.host.findChildren(QtWidgets.QTabWidget):
                for idx in range(tab_widget.count()):
                    if tab.lower() in tab_widget.tabText(idx).lower():
                        tab_widget.setCurrentIndex(idx)
                        return tab_widget.widget(idx)
        if title:
            for group in self.host.findChildren(QtWidgets.QGroupBox):
                if title.lower() in group.title().lower():
                    return group
        return None

    def next(self):
        if self.index + 1 >= len(self.steps):
            self.stop()
        else:
            self.index += 1
            self._show_step()

    def back(self):
        if self.index > 0:
            self.index -= 1
            self._show_step()

    def stop(self):
        if self.spotlight:
            self.spotlight.hide()
            self.spotlight.deleteLater()
            self.spotlight = None
        if self.bubble:
            self.bubble.hide()
            self.bubble.deleteLater()
            self.bubble = None
