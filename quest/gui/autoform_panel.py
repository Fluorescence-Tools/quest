"""Control panels rendered by host AutoForm or in-tree PyQt AutoForm.

Serves as the single renderer bridge for PyQt applications, using the exact
same AutoForm view-spec JSON schema (`quest.view.json`) as the Web UI, with standard native Qt controls.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from qtpy import QtCore, QtWidgets

__all__ = [
    "MANIFEST_PATH",
    "autoform_supported",
    "method_declaration",
    "declared_methods",
    "build_method_panel",
    "InTreeCollapsiblePanel",
    "InTreePyQtAutoForm",
]

MANIFEST_PATH = Path(__file__).resolve().parents[1] / "manifest.json"


@lru_cache(maxsize=1)
def _manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open("r", encoding="utf8") as handle:
        return json.load(handle)


def declared_methods() -> list[str]:
    """Names of the RPC methods the manifest declares, in declaration order."""
    return [method["name"] for method in _manifest()["rpc_methods"]]


def method_declaration(name: str) -> dict[str, Any]:
    """Return one `rpc_methods` entry."""
    for method in _manifest()["rpc_methods"]:
        if method["name"] == name:
            return method
    raise KeyError(f"manifest declares no method named {name!r}")


def autoform_supported() -> bool:
    """Whether the host's AutoForm can be reached from here."""
    from ..hosts import autoform_available

    return autoform_available()


def build_method_panel(name: str, parent=None, **kwargs):
    """Render the parameter form for one declared method."""
    if autoform_supported():
        from chisurf.gui.autoform.auto_form import AutoForm

        kwargs.setdefault("title", name)
        return AutoForm.from_rpc_method(method_declaration(name), parent=parent, **kwargs)

    # Standalone fallback when ChiSurf host is absent
    panel = QtWidgets.QWidget(parent)
    layout = QtWidgets.QVBoxLayout(panel)
    layout.addWidget(QtWidgets.QLabel(f"RPC Method Form: {name}", parent=panel))
    return panel


class InTreeCollapsiblePanel(QtWidgets.QGroupBox):
    """Standard Qt GroupBox panel with collapsible content."""

    def __init__(self, title: str, parent=None, collapsed: bool = False):
        super().__init__(title, parent)
        self.setCheckable(True)
        self.setChecked(not collapsed)

        self.content_widget = QtWidgets.QWidget(self)
        self.content_layout = QtWidgets.QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(4, 4, 4, 4)

        box_layout = QtWidgets.QVBoxLayout(self)
        box_layout.setContentsMargins(6, 14, 6, 6)
        box_layout.addWidget(self.content_widget)

        self.toggled.connect(self._on_toggled)
        self._on_toggled(not collapsed)

    @property
    def is_collapsed(self) -> bool:
        return not self.isChecked()

    def _on_toggled(self, checked: bool):
        if hasattr(self, "content_widget"):
            self.content_widget.setVisible(checked)

    def _toggle(self):
        self.setChecked(not self.isChecked())


class InTreePyQtAutoForm(QtWidgets.QWidget):
    """In-tree PyQt AutoForm renderer using standard desktop Qt controls."""

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model
        self._fields: list[tuple[str | None, str, QtWidgets.QWidget]] = []

        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(4, 4, 4, 4)

        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        container = QtWidgets.QWidget()
        self.scroll_layout = QtWidgets.QVBoxLayout(container)

        spec = getattr(model, "spec", None) or (model.view_spec() if hasattr(model, "view_spec") else {})
        if hasattr(spec, "to_dict"):
            spec = spec.to_dict()
        elif hasattr(spec, "sections"):
            spec = {"sections": [s.to_dict() if hasattr(s, "to_dict") else s for s in spec.sections]}

        sections = spec.get("sections", []) if isinstance(spec, dict) else []
        self._render_sections(sections, self.scroll_layout)

        scroll.setWidget(container)
        self.main_layout.addWidget(scroll)

    def _render_sections(self, sections: list, parent_layout: QtWidgets.QLayout):
        for sec in sections:
            sec_type = getattr(sec, "type", None) or (sec.get("type") if isinstance(sec, dict) else None)
            if not sec_type and hasattr(sec, "__class__"):
                cls_name = re.sub(r"(?<!^)(?=[A-Z])", "_", sec.__class__.__name__).lower().replace("_section", "")
                if cls_name in {"panel", "dock_area", "value", "choice", "toggle", "toggle_row", "button_row", "table", "info", "custom"}:
                    sec_type = cls_name

            if not sec_type:
                continue

            if sec_type in {"panel", "dock_area"}:
                title = getattr(sec, "title", None) or (sec.get("title") if isinstance(sec, dict) else "Panel") or "Panel"
                collapsed = bool(getattr(sec, "collapsed", False) if not isinstance(sec, dict) else sec.get("collapsed", False))
                panel = InTreeCollapsiblePanel(title, collapsed=collapsed)
                child_secs = getattr(sec, "sections", None) or (sec.get("sections") if isinstance(sec, dict) else []) or []
                if child_secs:
                    self._render_sections(child_secs, panel.content_layout)
                parent_layout.addWidget(panel)
            elif sec_type == "value":
                attr = getattr(sec, "attr", None) or (sec.get("attr") if isinstance(sec, dict) else None)
                label_text = getattr(sec, "label", None) or (sec.get("label") if isinstance(sec, dict) else None) or attr or ""
                kind = getattr(sec, "kind", None) or (sec.get("kind") if isinstance(sec, dict) else "str") or "str"
                minimum = getattr(sec, "minimum", -1e9) if not isinstance(sec, dict) else sec.get("minimum", -1e9)
                maximum = getattr(sec, "maximum", 1e9) if not isinstance(sec, dict) else sec.get("maximum", 1e9)
                step = getattr(sec, "step", None) if not isinstance(sec, dict) else sec.get("step")

                row = QtWidgets.QHBoxLayout()
                lbl = QtWidgets.QLabel(f"{label_text}:")
                row.addWidget(lbl)

                val = getattr(self.model, attr, None) if attr else None
                if kind in {"float", "number"}:
                    spin = QtWidgets.QDoubleSpinBox()
                    spin.setRange(minimum if minimum is not None else -1e9, maximum if maximum is not None else 1e9)
                    if step:
                        spin.setSingleStep(step)
                    if val is not None:
                        spin.setValue(float(val))
                    if attr:
                        spin.valueChanged.connect(lambda v, a=attr: setattr(self.model, a, v))
                    row.addWidget(spin)
                    self._fields.append((attr, kind, spin))
                elif kind == "int":
                    spin = QtWidgets.QSpinBox()
                    spin.setRange(int(minimum) if minimum is not None else -1000000, int(maximum) if maximum is not None else 1000000)
                    if val is not None:
                        spin.setValue(int(val))
                    if attr:
                        spin.valueChanged.connect(lambda v, a=attr: setattr(self.model, a, v))
                    row.addWidget(spin)
                    self._fields.append((attr, kind, spin))
                else:
                    edit = QtWidgets.QLineEdit()
                    if val is not None:
                        edit.setText(str(val))
                    if attr:
                        edit.textChanged.connect(lambda txt, a=attr: setattr(self.model, a, txt))
                    row.addWidget(edit)
                    self._fields.append((attr, kind, edit))

                parent_layout.addLayout(row)
            elif sec_type == "choice":
                attr = getattr(sec, "attr", None) or (sec.get("attr") if isinstance(sec, dict) else None)
                label_text = getattr(sec, "label", None) or (sec.get("label") if isinstance(sec, dict) else None) or attr or ""
                row = QtWidgets.QHBoxLayout()
                row.addWidget(QtWidgets.QLabel(f"{label_text}:"))
                combo = QtWidgets.QComboBox()
                opts = getattr(sec, "options", []) if not isinstance(sec, dict) else sec.get("options", [])
                for opt in (opts or []):
                    if isinstance(opt, dict):
                        combo.addItem(str(opt.get("label")), opt.get("value"))
                    elif hasattr(opt, "label"):
                        combo.addItem(str(opt.label), getattr(opt, "value", opt.label))
                    else:
                        combo.addItem(str(opt), opt)
                val = getattr(self.model, attr, None) if attr else None
                if val is not None:
                    idx = combo.findData(val)
                    if idx >= 0:
                        combo.setCurrentIndex(idx)
                if attr:
                    combo.currentIndexChanged.connect(
                        lambda _, c=combo, a=attr: setattr(self.model, a, c.currentData())
                    )
                row.addWidget(combo)
                self._fields.append((attr, "choice", combo))
                parent_layout.addLayout(row)
            elif sec_type in {"toggle", "toggle_row"}:
                attr = getattr(sec, "attr", None) or (sec.get("attr") if isinstance(sec, dict) else None)
                label_text = getattr(sec, "label", None) or (sec.get("label") if isinstance(sec, dict) else None) or attr or ""
                chk = QtWidgets.QCheckBox(label_text)
                val = getattr(self.model, attr, None) if attr else None
                if val is not None:
                    chk.setChecked(bool(val))
                if attr:
                    chk.toggled.connect(lambda b, a=attr: setattr(self.model, a, b))
                parent_layout.addWidget(chk)
                self._fields.append((attr, "toggle", chk))
            elif sec_type == "table":
                from .quencher_widget import QuencherTableWidget

                tbl_widget = QuencherTableWidget(self.model)
                parent_layout.addWidget(tbl_widget)
                if not hasattr(self, "_tables"):
                    self._tables = []
                self._tables.append(tbl_widget)
            elif sec_type == "plot":
                from .plots import QuEstPlotCanvas

                title = getattr(sec, "title", None) or (sec.get("title") if isinstance(sec, dict) else "Plot") or "Plot"
                x_label = getattr(sec, "x_label", "x") if not isinstance(sec, dict) else sec.get("x_label", "x")
                y_label = getattr(sec, "y_label", "y") if not isinstance(sec, dict) else sec.get("y_label", "y")
                log_y = bool(getattr(sec, "log_y", False) if not isinstance(sec, dict) else sec.get("log_y", False))
                log_x = bool(getattr(sec, "log_x", False) if not isinstance(sec, dict) else sec.get("log_x", False))
                source = getattr(sec, "source", None) or (sec.get("source") if isinstance(sec, dict) else None)

                plot_canvas = QuEstPlotCanvas(title=title, x_label=x_label, y_label=y_label, log_y=log_y, log_x=log_x)
                parent_layout.addWidget(plot_canvas)
                if not hasattr(self, "_plots"):
                    self._plots = []
                self._plots.append((source, plot_canvas))

    def sync_fields(self):
        for attr, kind, widget in self._fields:
            if not attr:
                continue
            val = getattr(self.model, attr, None)
            if val is None:
                continue
            if kind in {"float", "number"}:
                widget.setValue(float(val))
            elif kind == "int":
                widget.setValue(int(val))
            elif kind == "choice":
                idx = widget.findData(val)
                if idx >= 0:
                    widget.setCurrentIndex(idx)
            elif kind == "toggle":
                widget.setChecked(bool(val))
            else:
                widget.setText(str(val))

        self.refresh_plots()
        for tbl in getattr(self, "_tables", []):
            try:
                tbl.refresh()
            except Exception:
                pass

    def refresh_plots(self):
        for source, plot_canvas in getattr(self, "_plots", []):
            if not source:
                continue
            fn = getattr(self.model, source, None)
            if callable(fn):
                series = fn()
                plot_canvas.set_data(series)
