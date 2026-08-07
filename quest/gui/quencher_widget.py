"""Interactive Quenching Table Widget for QuEst desktop GUI.

Renders and edits amino-acid PET quenching parameters and diffusion slowing factors.
"""

from __future__ import annotations

from typing import Any
from qtpy import QtCore, QtWidgets

from ..project import template_project



class QuencherTableWidget(QtWidgets.QWidget):
    """Full interactive QTableWidget for amino-acid quenching parameters."""

    def __init__(self, model: Any, parent=None):
        super().__init__(parent)
        self.model = model
        self.columns = [
            ("residue", "Residue"),
            ("kQ", "kQ (1/ns)"),
            ("quench_radius", "Radius (Å)"),
            ("quench_atoms", "Atoms"),
            ("slow_factor", "Slow Factor"),
        ]

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(4, 4, 4, 4)

        # Toolbar above table
        toolbar = QtWidgets.QHBoxLayout()
        btn_reset = QtWidgets.QPushButton("Reset Defaults", self)
        btn_reset.clicked.connect(self._reset_defaults)
        toolbar.addWidget(btn_reset)

        toolbar.addStretch()
        self.layout.addLayout(toolbar)

        # Table
        self.table = QtWidgets.QTableWidget(self)
        self.table.setColumnCount(len(self.columns))
        self.table.setHorizontalHeaderLabels([label for _, label in self.columns])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.cellChanged.connect(self._on_cell_changed)
        self.layout.addWidget(self.table)

        self._updating = False
        self.refresh()

    def refresh(self):
        """Populate table rows from model.quencher_rows()."""
        self._updating = True
        rows = self.model.quencher_rows()
        self.table.setRowCount(len(rows))

        for r_idx, row in enumerate(rows):
            for c_idx, (key, _) in enumerate(self.columns):
                val = row.get(key)
                text = "" if val is None else str(val)
                item = QtWidgets.QTableWidgetItem(text)
                if key == "residue":
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)

                self.table.setItem(r_idx, c_idx, item)

        self._updating = False

    def _on_cell_changed(self, row: int, col: int):
        if self._updating:
            return
        key, _ = self.columns[col]
        if key == "residue":
            return
        item = self.table.item(row, col)
        val = item.text() if item else ""
        self.model.update_quencher_cell(row, key, val)

    def _reset_defaults(self):
        defaults = template_project().get("amino_acid_interactions", {})
        self.model.project["amino_acid_interactions"] = defaults
        self.model._notify()
        self.refresh()

