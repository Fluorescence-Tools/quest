"""ChiMol 3D Molecular Structure Viewer widget for QuEst desktop application.

Embeds ChiSurf's `chimol` 3D viewer (`chisurf.plugins.chimol.chimol.renderer.view.MolView`)
into the PyQt desktop workspace with auto-loading from `model.pdb`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from qtpy import QtCore, QtWidgets

from ..hosts import enable_chisurf

__all__ = ["ChiMolStructureWidget"]


class ChiMolStructureWidget(QtWidgets.QWidget):
    """3D Molecular Structure Viewer widget wrapping ChiSurf's ChiMol plugin."""

    def __init__(self, model: Any, parent=None):
        super().__init__(parent)
        self.model = model
        self._viewer = None
        self._loaded_pdb: str | None = None

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(4, 4, 4, 4)

        # Control Bar
        ctrl_bar = QtWidgets.QHBoxLayout()
        self.lbl_path = QtWidgets.QLabel("PDB: None", self)
        ctrl_bar.addWidget(self.lbl_path)

        ctrl_bar.addStretch()

        btn_reload = QtWidgets.QPushButton("Reload 3D Structure", self)
        btn_reload.clicked.connect(self.reload)
        ctrl_bar.addWidget(btn_reload)

        self.layout.addLayout(ctrl_bar)

        # Container for MolView or fallback placeholder
        self.container = QtWidgets.QWidget(self)
        self.container_layout = QtWidgets.QVBoxLayout(self.container)
        self.container_layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.container)

        self.reload()

    def reload(self):
        """Reload 3D molecular structure from `model.pdb` into ChiMol MolView."""
        pdb_path = getattr(self.model, "pdb", None)
        if not pdb_path or not Path(pdb_path).is_file():
            self._show_placeholder("No valid PDB structure file loaded.")
            return

        path_obj = Path(pdb_path).resolve()
        if str(path_obj) == self._loaded_pdb and self._viewer is not None:
            return

        self.lbl_path.setText(f"PDB: {path_obj.name}")

        enable_chisurf()
        try:
            from chisurf.plugins.chimol.chimol.renderer.view import MolView
            from chisurf.plugins.chimol.chimol.io.structure import load_structure_payload
            try:
                from chisurf.core.structure import Structure as _Struct
            except Exception:
                _Struct = None

            if self._viewer is None or not isinstance(self._viewer, MolView):
                # Clear container
                while self.container_layout.count():
                    child = self.container_layout.takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()

                try:
                    self._viewer = MolView(self.container)
                except TypeError:
                    self._viewer = MolView(self.container)
                self.container_layout.addWidget(self._viewer)

            structure, coords0 = load_structure_payload(path_obj, structure_factory=_Struct)
            name = path_obj.stem
            if structure is not None:
                self._viewer.add_structure(structure, name=name, source_path=str(path_obj))
            elif coords0 is not None:
                import numpy as np
                self._viewer.add_coordinates(np.asarray(coords0, dtype=float), name=name, source_path=str(path_obj))
            elif hasattr(self._viewer, "load_file"):
                self._viewer.load_file(str(path_obj))

            self._loaded_pdb = str(path_obj)

        except Exception as exc:
            self._show_placeholder(f"3D Structure Viewer (ChiMol)\nFile: {path_obj.name}\n({exc})")

    def _show_placeholder(self, message: str):
        while self.container_layout.count():
            child = self.container_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self._viewer = None
        lbl = QtWidgets.QLabel(message, self.container)
        lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lbl.setWordWrap(True)
        self.container_layout.addWidget(lbl)
