"""ChiMol 3D Molecular Structure Viewer widget for QuEst desktop application.

Embeds ChiSurf's `chimol` 3D viewer (`chisurf.plugins.chimol.chimol.renderer.view.MolView`)
into the PyQt desktop workspace with auto-loading from `model.pdb`.
Displays the 3D viewport canvas even when no structure is loaded.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from qtpy import QtCore, QtWidgets

from ..hosts import enable_chisurf

logger = logging.getLogger(__name__)

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
        self.lbl_path = QtWidgets.QLabel("PDB: None (3D Viewport Ready)", self)
        ctrl_bar.addWidget(self.lbl_path)

        ctrl_bar.addStretch()

        btn_reload = QtWidgets.QPushButton("Reload 3D Structure", self)
        btn_reload.clicked.connect(self.reload)
        ctrl_bar.addWidget(btn_reload)

        self.layout.addLayout(ctrl_bar)

        # Container for MolView viewport
        self.container = QtWidgets.QWidget(self)
        self.container_layout = QtWidgets.QVBoxLayout(self.container)
        self.container_layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.container)

        self._ensure_viewer()
        self.reload()

    def _ensure_viewer(self) -> Any:
        """Ensure the MolView 3D viewport instance is created and placed in container layout."""
        if self._viewer is not None:
            return self._viewer

        enable_chisurf()
        try:
            from chisurf.plugins.chimol.chimol.renderer.view import MolView
            self._viewer = MolView(self.container)
            self.container_layout.addWidget(self._viewer)
            return self._viewer
        except Exception as exc:
            logger.warning("Could not instantiate ChiMol MolView: %s", exc)
            self._show_placeholder(f"3D Structure Viewport\n({exc})")
            return None

    def reload(self):
        """Reload 3D molecular structure from `model.pdb` into ChiMol MolView."""
        viewer = self._ensure_viewer()

        pdb_path = getattr(self.model, "pdb", None)
        if not pdb_path or not Path(pdb_path).is_file():
            self.lbl_path.setText("PDB: None (3D Viewport Ready)")
            self._loaded_pdb = None
            return

        path_obj = Path(pdb_path).resolve()
        if str(path_obj) == self._loaded_pdb and viewer is not None:
            return

        self.lbl_path.setText(f"PDB: {path_obj.name}")

        if viewer is None:
            return

        try:
            from chisurf.plugins.chimol.chimol.io.structure import load_structure_payload
            try:
                from chisurf.core.structure import Structure as _Struct
            except Exception:
                _Struct = None

            structure, coords0 = load_structure_payload(path_obj, structure_factory=_Struct)
            name = path_obj.stem
            if structure is not None:
                viewer.add_structure(structure, name=name, source_path=str(path_obj))
            elif coords0 is not None:
                import numpy as np
                viewer.add_coordinates(np.asarray(coords0, dtype=float), name=name, source_path=str(path_obj))
            elif hasattr(viewer, "load_file"):
                viewer.load_file(str(path_obj))

            self._loaded_pdb = str(path_obj)
        except Exception as exc:
            logger.warning("Failed loading PDB %s into ChiMol: %s", path_obj, exc)
            self.lbl_path.setText(f"PDB: {path_obj.name} (Error: {exc})")

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
