"""ChiMol 3D Molecular Structure Viewer widget for QuEst desktop application.

Embeds ChiSurf's `chimol` 3D viewer (`MolView`) and `SequenceDock`, with:
- Water stripping before loading.
- Click-to-select on the 3D structure or the sequence bar.
- Single-AA selection: donor (cyan) and acceptor (magenta) toggle buttons.
- Selection syncs to the autoform attachment fields.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from qtpy import QtCore, QtWidgets

from ..hosts import enable_chisurf

logger = logging.getLogger(__name__)

__all__ = ["ChiMolStructureWidget"]

DONOR_COLOR = "#17becf"
ACCEPTOR_COLOR = "#e377c2"

_SEQ_ROLE = QtCore.Qt.UserRole + 150


class ChiMolStructureWidget(QtWidgets.QWidget):
    """3D viewer + sequence bar with click-to-select residue picking."""

    residueSelected = QtCore.Signal(str, int)

    def __init__(self, model: Any, parent=None):
        super().__init__(parent)
        self.model = model
        self._viewer = None
        self._loaded_pdb: str | None = None
        self._pick_target = "donor"
        self._on_selection_callback = None

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(4, 4, 4, 4)
        self.layout.setSpacing(2)

        self._build_controls()

        # Sequence bar: ChiSurf's SequenceDock (widget builder) + wiring.
        self._seq_dock = None
        self._build_sequence_dock()

        self.container = QtWidgets.QWidget(self)
        self.container_layout = QtWidgets.QVBoxLayout(self.container)
        self.container_layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.container)

        self._ensure_viewer()
        self.reload()

    def set_selection_callback(self, callback) -> None:
        self._on_selection_callback = callback

    def _build_controls(self) -> None:
        ctrl = QtWidgets.QHBoxLayout()

        self.lbl_path = QtWidgets.QLabel("PDB: None (3D Viewport Ready)", self)
        ctrl.addWidget(self.lbl_path)
        ctrl.addStretch()

        btn_reload = QtWidgets.QPushButton("Reload", self)
        btn_reload.clicked.connect(self.reload)
        ctrl.addWidget(btn_reload)

        self.donor_btn = QtWidgets.QPushButton("Donor", self)
        self.donor_btn.setCheckable(True)
        self.donor_btn.setChecked(True)
        self._style_toggle(self.donor_btn, DONOR_COLOR, True)
        self.donor_btn.clicked.connect(lambda: self._set_pick_target("donor"))
        ctrl.addWidget(self.donor_btn)

        self.acceptor_btn = QtWidgets.QPushButton("Acceptor", self)
        self.acceptor_btn.setCheckable(True)
        self._style_toggle(self.acceptor_btn, ACCEPTOR_COLOR, False)
        self.acceptor_btn.clicked.connect(lambda: self._set_pick_target("acceptor"))
        ctrl.addWidget(self.acceptor_btn)

        self.layout.addLayout(ctrl)

    def _style_toggle(self, btn, color: str, active: bool) -> None:
        if active:
            btn.setStyleSheet(
                f"QPushButton {{ background-color: {color}; color: white; "
                f"font-weight: bold; border: 1px solid {color}; border-radius: 4px; "
                f"padding: 4px 12px; }}"
            )
        else:
            btn.setStyleSheet(
                f"QPushButton {{ background: transparent; color: {color}; "
                f"font-weight: bold; border: 1px solid {color}; border-radius: 4px; "
                f"padding: 4px 12px; }}"
            )

    def _set_pick_target(self, target: str) -> None:
        self._pick_target = target
        self._style_toggle(self.donor_btn, DONOR_COLOR, target == "donor")
        self._style_toggle(self.acceptor_btn, ACCEPTOR_COLOR, target == "acceptor")

    # -- sequence bar: reuse ChiSurf's SequenceDock --------------------------

    def _build_sequence_dock(self) -> None:
        try:
            from chisurf.plugins.chimol.chimol.app.sequence_dock import SequenceDock
            self._seq_dock = SequenceDock(
                self,
                margins=(4, 2, 4, 2),
                spacing=6,
            )
            self.layout.addWidget(self._seq_dock.widget)
            self._seq_dock.seq_list.itemSelectionChanged.connect(
                self._on_sequence_click
            )
        except Exception as exc:
            logger.warning("SequenceDock unavailable (%s), using placeholder", exc)
            self._seq_dock = None
            lbl = QtWidgets.QLabel("Sequence bar unavailable (needs ChiSurf)", self)
            lbl.setStyleSheet("color: #999; font-size: 11px; padding: 4px;")
            self.layout.addWidget(lbl)

    def _populate_sequence(self) -> None:
        if self._seq_dock is None:
            return
        viewer = self._viewer
        if viewer is None:
            return

        obj_id = viewer.get_active_object_id() if hasattr(viewer, "get_active_object_id") else None
        if obj_id is None:
            return

        try:
            seq_codes, res_names = viewer.get_sequence_arrays(obj_id)
            res_numbers = viewer.get_residue_numbers(obj_id)
        except Exception as exc:
            logger.debug("Could not read sequence arrays: %s", exc)
            return

        seq_list = self._seq_dock.seq_list
        numbers_list = self._seq_dock.seq_numbers_list
        seq_list.blockSignals(True)
        seq_list.clear()
        numbers_list.clear()

        for i, (code, res_no) in enumerate(zip(seq_codes, res_numbers)):
            item = QtWidgets.QListWidgetItem(str(code))
            item.setData(_SEQ_ROLE, i)
            item.setToolTip(f"{res_names[i]} {res_no}")
            seq_list.addItem(item)

            num_item = QtWidgets.QListWidgetItem(
                str(int(res_no)) if i % 5 == 0 else ""
            )
            num_item.setData(_SEQ_ROLE, i)
            numbers_list.addItem(num_item)

        seq_list.blockSignals(False)

        # Sync scrollbar
        max_val = max(0, seq_list.count() - 1)
        self._seq_dock.seq_scrollbar.setRange(0, max_val)
        self._seq_dock.seq_scrollbar.setEnabled(max_val > 0)

    def _on_sequence_click(self) -> None:
        if self._seq_dock is None or self._viewer is None:
            return
        seq_list = self._seq_dock.seq_list
        indices = []
        for model_idx in seq_list.selectedIndexes():
            item = seq_list.item(model_idx.row())
            if item:
                idx = item.data(_SEQ_ROLE)
                if isinstance(idx, int) and idx >= 0:
                    indices.append(idx)
        if not indices:
            return

        viewer = self._viewer
        try:
            chain_ids = viewer._residue_chain_ids
            res_ids = viewer._residue_ids
            if chain_ids is not None and res_ids is not None:
                idx = indices[0]
                chain_id = str(chain_ids[idx])
                residue_id = int(res_ids[idx])
                self._apply_selection(chain_id, residue_id)
        except Exception as exc:
            logger.debug("Sequence click resolve failed: %s", exc)

    def _sync_sequence_from_viewer(self, object_id, residue_indices) -> None:
        """3D pick → highlight in sequence bar."""
        if self._seq_dock is None or not residue_indices:
            return
        seq_list = self._seq_dock.seq_list
        seq_list.blockSignals(True)
        seq_list.clearSelection()
        target = {int(i) for i in residue_indices}
        for row in range(seq_list.count()):
            item = seq_list.item(row)
            if item:
                idx = item.data(_SEQ_ROLE)
                if isinstance(idx, int) and idx in target:
                    item.setSelected(True)
        seq_list.blockSignals(False)

    # -- MolView picking ----------------------------------------------------

    def _on_molview_pick(self, object_id, residue_indices) -> None:
        if not residue_indices:
            return
        viewer = self._viewer
        if viewer is None:
            return
        idx = int(residue_indices[0])
        try:
            chain_ids = viewer._residue_chain_ids
            res_ids = viewer._residue_ids
            if chain_ids is not None and res_ids is not None and idx < len(res_ids):
                chain_id = str(chain_ids[idx])
                residue_id = int(res_ids[idx])
                self._apply_selection(chain_id, residue_id)
                self._sync_sequence_from_viewer(object_id, residue_indices)
        except Exception as exc:
            logger.debug("Pick resolve failed: %s", exc)

    def _apply_selection(self, chain_id: str, residue_id: int) -> None:
        target = self._pick_target
        if target == "donor":
            self.model.attachment_chain = chain_id
            self.model.attachment_residue = residue_id
            self.model.attachment_atom = "CB"
            dyes = self.model.project.get("fret", {}).get("dyes", [])
            if dyes:
                dyes[0]["attachment"] = {
                    "chain": chain_id, "residue": residue_id, "atom": "CB",
                }
        elif target == "acceptor":
            fret = self.model.project.setdefault("fret", {})
            fret["enabled"] = True
            dyes = fret.setdefault("dyes", [])
            while len(dyes) < 2:
                dyes.append({"name": "acceptor", "D": 7.5,
                            "attachment": {"chain": chain_id, "residue": residue_id, "atom": "CB"},
                            "av_parameter": {"linker_length": 21.5, "linker_width": 0.5, "radius1": 3.5}})
            dyes[1]["attachment"] = {
                "chain": chain_id, "residue": residue_id, "atom": "CB",
            }

        self.model.status = f"Selected {target}: {chain_id}:{residue_id}"
        self.residueSelected.emit(chain_id, residue_id)

        if self._on_selection_callback:
            self._on_selection_callback()

    # -- viewer lifecycle ---------------------------------------------------

    def _ensure_viewer(self) -> Any:
        if self._viewer is not None:
            return self._viewer

        enable_chisurf()
        try:
            from chisurf.plugins.chimol.chimol.renderer.view import MolView
            self._viewer = MolView(self.container)
            self.container_layout.addWidget(self._viewer)
            if hasattr(self._viewer, "objectResidueSelectionChanged"):
                self._viewer.objectResidueSelectionChanged.connect(
                    self._on_molview_pick
                )
            return self._viewer
        except Exception as exc:
            logger.warning("Could not instantiate ChiMol MolView: %s", exc)
            self._show_placeholder(f"3D Structure Viewport\n({exc})")
            return None

    def reload(self):
        viewer = self._ensure_viewer()

        pdb_path = getattr(self.model, "pdb", None)
        if not pdb_path or not Path(pdb_path).is_file():
            self.lbl_path.setText("PDB: None (3D Viewport Ready)")
            self._loaded_pdb = None
            if self._seq_dock:
                self._seq_dock.seq_list.clear()
            return

        path_obj = Path(pdb_path).resolve()
        if str(path_obj) == self._loaded_pdb and viewer is not None:
            return

        self.lbl_path.setText(f"PDB: {path_obj.name}")

        if viewer is None:
            return

        self._clear_viewer(viewer)

        try:
            from chisurf.plugins.chimol.chimol.io.structure import load_structure_payload
            try:
                from chisurf.core.structure import Structure as _Struct
            except Exception:
                _Struct = None

            clean_path, _ = water_stripped_pdb_path(path_obj)
            clean_path_obj = Path(clean_path)

            structure, coords0 = load_structure_payload(clean_path_obj, structure_factory=_Struct)
            name = path_obj.stem
            if structure is not None:
                viewer.add_structure(structure, name=name, source_path=str(clean_path_obj))
            elif coords0 is not None:
                import numpy as np
                viewer.add_coordinates(np.asarray(coords0, dtype=float), name=name, source_path=str(clean_path_obj))
            elif hasattr(viewer, "load_file"):
                viewer.load_file(str(clean_path_obj))

            self._loaded_pdb = str(path_obj)

            # Populate sequence from the viewer's internal state.
            self._populate_sequence()
            if self._seq_dock:
                self._seq_dock.seq_label.setText(name)
        except Exception as exc:
            logger.warning("Failed loading PDB %s into ChiMol: %s", path_obj, exc)
            self.lbl_path.setText(f"PDB: {path_obj.name} (Error: {exc})")

    @staticmethod
    def _clear_viewer(viewer):
        try:
            objects = getattr(viewer, "_objects", None)
            if objects:
                for oid in list(objects.keys()):
                    try:
                        viewer.remove_object(oid)
                    except Exception:
                        pass
                return
            if hasattr(viewer, "list_objects"):
                for obj in viewer.list_objects():
                    try:
                        viewer.remove_object(obj["id"])
                    except Exception:
                        pass
        except Exception:
            pass

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


# Late import to keep the module-level surface clean
from ..core.structure import water_stripped_pdb_path
