"""The form's state is the project, and nothing else.

QuEst already has one description of every input: `quest/settings/parameter_catalog.json`
names each field by its **path in the project dict** (`av_parameter.linker_length`,
`fret.dyes.1.attachment.chain`) and carries its label, type and help text. The
docs are generated from it and the web UI reads it. This module makes the Qt
form read it too, by exposing a project dict as a flat set of attributes that
ChiSurf's AutoForm can bind to.

That removes the thing the `.ui` files existed to do — and it removes a whole
class of bug with them. The old widget kept its state in ~40 Qt controls and
converted to a project dict on the way out (`to_project_dict`) and back on the
way in (`load_project_dict`); the two directions were written by hand, and the
conversion is exactly where the GUI drifted away from the CLI before. Here there
is no conversion: :attr:`ProjectFormModel.project` *is* the state, so what the
GUI saves is byte-for-byte a project the CLI and the web UI accept.

Field names
-----------
Attribute names are the catalog paths, shortened where the path is long enough
to hurt (`av_parameter.` → `av_`, `fret.dyes.0.` → `donor_`, `fret.dyes.1.` →
`acceptor_`). :data:`FIELD_PATHS` is that map, and
`tests/test_form_model.py` fails if the catalog grows a parameter the map does
not cover — so a new input cannot appear in the docs and quietly miss the form.

Nothing here imports Qt. The model is testable, and usable, without a display.
"""

from __future__ import annotations

import copy
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

from ..project import (
    DEFAULT_FORSTER_RADIUS,
    ProjectValidationError,
    template_project,
    validate_project,
)
from ..i18n import resolve_locale, translate_catalog, ui

__all__ = [
    "FIELD_PATHS",
    "NON_SCALAR_PARAMETERS",
    "QUENCHER_COLUMNS",
    "ProjectFormModel",
]

#: Attribute name → path in the project dict. Every scalar the catalog
#: documents appears here; the order is the order the catalog lists them, which
#: is also the order the form shows them.
FIELD_PATHS: "OrderedDict[str, str]" = OrderedDict(
    (
        ("pdb", "pdb"),
        ("attachment_chain", "attachment.chain"),
        ("attachment_residue", "attachment.residue"),
        ("attachment_atom", "attachment.atom"),
        ("av_linker_length", "av_parameter.linker_length"),
        ("av_linker_width", "av_parameter.linker_width"),
        ("av_radius1", "av_parameter.radius1"),
        ("dg", "dg"),
        ("tau0", "tau0"),
        ("donor_D", "fret.dyes.0.D"),
        ("critical_distance", "critical_distance"),
        ("t_max", "t_max"),
        ("t_step", "t_step"),
        ("n_photons", "n_photons"),
        ("n_bins", "n_bins"),
        ("parallel_trajectories", "parallel_trajectories"),
        ("slow_radius", "slow_radius"),
        ("save_avs", "save_avs"),
        ("output_file", "output_file"),
        ("skip_frame", "skip_frame"),
        ("coarse_grained", "coarse_grained"),
        ("random_seed", "random_seed"),
        ("fret_enabled", "fret.enabled"),
        ("acceptor_chain", "fret.dyes.1.attachment.chain"),
        ("acceptor_residue", "fret.dyes.1.attachment.residue"),
        ("acceptor_atom", "fret.dyes.1.attachment.atom"),
        ("acceptor_linker_length", "fret.dyes.1.av_parameter.linker_length"),
        ("acceptor_linker_width", "fret.dyes.1.av_parameter.linker_width"),
        ("acceptor_radius1", "fret.dyes.1.av_parameter.radius1"),
        ("acceptor_D", "fret.dyes.1.D"),
        ("acceptor_slow_radius", "fret.dyes.1.slow_radius"),
        ("acceptor_dynamics", "fret.acceptor_dynamics"),
        ("fret_kappa2", "fret.kappa2"),
    )
)

#: Catalog entries that are *not* a scalar field, and where they live instead.
#: Listed so the sync test can tell "deliberately not a spin box" from "someone
#: added a parameter and forgot the form".
NON_SCALAR_PARAMETERS = {
    "fret.R0_matrix": "the fret_R0 property — a 2-dye matrix is one number",
    "amino_acid_interactions": "the quencher table",
    "amino_acid_interactions.kQ": "quencher table column",
    "amino_acid_interactions.quench_radius": "quencher table column",
    "amino_acid_interactions.quench_atoms": "quencher table column",
    "amino_acid_interactions.slow_factor": "quencher table column",
    # The acceptor's own chemistry. Only `slow_factor` is read (QuEst reports
    # the donor's decay), and it defaults to the donor's table, so it is not
    # worth a second 20-row grid in the form until someone needs to edit it.
    "fret.dyes.1.amino_acid_interactions": "defaults to the donor's table",
}

#: Columns of the quencher table, in display order. `key` is also the key in the
#: per-residue `amino_acid_interactions` entry, except `residue` (the map key).
QUENCHER_COLUMNS = ("residue", "kQ", "quench_radius", "quench_atoms", "slow_factor")


def _get_path(mapping: Any, path: str) -> Any:
    """Read a dotted path; integer-looking parts index into sequences."""
    node = mapping
    for part in path.split("."):
        if node is None:
            return None
        if isinstance(node, (list, tuple)):
            index = int(part)
            node = node[index] if -len(node) <= index < len(node) else None
        else:
            node = node.get(part)
    return node


def _set_path(mapping: Any, path: str, value: Any) -> None:
    """Write a dotted path, creating the intermediate dicts it needs.

    A missing *list* element is an error rather than something to create: the
    dye list has a fixed meaning (0 = donor, 1 = acceptor) and silently growing
    it would produce a project that validates but simulates something else.
    """
    parts = path.split(".")
    node = mapping
    for part in parts[:-1]:
        if isinstance(node, (list, tuple)):
            node = node[int(part)]
        else:
            if part not in node or not isinstance(node[part], (dict, list)):
                node[part] = OrderedDict()
            node = node[part]
    last = parts[-1]
    if isinstance(node, (list, tuple)):
        node[int(last)] = value
    else:
        node[last] = value


class ProjectFormModel:
    """A project dict presented as flat attributes, for AutoForm to bind to.

    Parameters
    ----------
    project : dict, optional
        The project to edit. Copied, never aliased — editing a form must not
        mutate a caller's dict behind its back. Defaults to
        :func:`quest.project.template_project`.
    locale : str, optional
        Language for labels, help and status text. Defaults to
        :func:`quest.i18n.resolve_locale` — `QUEST_LOCALE`, else the operating
        system's language, else English. Only presentation changes; the project
        the form produces is identical in every language.

    Examples
    --------
    >>> model = ProjectFormModel()
    >>> model.tau0 = 3.5
    >>> model.project["tau0"]
    3.5
    """

    def __init__(self, project: dict | None = None, locale: str | None = None) -> None:
        # Bypass __setattr__: the field machinery needs _project to exist first.
        object.__setattr__(self, "_project", copy.deepcopy(project or template_project()))
        object.__setattr__(self, "_listeners", [])
        object.__setattr__(self, "locale", resolve_locale(locale))
        #: The last :class:`~quest.core.DecaySimulationResult`, or None.
        self.results = None
        #: One line of human-readable state, shown read-only in the form.
        self.status = self._t("status.idle")

    # -- AutoForm plumbing --------------------------------------------------

    def _t(self, key: str) -> str:
        """One UI string in this model's language."""
        return ui(key, self.locale)

    def view_spec(self):
        """The generated view spec for this model's language."""
        from .generate_view_spec import view_spec_path

        path = view_spec_path(self.locale)
        if not path.is_file():
            path = view_spec_path()

        try:
            from chisurf.core.dataspec import load_view_spec
            return load_view_spec(path)
        except Exception:
            return json.loads(path.read_text(encoding="utf-8"))


    def set_changed_callback(self, callback) -> None:
        """AutoForm's hook: it wants to know when the model changed."""
        self.add_listener(callback)

    # -- the flat attribute surface ---------------------------------------

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes not found normally, so real methods and
        # properties keep working.
        path = FIELD_PATHS.get(name)
        if path is None:
            raise AttributeError(name)
        return _get_path(self._project, path)

    def __setattr__(self, name: str, value: Any) -> None:
        path = FIELD_PATHS.get(name)
        if path is None:
            object.__setattr__(self, name, value)
            return
        _set_path(self._project, path, value)
        self._notify()

    def __dir__(self) -> Iterable[str]:
        return sorted(set(super().__dir__()) | set(FIELD_PATHS))

    # -- the project ------------------------------------------------------

    @property
    def project(self) -> dict:
        """The live project dict. Editing it edits the form."""
        return self._project

    def to_project_dict(self) -> dict:
        """A copy of the project, safe to hand to the service layer."""
        return copy.deepcopy(self._project)

    def load_project_dict(self, project: dict) -> None:
        """Replace the whole state with *project*."""
        object.__setattr__(self, "_project", copy.deepcopy(project))
        self._notify()

    def validate(self) -> list[str]:
        """Problems with the current project; empty when it is simulatable."""
        try:
            validate_project(self._project)
        except ProjectValidationError as error:
            return [str(error)]
        return []

    # -- FRET: one number, stored as a matrix -----------------------------

    @property
    def fret_R0(self) -> float:
        """Förster radius in **Ångström** (`fret.R0_matrix[0][1]`).

        The project stores a full donor/acceptor matrix because there may one
        day be more than two dyes. With two there is a single off-diagonal
        number, and showing a 2×2 grid of which half is `null` helps nobody.
        """
        value = _get_path(self._project, "fret.R0_matrix.0.1")
        return DEFAULT_FORSTER_RADIUS if value is None else float(value)

    @fret_R0.setter
    def fret_R0(self, value: float) -> None:
        r0 = float(value)
        _set_path(self._project, "fret.R0_matrix", [[None, r0], [r0, None]])
        self._notify()

    # -- the quencher table ------------------------------------------------

    def quencher_rows(self) -> list[dict]:
        """One row per amino acid, for the table section.

        Only residues that actually quench are worth a row by default, but the
        table shows all of them: which residues are *not* quenchers is part of
        the model, and hiding them makes a zero look like an omission.
        """
        interactions = self._project.get("amino_acid_interactions") or {}
        rows = []
        for residue in sorted(interactions):
            entry = interactions[residue] or {}
            atoms = entry.get("quench_atoms") or []
            rows.append(
                {
                    "residue": residue,
                    "kQ": entry.get("kQ", 0.0),
                    "quench_radius": entry.get("quench_radius"),
                    "quench_atoms": ", ".join(atoms),
                    "slow_factor": entry.get("slow_factor", 0.1),
                }
            )
        return rows

    def update_quencher_cell(self, row_index: int, column_key: str, value: Any) -> None:
        """Write one edited cell back into `amino_acid_interactions`.

        Bad input is dropped rather than raising: the user is mid-typing, and a
        traceback out of a cell editor is not a useful way to say "not a number".
        """
        rows = self.quencher_rows()
        if not 0 <= row_index < len(rows):
            return
        residue = rows[row_index]["residue"]
        entry = self._project.setdefault("amino_acid_interactions", {}).setdefault(
            residue, {}
        )
        text = "" if value is None else str(value).strip()
        if column_key == "quench_atoms":
            entry["quench_atoms"] = [a.strip() for a in text.split(",") if a.strip()]
        elif column_key == "quench_radius":
            # Blank means "use the global critical distance", which is null in
            # the schema — not zero, which would switch quenching off.
            entry["quench_radius"] = None if not text else float(text)
        elif column_key in ("kQ", "slow_factor"):
            try:
                entry[column_key] = float(text)
            except ValueError:
                return
        else:
            return
        self._notify()

    # -- structure preview -------------------------------------------------

    def preview_models(self) -> list[str]:
        """Paths for the 3D viewer: the structure, when there is a readable one."""
        pdb = self._project.get("pdb")
        if pdb and Path(str(pdb)).is_file():
            return [str(pdb)]
        return []

    # -- persistence: the saved file is a project ---------------------------

    def save_project(self, path: str | Path) -> Path:
        """Write the state as project JSON — the CLI's and web UI's format."""
        destination = Path(path)
        destination.write_text(
            json.dumps(self._project, indent=2, default=str), encoding="utf8"
        )
        return destination

    def load_project(self, path: str | Path) -> None:
        """Read a project written by any surface."""
        with Path(path).open("r", encoding="utf8") as handle:
            self.load_project_dict(json.load(handle))

    # The two dialog actions are what the toolbar buttons call. Qt is imported
    # inside them, not at module scope: only a click can reach this code, and
    # the rest of the model must stay importable where there is no display.

    def save_project_dialog(self) -> Path | None:
        from qtpy import QtWidgets

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            None,
            self._t("dialog.save_project"),
            "project.json",
            self._t("filter.project"),
        )
        if not path:
            return None
        saved = self.save_project(path)
        self.status = self._t("status.saved").format(name=saved.name)
        self._notify()
        return saved

    def load_project_dialog(self) -> Path | None:
        from qtpy import QtWidgets

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None,
            self._t("dialog.load_project"),
            "",
            f"{self._t('filter.project')};;{self._t('filter.all')}",
        )
        if not path:
            return None
        self.load_project(path)
        self.status = self._t("status.loaded").format(name=Path(path).name)
        self._notify()
        return Path(path)

    # -- running, and what came out ----------------------------------------

    def run_simulation(self) -> Any:
        """Simulate the current project through the service layer.

        The view-model calls `quest.api`, exactly as the CLI and the web
        backend do — the form contributes no physics of its own. That is the
        rule the GUI broke before, when it shipped its own quenching chemistry.

        Runs synchronously, so a long simulation blocks the form (`LAY-07`).
        `save_outputs=False` keeps a GUI run from littering `jobs/` in whatever
        directory the app happened to start in.
        """
        from ..api import simulate_project

        self.status = self._t("status.running")
        self._notify()
        try:
            self.results = simulate_project(self._project, save_outputs=False)
        except Exception as error:
            self.results = None
            self.status = self._t("status.failed").format(error=error)
            self._notify()
            raise
        self.status = self._describe(self.results)
        self._notify()
        return self.results

    @staticmethod
    def _describe(result: Any) -> str:
        parts = [
            f"QY = {result.quantum_yield_donor:.3f}",
            f"⟨τ⟩ = {result.lifetime_donor:.2f} ns",
        ]
        if result.fret_efficiency is not None:
            parts.append(f"E = {result.fret_efficiency:.3f}")
        return "   ".join(parts)

    def decay_series(self) -> list[dict]:
        """Donor (and FRET) decay histograms, for the decay plot."""
        result = getattr(self, "results", None)
        if result is None:
            return []
        series = [
            {"x": result.time, "y": result.donor_counts, "name": self._t("plot.decay.donor")}
        ]
        if result.fret_counts is not None:
            series.append(
                {"x": result.time, "y": result.fret_counts, "name": self._t("plot.decay.fret")}
            )
        return series

    def trajectory_series(self) -> list[dict]:
        """Dye distance from the mean position over the trajectory."""
        result = getattr(self, "results", None)
        if result is None or result.trajectory_time_ns is None:
            return []
        return [
            {
                "x": result.trajectory_time_ns,
                "y": result.trajectory_distance_to_mean,
                "name": "|r − ⟨r⟩|",
            }
        ]

    def autocorrelation_series(self) -> list[dict]:
        """Autocorrelation of the dye position — how fast the dye forgets."""
        result = getattr(self, "results", None)
        if result is None or result.trajectory_autocorr is None:
            return []
        return [
            {
                "x": result.trajectory_autocorr_lag_ns,
                "y": result.trajectory_autocorr,
                "name": "ACF",
            }
        ]

    # -- change notification ------------------------------------------------

    def add_listener(self, callback) -> None:
        """Call *callback* after any field changes (the form uses this to redraw)."""
        self._listeners.append(callback)

    def _notify(self) -> None:
        for callback in list(self._listeners):
            try:
                callback()
            except Exception:  # pragma: no cover - a bad listener must not
                pass          # take the form down mid-edit

    # -- help text, from the one place it is written ------------------------

    def catalog_entry(self, path: str) -> dict:
        """The catalog entry for a project path, in this model's language."""
        return dict(translate_catalog(self.locale).get("parameters", {}).get(path, {}))
