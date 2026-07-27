"""The Qt form: rendered from the parameter catalog, bound to a project.

These run only where a Qt binding *and* ChiSurf exist — QuEst's desktop surface
is rendered by the host's AutoForm, so there is nothing to test without it. The
default development interpreter has neither, so they skip there:

    QT_QPA_PLATFORM=offscreen \\
      /Users/tpeulen/mambaforge/envs/arm64/bin/python -m pytest tests/test_gui_widget.py

Construction alone proves little, so these assert the things that have actually
gone wrong: the widget simulating different chemistry from every other surface,
a Förster radius silently in the wrong unit, and a parameter existing in the
model but reachable nowhere in the UI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("qtpy", reason="no Qt binding in this environment")
QtWidgets = pytest.importorskip("qtpy.QtWidgets", reason="no Qt binding")

from quest.chisurf_host import autoform_available  # noqa: E402
from quest.core.dye_diffusion import (  # noqa: E402
    DEFAULT_QUENCHER_ATOMS,
    REFERENCE_PET_QUENCHING,
)
from quest.gui.form_model import FIELD_PATHS  # noqa: E402
from quest.project import DEFAULT_FORSTER_RADIUS, template_project  # noqa: E402

pytestmark = pytest.mark.skipif(
    not autoform_available(),
    reason="the Qt form is rendered by ChiSurf's AutoForm (set CHISURF_ROOT)",
)


@pytest.fixture(scope="module")
def app():
    yield QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def widget(app):
    from quest.gui import TransientDecayGenerator

    generator = TransientDecayGenerator()
    yield generator
    generator.close()


class TestTheFormIsGenerated:
    """LAY-06: the layout is data, not a hand-built widget tree."""

    def test_it_renders(self, widget) -> None:
        assert widget.form is not None, "AutoForm did not build the form"
        widget.resize(1400, 900)
        pixmap = widget.grab()
        assert not pixmap.isNull()
        assert pixmap.width() > 800 and pixmap.height() > 500

    def test_no_ui_files_remain(self) -> None:
        """The whole point of the migration; a new `.ui` is a regression."""
        gui = Path(__file__).resolve().parents[1] / "quest" / "gui"
        assert not list(gui.rglob("*.ui")), "a .ui file came back"

    def test_the_widget_is_not_the_model(self) -> None:
        """LAY-02: presentation holds a domain object, it does not *be* one."""
        from quest.core.dye_diffusion import DonorDecay
        from quest.gui import TransientDecayGenerator

        assert DonorDecay not in TransientDecayGenerator.__mro__

    def test_the_model_is_headless(self, widget) -> None:
        """What the split buys: the state runs with no widget attached."""
        assert not isinstance(widget.model, QtWidgets.QWidget)


class TestTheStateIsTheProject:
    """No conversion step means no place for the GUI to drift from the CLI."""

    def test_every_declared_field_is_readable(self, widget) -> None:
        for attr in FIELD_PATHS:
            assert hasattr(widget.model, attr), f"{attr} is declared but unreadable"

    def test_editing_a_field_edits_the_project(self, widget) -> None:
        widget.model.tau0 = 3.14
        assert widget.to_project_dict()["tau0"] == 3.14

        widget.model.acceptor_residue = 42
        acceptor = widget.to_project_dict()["fret"]["dyes"][1]
        assert acceptor["attachment"]["residue"] == 42

    def test_loading_a_project_redraws_the_form(self, widget) -> None:
        project = template_project()
        project["tau0"] = 2.5
        project["n_photons"] = 1234
        widget.load_project_dict(project)

        assert widget.model.tau0 == 2.5
        assert widget.model.n_photons == 1234
        # The controls must show it, not merely the dict hold it.
        assert not widget.grab().isNull()

    def test_a_loaded_project_is_not_aliased(self, widget) -> None:
        project = template_project()
        widget.load_project_dict(project)
        widget.model.tau0 = 9.9
        assert project["tau0"] != 9.9, "editing the form mutated the caller's dict"


class TestQuenchingChemistry:
    """The GUI must simulate what the CLI, API and web UI simulate."""

    def test_quenchers_sit_on_the_redox_active_moiety(self, widget) -> None:
        interactions = widget.to_project_dict()["amino_acid_interactions"]
        for residue in REFERENCE_PET_QUENCHING:
            assert interactions[residue]["quench_atoms"] == list(
                DEFAULT_QUENCHER_ATOMS[residue]
            ), residue
        # The specific regression: MET's quencher is the thioether sulfur and
        # TRP's is the indole ring — not CB, as the old settings file said.
        assert interactions["MET"]["quench_atoms"] == ["SD"]
        assert "NE1" in interactions["TRP"]["quench_atoms"]

    def test_each_residue_carries_its_own_rate(self, widget) -> None:
        interactions = widget.to_project_dict()["amino_acid_interactions"]
        rates = {r: interactions[r]["kQ"] for r in REFERENCE_PET_QUENCHING}
        assert rates == {
            residue: reference["kQ"]
            for residue, reference in REFERENCE_PET_QUENCHING.items()
        }
        # Not one global rate for everything, which is what 2.0 everywhere was.
        assert len(set(rates.values())) > 1

    def test_contact_radius_matches_the_project_default(self, widget) -> None:
        assert widget.model.critical_distance == pytest.approx(8.5)

    def test_the_table_edits_the_chemistry(self, widget) -> None:
        rows = widget.model.quencher_rows()
        trp = next(i for i, row in enumerate(rows) if row["residue"] == "TRP")

        widget.model.update_quencher_cell(trp, "kQ", "9.5")
        interactions = widget.to_project_dict()["amino_acid_interactions"]
        assert interactions["TRP"]["kQ"] == 9.5

        widget.model.update_quencher_cell(trp, "quench_atoms", "NE1, CZ2")
        assert widget.to_project_dict()["amino_acid_interactions"]["TRP"][
            "quench_atoms"
        ] == ["NE1", "CZ2"]

        # A blank radius means "fall back to critical_distance" — null, not
        # zero, which would switch quenching off entirely.
        widget.model.update_quencher_cell(trp, "quench_radius", "")
        assert (
            widget.to_project_dict()["amino_acid_interactions"]["TRP"]["quench_radius"]
            is None
        )

        # Garbage is dropped rather than raised out of a cell editor mid-typing.
        widget.model.update_quencher_cell(trp, "kQ", "not a number")
        assert widget.to_project_dict()["amino_acid_interactions"]["TRP"]["kQ"] == 9.5


class TestForsterRadius:
    def test_it_is_angstrom(self, widget) -> None:
        """A nanometre R0 makes FRET vanish silently; it happened once."""
        assert widget.model.fret_R0 == pytest.approx(DEFAULT_FORSTER_RADIUS)
        assert widget.model.fret_R0 > 10, "an R0 that small is nm, not Å"

    def test_it_round_trips_as_a_symmetric_matrix(self, widget) -> None:
        widget.model.fret_R0 = 58.0
        matrix = widget.to_project_dict()["fret"]["R0_matrix"]
        assert matrix == [[None, 58.0], [58.0, None]]

    def test_the_control_accepts_the_angstrom_range(self) -> None:
        """The spin box must not clamp R0 into a nanometre-sized range."""
        from quest.gui.generate_view_spec import build_view_spec

        def find(sections):
            for section in sections:
                if section.get("attr") == "fret_R0":
                    return section
                found = find(section.get("sections", []))
                if found:
                    return found
            return None

        field = find(build_view_spec()["sections"])
        assert field is not None, "R0 has no control"
        assert field["minimum"] <= 20.0 and field["maximum"] >= 80.0


class TestHostIntegration:
    def test_the_menu_actions_exist(self, widget) -> None:
        """ChiSurf's quenching_estimator plugin connects its File menu to these."""
        for slot in ("onLoadPDB", "onSaveProject", "onLoadProject", "update_all"):
            assert callable(getattr(widget, slot))
