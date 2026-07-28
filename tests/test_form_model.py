"""One JSON state behind every surface.

`ProjectFormModel` is what the Qt form binds to, and its state is a QuEst
*project* — the same document the CLI runs, the web backend posts and
`quest.api` validates. These tests run **without Qt**, which is the first thing
worth proving: the state of the desktop form is not trapped inside a widget.

What they pin:

- the field map covers exactly what the parameter catalog documents, so a new
  parameter cannot reach the docs and the web UI while missing from the form;
- editing a field edits the project, with no conversion step in between (the
  hand-written `to_project_dict`/`load_project_dict` pair is where the GUI
  drifted away from the CLI before);
- what the form saves is what the other surfaces accept.
"""

from __future__ import annotations

import json

import pytest

from quest.gui.form_model import (
    FIELD_PATHS,
    NON_SCALAR_PARAMETERS,
    ProjectFormModel,
)
from quest.project import DEFAULT_FORSTER_RADIUS, template_project
from quest.settings import parameter_catalog


def test_the_model_needs_no_qt() -> None:
    """The desktop form's state must not be trapped inside a widget.

    Checked in a subprocess: this suite runs in environments that have Qt and
    other modules import it, so `sys.modules` in *this* process proves nothing.
    """
    import subprocess
    import sys
    import textwrap

    probe = textwrap.dedent(
        """
        import sys
        from quest.gui.form_model import ProjectFormModel
        model = ProjectFormModel()
        model.tau0 = 1.0
        assert model.project["tau0"] == 1.0
        leaked = [n for n in sys.modules if n.split(".")[0] in
                  ("PyQt5", "PyQt6", "PySide2", "PySide6", "qtpy")]
        assert not leaked, leaked
        print("clean")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "clean" in result.stdout


class TestTheFieldMapMatchesTheCatalog:
    """The catalog is the single description; the form must cover all of it."""

    def test_every_documented_parameter_has_a_home(self) -> None:
        documented = set(parameter_catalog()["parameters"])
        covered = set(FIELD_PATHS.values()) | set(NON_SCALAR_PARAMETERS)
        missing = documented - covered
        assert not missing, (
            f"documented but not in the form: {sorted(missing)} — add it to "
            "FIELD_PATHS, or to NON_SCALAR_PARAMETERS if a table owns it"
        )

    def test_the_form_invents_nothing(self) -> None:
        documented = set(parameter_catalog()["parameters"])
        invented = (set(FIELD_PATHS.values()) | set(NON_SCALAR_PARAMETERS)) - documented
        assert not invented, (
            f"in the form but undocumented: {sorted(invented)} — a control with "
            "no catalog entry has no help text and no docs"
        )

    def test_field_names_are_identifiers(self) -> None:
        for attr in FIELD_PATHS:
            assert attr.isidentifier(), attr

    def test_every_field_reads_back(self) -> None:
        model = ProjectFormModel()
        for attr in FIELD_PATHS:
            getattr(model, attr)  # must not raise


class TestEditingTheFormEditsTheProject:
    def test_a_scalar(self) -> None:
        model = ProjectFormModel()
        model.tau0 = 3.25
        assert model.project["tau0"] == 3.25

    def test_a_nested_path(self) -> None:
        model = ProjectFormModel()
        model.av_linker_length = 25.0
        assert model.project["av_parameter"]["linker_length"] == 25.0

    def test_a_path_through_the_dye_list(self) -> None:
        model = ProjectFormModel()
        model.acceptor_chain = "B"
        model.acceptor_D = 4.0
        acceptor = model.project["fret"]["dyes"][1]
        assert acceptor["attachment"]["chain"] == "B"
        assert acceptor["D"] == 4.0
        # Donor untouched: the two dyes are separate rows of the same list.
        assert model.project["fret"]["dyes"][0]["attachment"]["chain"] == "A"

    def test_an_unknown_attribute_is_an_ordinary_one(self) -> None:
        model = ProjectFormModel()
        model.scratch = 7
        assert model.scratch == 7
        assert "scratch" not in model.project

    def test_the_caller_dict_is_not_aliased(self) -> None:
        project = template_project()
        model = ProjectFormModel(project)
        model.tau0 = 99.0
        assert project["tau0"] != 99.0


class TestTheForsterRadius:
    def test_it_defaults_to_angstrom(self) -> None:
        assert ProjectFormModel().fret_R0 == pytest.approx(DEFAULT_FORSTER_RADIUS)

    def test_setting_it_keeps_the_matrix_symmetric(self) -> None:
        model = ProjectFormModel()
        model.fret_R0 = 47.5
        assert model.project["fret"]["R0_matrix"] == [[None, 47.5], [47.5, None]]

    def test_a_missing_matrix_reads_as_the_default(self) -> None:
        model = ProjectFormModel()
        model.project["fret"]["R0_matrix"] = [[None, None], [None, None]]
        assert model.fret_R0 == pytest.approx(DEFAULT_FORSTER_RADIUS)


class TestTheQuencherTable:
    def test_it_lists_every_amino_acid(self) -> None:
        model = ProjectFormModel()
        rows = model.quencher_rows()
        residues = {row["residue"] for row in rows}
        assert set(model.project["amino_acid_interactions"]) == residues
        assert len(rows) > 15

    def test_atoms_render_as_text_and_parse_back(self) -> None:
        model = ProjectFormModel()
        rows = model.quencher_rows()
        trp = next(i for i, row in enumerate(rows) if row["residue"] == "TRP")
        assert ", " in rows[trp]["quench_atoms"]

        model.update_quencher_cell(trp, "quench_atoms", " NE1 , CZ2 ,")
        assert model.project["amino_acid_interactions"]["TRP"]["quench_atoms"] == [
            "NE1",
            "CZ2",
        ]

    def test_an_out_of_range_row_is_ignored(self) -> None:
        model = ProjectFormModel()
        model.update_quencher_cell(9999, "kQ", "1.0")  # must not raise

    def test_an_unknown_column_is_ignored(self) -> None:
        model = ProjectFormModel()
        before = json.dumps(model.project, sort_keys=True, default=str)
        model.update_quencher_cell(0, "not_a_column", "1.0")
        assert json.dumps(model.project, sort_keys=True, default=str) == before


class TestSurfacesShareTheState:
    """The objective: one JSON, whichever surface wrote it."""

    def test_the_saved_file_validates_as_a_project(self, tmp_path) -> None:
        from quest.api import load_project, validate_project

        model = ProjectFormModel()
        model.pdb = str(tmp_path / "structure.pdb")
        (tmp_path / "structure.pdb").write_text("ATOM\n", encoding="utf8")

        path = model.save_project(tmp_path / "project.json")
        validate_project(load_project(path))

    def test_it_round_trips_through_a_file(self, tmp_path) -> None:
        model = ProjectFormModel()
        model.tau0 = 1.75
        model.acceptor_atom = "SG"
        model.fret_R0 = 61.0
        path = model.save_project(tmp_path / "project.json")

        restored = ProjectFormModel()
        restored.load_project(path)
        assert restored.to_project_dict() == model.to_project_dict()
        assert restored.tau0 == 1.75
        assert restored.fret_R0 == 61.0

    def test_a_cli_written_project_loads_into_the_form(self, tmp_path) -> None:
        """The other direction: what the CLI writes, the form must open."""
        project = template_project()
        project["tau0"] = 5.5
        project["fret"]["enabled"] = False
        path = tmp_path / "cli.json"
        path.write_text(json.dumps(project), encoding="utf8")

        model = ProjectFormModel()
        model.load_project(path)
        assert model.tau0 == 5.5
        assert model.fret_enabled is False

    def test_the_web_backend_accepts_what_the_form_saves(self, tmp_path) -> None:
        """End to end across surfaces, through the service layer's own methods."""
        pytest.importorskip("fastapi", reason="the web extra is not installed")

        from quest.rpc.services import local_client

        structure = tmp_path / "structure.pdb"
        structure.write_text("ATOM\n", encoding="utf8")

        model = ProjectFormModel()
        model.pdb = str(structure)
        model.tau0 = 2.25

        client = local_client()
        response = client.call("quest.validate", {"project": model.project})
        assert response["ok"] is True, response


class TestStatusAndResults:
    def test_it_starts_with_no_results(self) -> None:
        model = ProjectFormModel()
        assert model.results is None
        assert model.decay_series() == []
        assert model.trajectory_series() == []
        assert model.autocorrelation_series() == []
        assert "Not simulated" in model.status

    def test_listeners_fire_on_edit(self) -> None:
        model = ProjectFormModel()
        seen = []
        model.add_listener(lambda: seen.append(1))
        model.tau0 = 2.0
        model.fret_R0 = 50.0
        model.update_quencher_cell(0, "kQ", "0.5")
        assert len(seen) == 3

    def test_a_broken_listener_does_not_break_editing(self) -> None:
        model = ProjectFormModel()
        model.add_listener(lambda: 1 / 0)
        model.tau0 = 2.0  # must not raise
        assert model.tau0 == 2.0
