from __future__ import annotations

import json
import sys
from pathlib import Path

from click.testing import CliRunner

# Ensure vendored quest package (modules/quest) is on sys.path
ROOT = Path(__file__).resolve().parents[3]
QUEST_ROOT = ROOT / "modules" / "quest"
if str(QUEST_ROOT) not in sys.path:
    sys.path.insert(0, str(QUEST_ROOT))

from quest.cli import cli  # type: ignore[import]


def _make_minimal_project() -> dict:
    project: dict = {}
    project["pdb"] = str(ROOT / "test" / "data" / "atomic_coordinates" / "pdb_files" / "148l.pdb")
    project["attachment"] = {"chain": "E", "residue": 10, "atom": "CB"}
    project["av_parameter"] = {"linker_length": 21.5, "linker_width": 0.5, "radius1": 3.5}
    project["dg"] = 0.5
    project["slow_radius"] = 8.5
    project["t_max"] = 200.0
    project["t_step"] = 0.02
    project["n_photons"] = 5_000  # small to keep simulations light if invoked
    project["n_bins"] = 128
    project["skip_frame"] = 5
    project["tau0"] = 4.2
    project["kQ"] = 2.0
    project["slow_fact"] = 0.1
    project["critical_distance"] = 6.5
    project["quencher"] = {"TYR": ["CB"]}
    project["output_file"] = "test_out"
    project["save_avs"] = False
    project["all_quencher_atoms"] = True
    project["sticky_mode"] = "quencher"
    project["fret"] = {"enabled": False, "dyes": [
        {
            "name": "D0",
            "role": "donor",
            "attachment": project["attachment"],
            "av_parameter": project["av_parameter"],
            "D": 7.0,
        }
    ], "R0_matrix": []}
    return project


def test_cli_template_and_validate(tmp_path: Path) -> None:
    runner = CliRunner()

    # Template (no-fret) should succeed
    result_template = runner.invoke(cli, ["template", "--no-fret"], catch_exceptions=False)
    assert result_template.exit_code == 0
    assert "fret" in result_template.output

    # Write a minimal project and validate it via CLI
    project = _make_minimal_project()
    project_path = tmp_path / "proj.quest.json"
    project_path.write_text(json.dumps(project), encoding="utf8")

    result_validate = runner.invoke(cli, ["validate", "-p", str(project_path)], catch_exceptions=False)
    assert result_validate.exit_code == 0
    assert "valid" in result_validate.output.lower()
