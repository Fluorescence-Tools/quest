"""Tests for the high-level quest.api module.

These tests exercise the public API functions that are intended to be
used from scripts or Jupyter notebooks:

- quest.api.load_project
- quest.api.simulate_project

The tests construct a small in-memory project dictionary and use a
bundled PDB file so they do not depend on the GUI.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any

import json

# Ensure the vendored "quest" package (in modules/quest) is importable
# tests are in modules/quest/tests, so parents[3] is the repository root
ROOT = Path(__file__).resolve().parents[3]
QUEST_ROOT = ROOT / "modules" / "quest"
if str(QUEST_ROOT) not in sys.path:
    sys.path.insert(0, str(QUEST_ROOT))

from quest.api import load_project, simulate_project, DecaySimulationResult  # type: ignore[import]


PDB_PATH = ROOT / "test" / "data" / "atomic_coordinates" / "pdb_files" / "148l.pdb"


def _make_minimal_project(with_fret: bool) -> Dict[str, Any]:
    """Return a small but complete QuEst project dictionary for testing.

    The parameters are chosen so that simulations run quickly while still
    exercising the core diffusion and decay logic. Attachment and
    quencher positions are selected from the bundled 148l.pdb file.
    """

    project: Dict[str, Any] = {}

    project["pdb"] = str(PDB_PATH)
    project["attachment"] = {
        "chain": "E",
        "residue": 10,
        "atom": "CB",
    }

    project["av_parameter"] = {
        "linker_length": 21.5,
        "linker_width": 0.5,
        "radius1": 3.5,
    }

    project["dg"] = 0.5
    project["slow_radius"] = 8.5

    project["t_max"] = 200.0
    project["t_step"] = 0.02
    project["n_photons"] = 10_000
    project["n_bins"] = 512
    project["skip_frame"] = 10

    project["tau0"] = 4.2
    project["kQ"] = 2.0
    project["slow_fact"] = 0.1
    project["critical_distance"] = 6.5
    project["quencher"] = {"TYR": ["CB"]}

    project["output_file"] = "test_out"
    project["save_avs"] = False
    project["all_quencher_atoms"] = True
    project["sticky_mode"] = "quencher"

    # Dyes and per-dye diffusion coefficients
    dyes = []

    donor = {
        "name": "D0",
        "role": "donor",
        "attachment": project["attachment"],
        "av_parameter": project["av_parameter"],
        "D": 7.0,
    }
    dyes.append(donor)

    if with_fret:
        acceptor = {
            "name": "A0",
            "role": "acceptor",
            "attachment": {
                "chain": "E",
                "residue": 18,
                "atom": "CB",
            },
            "av_parameter": project["av_parameter"],
            "D": 7.0,
        }
        dyes.append(acceptor)

        project["fret"] = {
            "enabled": True,
            "dyes": dyes,
            "R0_matrix": [
                [None, 52.0],
                [52.0, None],
            ],
        }
    else:
        project["fret"] = {
            "enabled": False,
            "dyes": dyes,
            "R0_matrix": [],
        }

    return project


def test_simulate_project_donor_only() -> None:
    """Simulate a donor-only decay and validate basic properties.

    This test checks that the API returns arrays of matching length,
    a reasonable quantum yield in the (0, 1] range, and a valid
    collision fraction.
    """

    project = _make_minimal_project(with_fret=False)
    result = simulate_project(project)

    assert isinstance(result, DecaySimulationResult)

    assert result.time.ndim == 1
    assert result.donor_counts.ndim == 1
    assert result.time.shape == result.donor_counts.shape

    assert result.fret_counts is None
    assert 0.0 < result.quantum_yield_donor <= 1.0

    assert 0.0 <= result.collisions_fraction <= 1.0


def test_load_project_roundtrip(tmp_path: Path) -> None:
    """Write a minimal project to disk and load it via load_project.

    This test ensures that the JSON loader returns a dictionary with the
    expected keys and values, and that it can be passed directly into
    :func:`simulate_project`.
    """

    project = _make_minimal_project(with_fret=False)
    target = tmp_path / "proj.quest.json"
    target.write_text(json.dumps(project), encoding="utf8")

    loaded = load_project(target)

    assert loaded["pdb"] == project["pdb"]
    assert loaded["attachment"]["residue"] == project["attachment"]["residue"]

    result = simulate_project(loaded)
    assert isinstance(result, DecaySimulationResult)


def test_simulate_project_with_fret() -> None:
    """Simulate a donor+FRET decay and validate FRET-specific outputs.

    The test enables FRET, runs the simulation, and verifies that a
    second decay curve is produced together with a FRET quantum yield
    and efficiency estimate.
    """

    project = _make_minimal_project(with_fret=True)
    result = simulate_project(project)

    assert isinstance(result, DecaySimulationResult)

    assert result.fret_counts is not None
    assert result.fret_counts.shape == result.donor_counts.shape

    assert result.quantum_yield_fret is not None
    assert 0.0 <= result.quantum_yield_fret <= 1.0

    if result.fret_efficiency is not None:
        assert 0.0 <= result.fret_efficiency <= 1.0
