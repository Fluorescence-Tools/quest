from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any

import json

# Ensure vendored quest package (modules/quest) is on sys.path
ROOT = Path(__file__).resolve().parents[3]
QUEST_ROOT = ROOT / "modules" / "quest"
if str(QUEST_ROOT) not in sys.path:
    sys.path.insert(0, str(QUEST_ROOT))

from quest.core import load_project, simulate_project, DecaySimulationResult  # type: ignore[import]


def _make_minimal_project(with_fret: bool) -> Dict[str, Any]:
    """Small helper mirroring test_api but importing quest.core."""

    project: Dict[str, Any] = {}

    project["pdb"] = str(Path(__file__).resolve().parents[3] / "test" / "data" / "atomic_coordinates" / "pdb_files" / "148l.pdb")
    project["attachment"] = {"chain": "E", "residue": 10, "atom": "CB"}

    project["av_parameter"] = {"linker_length": 21.5, "linker_width": 0.5, "radius1": 3.5}
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

    dyes = [
        {
            "name": "D0",
            "role": "donor",
            "attachment": project["attachment"],
            "av_parameter": project["av_parameter"],
            "D": 7.0,
        }
    ]

    if with_fret:
        dyes.append(
            {
                "name": "A0",
                "role": "acceptor",
                "attachment": {"chain": "E", "residue": 18, "atom": "CB"},
                "av_parameter": project["av_parameter"],
                "D": 7.0,
            }
        )
        project["fret"] = {"enabled": True, "dyes": dyes, "R0_matrix": [[None, 52.0], [52.0, None]]}
    else:
        project["fret"] = {"enabled": False, "dyes": dyes, "R0_matrix": []}

    return project


def test_simulate_project_core_imports() -> None:
    project = _make_minimal_project(with_fret=False)
    result = simulate_project(project)

    assert isinstance(result, DecaySimulationResult)
    assert result.time.ndim == 1
    assert result.donor_counts.ndim == 1
    assert result.time.shape == result.donor_counts.shape
    assert result.fret_counts is None
    assert 0.0 < result.quantum_yield_donor <= 1.0
    assert 0.0 <= result.collisions_fraction <= 1.0


def test_load_project_core_roundtrip(tmp_path: Path) -> None:
    project = _make_minimal_project(with_fret=False)
    target = tmp_path / "proj.quest.json"
    target.write_text(json.dumps(project), encoding="utf8")

    loaded = load_project(target)
    assert loaded["pdb"] == project["pdb"]
    assert loaded["attachment"]["residue"] == project["attachment"]["residue"]

    result = simulate_project(loaded)
    assert isinstance(result, DecaySimulationResult)
