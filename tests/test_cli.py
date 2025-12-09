import json
from pathlib import Path

import pytest

from quest import api
from quest import cli


class DummyResult:
    def __init__(self, time, donor_counts, fret_counts=None):
        self.time = time
        self.donor_counts = donor_counts
        self.fret_counts = fret_counts
        self.quantum_yield_donor = 1.0
        self.quantum_yield_fret = None
        self.collisions_fraction = 0.1
        self.fret_efficiency = None
        self.project = {"dummy": True}


def test_set_path_nested():
    target = {}
    cli._set_path(target, "a.b.c", 5)
    assert target == {"a": {"b": {"c": 5}}}


def test_clone_overrides():
    base = {"a": {"b": 1}, "x": 2}
    clone = cli._clone_with_overrides(base, [("a.b", 3), ("y", 4)])
    assert clone["a"]["b"] == 3
    assert clone["x"] == 2
    assert clone["y"] == 4
    # original untouched
    assert base == {"a": {"b": 1}, "x": 2}


def test_build_runs_cartesian():
    base = {"p": 1}
    runs = cli._build_runs(base, [("a", [1, 2]), ("b", ["x", "y"])])
    tags = [t for _proj, t in runs]
    assert len(runs) == 4
    assert {"a": 1, "b": "x"} in tags
    assert {"a": 2, "b": "y"} in tags


def test_resolve_output_path_multiple(tmp_path):
    project_path = tmp_path / "proj.quest.json"
    project_path.write_text("{}", encoding="utf8")
    out = cli._resolve_output_path(
        output_path=None,
        project_path=project_path,
        idx=0,
        tags={"a": 1, "b": "x"},
        multiple=True,
    )
    assert "a-1" in out.stem and "b-x" in out.stem
    assert out.suffix == ".csv"


def test_validate_project_missing_key_raises():
    bad = {"tau0": 1.0}  # missing many keys
    with pytest.raises(SystemExit):
        # click exceptions propagate as SystemExit in tests
        cli._validate_project(bad)


def test_simulate_set_and_grid(monkeypatch, tmp_path):
    project = {
        "tau0": 4.2,
        "kQ": 0.0,
        "n_photons": 10,
        "t_max": 10.0,
        "t_step": 0.1,
        "dg": 0.0,
        "slow_fact": 0.1,
        "critical_distance": 8.5,
        "av_parameter": 1000,
        "quencher": "TRY",
        "pdb": str(tmp_path / "dummy.pdb"),
        "fret": {"dyes": [{"D": 7.5}]},
    }
    Path(project["pdb"]).write_text("PDB", encoding="utf8")
    proj_path = tmp_path / "proj.json"
    proj_path.write_text(json.dumps(project), encoding="utf8")

    # stub api.load_project and simulate_project
    monkeypatch.setattr(api, "load_project", lambda path: project)

    def fake_sim(project, pdb_path=None, nbins=None, tac_range=None):
        return DummyResult(time=[0, 1], donor_counts=[1, 2])

    monkeypatch.setattr(api, "simulate_project", fake_sim)

    runner = cli.click.testing.CliRunner()
    result = runner.invoke(
        cli.cli,
        [
            "simulate",
            "-p",
            str(proj_path),
            "--set",
            "tau0=3.0",
            "--grid",
            "fret.dyes.0.D=7.5,8.5",
            "-o",
            str(tmp_path / "out.json"),
        ],
    )
    assert result.exit_code == 0
    # two runs written
    outs = list(tmp_path.glob("out_fret.dyes.0.D-*.json"))
    assert len(outs) == 2
