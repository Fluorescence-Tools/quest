from __future__ import annotations

import json
import os
import uuid
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from webui.backend.app import main as backend_main
from webui.backend.app.main import app
from quest import jobs as quest_jobs
from quest.core import DecaySimulationResult

AAS = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]


def _aa_interactions() -> dict[str, dict[str, float]]:
    interactions = {aa: {"slow_factor": 1.0, "kQ": 0.0} for aa in AAS}
    interactions["TYR"]["kQ"] = 2.0
    return interactions


def _minimal_project(tmp_path: Path) -> dict:
    pdb = tmp_path / "mock.pdb"
    pdb.write_text("ATOM\n", encoding="utf8")
    return {
        "pdb": str(pdb),
        "tau0": 4.2,
        "n_photons": 1000,
        "t_max": 200.0,
        "t_step": 0.02,
        "dg": 0.5,
        "critical_distance": 6.5,
        "av_parameter": {"linker_length": 21.5, "linker_width": 0.5, "radius1": 3.5},
        "amino_acid_interactions": _aa_interactions(),
        "slow_radius": 8.5,
        "fret": {
            "enabled": False,
            "dyes": [
                {
                    "name": "D0",
                    "D": 7.0,
                    "attachment": {"chain": "E", "residue": 10, "atom": "CB"},
                    "av_parameter": {"linker_length": 21.5, "linker_width": 0.5, "radius1": 3.5},
                }
            ],
        },
        "attachment": {"chain": "E", "residue": 10, "atom": "CB"},
    }


def test_health() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_template() -> None:
    client = TestClient(app)
    response = client.get("/api/template?with_fret=false")
    assert response.status_code == 200
    payload = response.json()["project"]
    assert payload["fret"]["enabled"] is False


def test_validate_success(tmp_path: Path) -> None:
    client = TestClient(app)
    project = _minimal_project(tmp_path)
    response = client.post("/api/validate", data={"project_json": json.dumps(project)})
    assert response.status_code == 200
    assert response.json()["valid"] is True


def test_validate_bad_json() -> None:
    client = TestClient(app)
    response = client.post("/api/validate", data={"project_json": "not-json"})
    assert response.status_code == 400


def test_simulate_uses_adapter(monkeypatch, tmp_path: Path) -> None:
    client = TestClient(app)
    project = _minimal_project(tmp_path)

    def fake_simulate(*args, **kwargs):
        return DecaySimulationResult(
            time=np.array([0.0, 1.0]),
            donor_counts=np.array([10.0, 5.0]),
            fret_counts=None,
            quantum_yield_donor=0.5,
            quantum_yield_fret=None,
            collisions_fraction=0.1,
            fret_efficiency=None,
            project=args[0],
            aa_residence=[
                {
                    "residue_name": "ASP",
                    "center_count": 1,
                    "effective_slow_factor": 0.001,
                    "frames_near": 2,
                    "fraction_near": 1.0,
                    "total_time_ns": 0.04,
                    "event_count": 1,
                    "mean_event_time_ns": 0.04,
                    "max_event_time_ns": 0.04,
                    "mean_min_distance_angstrom": 1.2,
                }
            ],
        )

    monkeypatch.setattr("webui.backend.app.main.api.simulate_project", fake_simulate)

    response = client.post(
        "/api/simulate",
        data={"project_json": json.dumps(project), "tac_start": "0.0", "tac_stop": "2.0"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["time_ns"] == [0.0, 1.0]
    assert body["donor_counts"] == [10.0, 5.0]
    assert body["quantum_yield_donor"] == 0.5
    assert body["aa_residence"][0]["residue_name"] == "ASP"


def test_get_dyes() -> None:
    client = TestClient(app)
    response = client.get("/api/dyes")
    assert response.status_code == 200
    dyes = response.json()
    assert "Alexa488" in dyes
    assert dyes["Alexa488"]["tau0"] == 4.2
    assert dyes["Alexa488"]["slow_radius"] == 8.5
    interactions = dyes["Alexa488"]["amino_acid_interactions"]
    # Residues without a redox-active side chain do not quench and inherit the
    # project-wide contact radius.
    assert interactions["ALA"] == {
        "slow_factor": 0.1,
        "kQ": 0.0,
        "quench_radius": None,
        "quench_atoms": ["CB"],
    }
    # Quenching is amino-acid specific: rate, contact radius and the atoms that
    # define the quenching centre all come from the residue's chemistry.
    assert interactions["TRP"]["slow_factor"] == 0.1
    assert interactions["TRP"]["kQ"] == 3.5
    assert interactions["TRP"]["quench_radius"] == 8.5
    assert "NE1" in interactions["TRP"]["quench_atoms"]
    assert "CB" not in interactions["TRP"]["quench_atoms"]
    assert interactions["MET"]["quench_atoms"] == ["SD"]
    assert interactions["HIS"]["kQ"] > 0.0


def test_scan_uses_adapter(monkeypatch, tmp_path: Path) -> None:
    client = TestClient(app)
    project = _minimal_project(tmp_path)
    
    sites = [
        {"chain": "E", "residue": 10, "residue_name": "ASP", "atom": "CB"}
    ]

    def fake_simulate(*args, **kwargs):
        proj = args[0]
        assert proj["attachment"]["chain"] == "E"
        assert proj["attachment"]["residue"] == 10
        assert proj["attachment"]["atom"] == "CB"
        
        result = DecaySimulationResult(
            time=np.array([0.0, 1.0]),
            donor_counts=np.array([10.0, 5.0]),
            fret_counts=None,
            quantum_yield_donor=0.5,
            quantum_yield_fret=None,
            collisions_fraction=0.1,
            fret_efficiency=None,
            project=proj,
        )
        result.lifetime_donor = 4.0
        return result

    monkeypatch.setattr("webui.backend.app.main.api.simulate_project", fake_simulate)

    response = client.post(
        "/api/scan",
        data={
            "project_json": json.dumps(project),
            "sites_json": json.dumps(sites),
            "tac_start": "0.0",
            "tac_stop": "2.0"
        },
    )
    assert response.status_code == 200
    
    results = []
    for line in response.iter_lines():
        if line.strip():
            chunk = json.loads(line)
            assert "current" in chunk
            assert "total" in chunk
            results.append(chunk["result"])

    assert len(results) == 1
    res = results[0]
    assert res["chain"] == "E"
    assert res["residue"] == 10
    assert res["status"] == "success"
    assert res["lifetime"] == 4.0
    assert res["quantum_yield"] == 0.5
    assert res["contact_fraction"] == 0.1


def test_download_zip_includes_mrc_outputs(monkeypatch, tmp_path: Path) -> None:
    client = TestClient(app)
    monkeypatch.setenv(quest_jobs.QUEST_JOBS_DIR_ENV, str(tmp_path / "jobs"))
    job_id = str(uuid.uuid4())
    job_dir = tmp_path / "jobs" / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "result.json").write_text("{}", encoding="utf8")
    (job_dir / "sampled_dye_density.mrc").write_bytes(b"mrc-data")
    (job_dir / "diffusion_coefficients.mrc").write_bytes(b"mrc-data")
    (job_dir / "quenching_rates.mrc").write_bytes(b"mrc-data")

    response = client.get(f"/api/jobs/{job_id}/download")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        assert "result.json" in archive.namelist()
        assert "sampled_dye_density.mrc" in archive.namelist()
        assert "diffusion_coefficients.mrc" in archive.namelist()
        assert "quenching_rates.mrc" in archive.namelist()


def test_list_jobs_uses_project_sidecar(monkeypatch, tmp_path: Path) -> None:
    client = TestClient(app)
    monkeypatch.setenv(quest_jobs.QUEST_JOBS_DIR_ENV, str(tmp_path / "jobs"))
    job_id = str(uuid.uuid4())
    job_dir = tmp_path / "jobs" / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "project.json").write_text(
        json.dumps({"attachment": {"chain": "A", "residue": 42}}),
        encoding="utf8",
    )
    (job_dir / "result.json").write_text("{not valid json", encoding="utf8")

    response = client.get("/api/jobs")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["jobs"][0]["job_id"] == job_id
    assert payload["jobs"][0]["labeling_site"] == "A:42"
    # The list is a summary: the full project stays behind /api/jobs/{id}.
    assert "project" not in payload["jobs"][0]


def test_list_jobs_does_not_parse_result_without_sidecar(monkeypatch, tmp_path: Path) -> None:
    client = TestClient(app)
    monkeypatch.setenv(quest_jobs.QUEST_JOBS_DIR_ENV, str(tmp_path / "jobs"))
    job_id = str(uuid.uuid4())
    job_dir = tmp_path / "jobs" / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "result.json").write_text("{not valid json", encoding="utf8")

    response = client.get("/api/jobs")

    assert response.status_code == 200
    payload = response.json()
    assert payload["jobs"][0]["job_id"] == job_id
    assert payload["jobs"][0]["labeling_site"] == "Unknown"


def test_list_jobs_is_paginated_newest_first(monkeypatch, tmp_path: Path) -> None:
    """The picker must not grow without bound as runs accumulate."""
    client = TestClient(app)
    monkeypatch.setenv(quest_jobs.QUEST_JOBS_DIR_ENV, str(tmp_path / "jobs"))

    job_ids = []
    for index in range(5):
        job_id = str(uuid.uuid4())
        job_dir = tmp_path / "jobs" / job_id
        job_dir.mkdir(parents=True)
        (job_dir / "project.json").write_text(
            json.dumps({"attachment": {"chain": "A", "residue": index}}), encoding="utf8"
        )
        os.utime(job_dir, (1_700_000_000 + index, 1_700_000_000 + index))
        job_ids.append(job_id)

    payload = client.get("/api/jobs", params={"limit": 2}).json()
    assert payload["total"] == 5
    assert payload["limit"] == 2
    assert [job["job_id"] for job in payload["jobs"]] == job_ids[::-1][:2]

    second_page = client.get("/api/jobs", params={"limit": 2, "offset": 2}).json()
    assert [job["job_id"] for job in second_page["jobs"]] == job_ids[::-1][2:4]

    # Out-of-range limits are clamped rather than rejected.
    assert client.get("/api/jobs", params={"limit": 0}).json()["limit"] == 1
    assert (
        client.get("/api/jobs", params={"limit": 10_000}).json()["limit"]
        == quest_jobs.MAX_JOB_LIST_LIMIT
    )


def test_list_jobs_ignores_non_uuid_directories(monkeypatch, tmp_path: Path) -> None:
    client = TestClient(app)
    monkeypatch.setenv(quest_jobs.QUEST_JOBS_DIR_ENV, str(tmp_path / "jobs"))
    (tmp_path / "jobs" / "not-a-job").mkdir(parents=True)
    (tmp_path / "jobs" / "stray.txt").write_text("x", encoding="utf8")

    payload = client.get("/api/jobs").json()
    assert payload["jobs"] == []
    assert payload["total"] == 0


class TestTopQuenchers:
    """The scan surfaces which residue types quench each candidate site.

    ``top_quenchers`` is domain code (``quest.core``); the backend re-exports it
    by importing it. These cases pin the ranking the scan table depends on.
    """

    def test_orders_by_share_and_drops_non_quenchers(self) -> None:
        rows = [
            {"residue_name": "ALA", "mean_quench_rate": 0.0, "quench_rate_fraction": 0.0},
            {"residue_name": "TYR", "mean_quench_rate": 0.4, "quench_rate_fraction": 0.25},
            {"residue_name": "TRP", "mean_quench_rate": 1.2, "quench_rate_fraction": 0.75},
        ]
        top = backend_main.top_quenchers(rows)
        assert [item["residue_name"] for item in top] == ["TRP", "TYR"]
        assert top[0]["quench_rate_fraction"] == 0.75

    def test_limit_is_respected(self) -> None:
        rows = [
            {"residue_name": name, "mean_quench_rate": rate, "quench_rate_fraction": rate}
            for name, rate in (("TRP", 0.5), ("TYR", 0.3), ("MET", 0.1), ("HIS", 0.05))
        ]
        assert len(backend_main.top_quenchers(rows, limit=2)) == 2

    def test_handles_missing_data(self) -> None:
        assert backend_main.top_quenchers(None) == []
        assert backend_main.top_quenchers([]) == []


def test_uploaded_structure_reports_its_own_name() -> None:
    """The UI shows what the user uploaded, not how the server stored it.

    Uploads are saved as "<uuid>_<name>" so two users cannot collide; that
    prefix used to reach the topbar, the structure card and the viewer footer.
    """
    client = TestClient(app)
    pdb = (Path(__file__).resolve().parents[3] / "tests" / "148l.pdb").read_bytes()

    response = client.post(
        "/api/structure/metadata",
        files={"pdb_file": ("148l.pdb", BytesIO(pdb), "chemical/x-pdb")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "148l.pdb"
    assert "_" not in payload["filename"].split(".")[0][:9]
    assert payload["atom_count"] == 1525
