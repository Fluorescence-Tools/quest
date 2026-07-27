"""Job artefacts and structure metadata, as domain operations.

Both used to live inside the web backend, reachable only from a browser. A
notebook, the CLI and a host application have the same need, so they are tested
here against the domain rather than through HTTP.
"""

from __future__ import annotations

import json
import uuid
import zipfile
from pathlib import Path

import pytest

from quest import jobs
from quest.structure_info import structure_metadata

STRUCTURE = Path(__file__).parent / "148l.pdb"


def _make_job(root: Path, *, chain: str = "E", residue: int = 55, result: bool = True) -> str:
    job_id = str(uuid.uuid4())
    job_dir = root / "jobs" / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "project.json").write_text(
        json.dumps({"attachment": {"chain": chain, "residue": residue, "atom": "CB"}}),
        encoding="utf8",
    )
    if result:
        (job_dir / "result.json").write_text(json.dumps({"quantum_yield_donor": 0.5}), encoding="utf8")
        (job_dir / "decay.csv").write_text("t,counts\n0,1\n", encoding="utf8")
    return job_id


class TestStructureMetadata:
    def test_reports_chains_residues_and_atoms(self) -> None:
        metadata = structure_metadata(STRUCTURE)
        assert metadata["atom_count"] == 1525
        chains = {chain["id"]: chain for chain in metadata["chains"]}
        assert set(chains) == {"A", "E", "S"}
        # T4 lysozyme is the long chain; the ligand and solvent chains are short.
        assert len(chains["E"]["residues"]) == 291

        residue = next(r for r in chains["E"]["residues"] if r["id"] == 55)
        assert "CB" in residue["atoms"]
        assert residue["atoms"] == sorted(residue["atoms"])

    def test_residues_come_back_in_sequence_order(self) -> None:
        residues = structure_metadata(STRUCTURE)["chains"][1]["residues"]
        ids = [residue["id"] for residue in residues]
        assert ids == sorted(ids)

    def test_the_raw_text_is_included_for_a_viewer(self) -> None:
        metadata = structure_metadata(STRUCTURE)
        assert metadata["pdb_text"].startswith("HEADER") or "ATOM" in metadata["pdb_text"]
        assert metadata["filename"] == "148l.pdb"

    def test_a_missing_file_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            structure_metadata(tmp_path / "nope.pdb")


class TestJobListing:
    def test_newest_first_with_paging(self, tmp_path: Path) -> None:
        import os
        import time

        ids = []
        for index in range(4):
            job_id = _make_job(tmp_path, residue=50 + index)
            # Distinct mtimes, so "newest first" is well defined.
            os.utime(tmp_path / "jobs" / job_id, (time.time() + index, time.time() + index))
            ids.append(job_id)

        page = jobs.list_jobs(tmp_path, limit=2)
        assert page["total"] == 4
        assert [job["job_id"] for job in page["jobs"]] == ids[::-1][:2]

        second = jobs.list_jobs(tmp_path, limit=2, offset=2)
        assert [job["job_id"] for job in second["jobs"]] == ids[::-1][2:]

    def test_paging_is_clamped_not_rejected(self, tmp_path: Path) -> None:
        assert jobs.list_jobs(tmp_path, limit=0)["limit"] == 1
        assert jobs.list_jobs(tmp_path, limit=10_000)["limit"] == jobs.MAX_JOB_LIST_LIMIT
        assert jobs.list_jobs(tmp_path, offset=-5)["offset"] == 0

    def test_only_uuid_directories_count_as_runs(self, tmp_path: Path) -> None:
        _make_job(tmp_path)
        (tmp_path / "jobs" / "not-a-job").mkdir()
        (tmp_path / "jobs" / "stray.txt").write_text("x", encoding="utf8")

        assert jobs.list_jobs(tmp_path)["total"] == 1

    def test_a_missing_jobs_directory_is_empty_not_an_error(self, tmp_path: Path) -> None:
        assert jobs.list_jobs(tmp_path) == {"jobs": [], "total": 0, "limit": 50, "offset": 0}

    def test_the_summary_names_the_labeling_site(self, tmp_path: Path) -> None:
        _make_job(tmp_path, chain="E", residue=132)
        summary = jobs.list_jobs(tmp_path)["jobs"][0]
        assert summary["labeling_site"] == "E:132"
        assert summary["date"]
        # Summaries only — the project and result stay on disk.
        assert set(summary) == {"job_id", "labeling_site", "date"}

    def test_an_unfinished_run_says_so(self, tmp_path: Path) -> None:
        job_id = str(uuid.uuid4())
        (tmp_path / "jobs" / job_id).mkdir(parents=True)
        summary = jobs.list_jobs(tmp_path)["jobs"][0]
        assert summary["labeling_site"] == "No result yet"


class TestReadingAndArchiving:
    def test_reads_the_result(self, tmp_path: Path) -> None:
        job_id = _make_job(tmp_path)
        assert jobs.read_job(job_id, tmp_path)["quantum_yield_donor"] == 0.5

    def test_a_non_uuid_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            jobs.read_job("../etc/passwd", tmp_path)

    def test_a_missing_result_is_not_found(self, tmp_path: Path) -> None:
        job_id = _make_job(tmp_path, result=False)
        with pytest.raises(FileNotFoundError):
            jobs.read_job(job_id, tmp_path)

    def test_archiving_collects_the_artefacts(self, tmp_path: Path) -> None:
        job_id = _make_job(tmp_path)
        archive = jobs.archive_job(job_id, tmp_path)

        with zipfile.ZipFile(archive) as zipped:
            names = set(zipped.namelist())
        assert {"project.json", "result.json", "decay.csv"} <= names
        # The archive never contains itself, however many times it is built.
        assert archive.name not in names
        assert jobs.archive_job(job_id, tmp_path) == archive
        with zipfile.ZipFile(archive) as zipped:
            assert archive.name not in set(zipped.namelist())
