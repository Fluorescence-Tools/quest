"""Reading back what a run wrote.

Every simulation writes a ``jobs/<uuid>/`` directory
(see okf/architecture/job-artefacts.md). Listing those runs, summarising them
for a picker, reading one back and packaging one for download are domain
operations -- a notebook or a host application has the same need as the browser
that first grew them.

The directory name is a UUID, which is also the validity check: anything else
in ``jobs/`` is not a run and is ignored rather than half-parsed.
"""

from __future__ import annotations

import datetime
import json
import uuid
import zipfile
from pathlib import Path
from typing import Any, Optional

__all__ = [
    "DEFAULT_JOB_LIST_LIMIT",
    "MAX_JOB_LIST_LIMIT",
    "jobs_directory",
    "is_job_id",
    "job_summary",
    "list_jobs",
    "read_job",
    "archive_job",
]

#: Page size for :func:`list_jobs`, and the ceiling a caller can ask for. The
#: picker shows a bounded window; without a cap the response grows without limit
#: as runs accumulate (a development tree here holds several hundred).
DEFAULT_JOB_LIST_LIMIT = 50
MAX_JOB_LIST_LIMIT = 500


def jobs_directory(project_dir: Optional[Path | str] = None) -> Path:
    """Return the ``jobs/`` directory under *project_dir* (default: cwd)."""

    base = Path(project_dir) if project_dir is not None else Path.cwd()
    return base / "jobs"


def is_job_id(value: Any) -> bool:
    """Whether *value* is a job identifier, i.e. a UUID."""

    try:
        uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return False
    return True


def job_summary(path: Path) -> dict[str, Any]:
    """Summarise one job directory for a run picker.

    Only the fields a picker shows. The project and the full result live in the
    directory and are read on demand by :func:`read_job`; embedding them here
    made the listing grow without bound.
    """

    date_str = datetime.datetime.fromtimestamp(path.stat().st_mtime).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    project_file = path / "project.json"

    if project_file.exists():
        try:
            with project_file.open("r", encoding="utf8") as handle:
                attachment = json.load(handle).get("attachment") or {}
            labeling_site = f"{attachment.get('chain', '')}:{attachment.get('residue', '')}"
        except Exception:
            labeling_site = "Error loading"
    elif (path / "result.json").exists():
        labeling_site = "Unknown"
    else:
        # A directory with neither is a run that was started and never finished.
        labeling_site = "No result yet"
        date_str = ""

    return {"job_id": path.name, "labeling_site": labeling_site, "date": date_str}


def list_jobs(
    project_dir: Optional[Path | str] = None,
    limit: int = DEFAULT_JOB_LIST_LIMIT,
    offset: int = 0,
) -> dict[str, Any]:
    """Return job summaries, newest first, as ``{jobs, total, limit, offset}``.

    Out-of-range paging is clamped rather than rejected: a caller asking for
    10 000 runs gets :data:`MAX_JOB_LIST_LIMIT`, not an error.
    """

    directory = jobs_directory(project_dir)
    limit = max(1, min(int(limit), MAX_JOB_LIST_LIMIT))
    offset = max(0, int(offset))

    if not directory.exists():
        return {"jobs": [], "total": 0, "limit": limit, "offset": offset}

    candidates: list[tuple[float, Path]] = []
    for path in directory.iterdir():
        if not path.is_dir() or not is_job_id(path.name):
            continue
        try:
            candidates.append((path.stat().st_mtime, path))
        except OSError:
            continue

    candidates.sort(key=lambda item: item[0], reverse=True)
    window = candidates[offset : offset + limit]
    return {
        "jobs": [job_summary(path) for _mtime, path in window],
        "total": len(candidates),
        "limit": limit,
        "offset": offset,
    }


def read_job(job_id: str, project_dir: Optional[Path | str] = None) -> dict[str, Any]:
    """Return a finished run's ``result.json``.

    Raises
    ------
    ValueError
        If *job_id* is not a UUID.
    FileNotFoundError
        If there is no such run, or it has no result yet.
    """

    if not is_job_id(job_id):
        raise ValueError(f"Invalid job id (must be a UUID): {job_id!r}")
    result_file = jobs_directory(project_dir) / str(job_id) / "result.json"
    if not result_file.exists():
        raise FileNotFoundError(f"No result for job {job_id}")
    with result_file.open("r", encoding="utf8") as handle:
        return json.load(handle)


def archive_job(job_id: str, project_dir: Optional[Path | str] = None) -> Path:
    """Zip a run's artefacts and return the archive path.

    The archive is written inside the job directory and excludes any previous
    archive, so repeated calls do not nest.

    Raises
    ------
    ValueError
        If *job_id* is not a UUID.
    FileNotFoundError
        If there is no such run.
    """

    if not is_job_id(job_id):
        raise ValueError(f"Invalid job id (must be a UUID): {job_id!r}")
    job_dir = jobs_directory(project_dir) / str(job_id)
    if not job_dir.is_dir():
        raise FileNotFoundError(f"No such job: {job_id}")

    archive_name = f"quest_simulation_{job_id}.zip"
    archive_path = job_dir / archive_name
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(job_dir.iterdir()):
            if file_path.is_file() and file_path.name != archive_name:
                archive.write(file_path, file_path.name)
    return archive_path
