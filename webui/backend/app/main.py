from __future__ import annotations

import sys

class SafeStream:
    def __init__(self, original_stream):
        self.original_stream = original_stream

    def write(self, data):
        if not self.original_stream:
            return 0
        try:
            return self.original_stream.write(data)
        except OSError as e:
            if e.errno == 5:  # EIO
                return len(data)
            raise

    def flush(self):
        if not self.original_stream:
            return
        try:
            self.original_stream.flush()
        except OSError as e:
            if e.errno == 5:
                return
            raise

    def isatty(self):
        if not self.original_stream:
            return False
        try:
            return self.original_stream.isatty()
        except Exception:
            return False

    def __getattr__(self, name):
        if not self.original_stream:
            raise AttributeError(name)
        return getattr(self.original_stream, name)

sys.stdout = SafeStream(sys.stdout)
sys.stderr = SafeStream(sys.stderr)

import datetime
import json
import os
import tempfile
import uuid
import zipfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import asyncio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from quest import api
from quest.core import top_quenchers
from quest.scan import simulate_site
from quest import i18n as quest_i18n
from quest import settings as quest_settings
from quest.backend import result_payload
from quest.backend.services import locales_handler, structure_metadata_handler
from quest import jobs as quest_jobs
from quest.structure_fetch import resolve_structure_path


def _data_dir() -> Path:
    target = Path(os.getenv("QUEST_WEBUI_DATA_DIR", Path(tempfile.gettempdir()) / "quest-webui"))
    target.mkdir(parents=True, exist_ok=True)
    return target


# Sites simulated concurrently by /api/scan. Kept low on purpose: LabelLib's
# accessible-volume kernel holds the GIL, so extra workers add contention
# rather than throughput. The real reason to use a pool at all is that it keeps
# the simulation off the event loop, so the server stays responsive (and
# client disconnects are still noticed) while a long scan runs.
SCAN_WORKERS = max(1, min(4, (os.cpu_count() or 1) - 1))

DATA_DIR = _data_dir()
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Run artefacts do **not** live under the repository root. They go to the
# per-user artefact root that `quest.jobs` owns, which is also what the job
# routes read -- so a run list no longer depends on where uvicorn was started.
# The settings files are package data, resolved by `quest.settings` itself.

app = FastAPI(title="QuEst Web API")

allowed_origins = os.getenv(
    "QUEST_WEBUI_CORS_ORIGINS",
    "http://127.0.0.1:3000,http://localhost:3000,http://127.0.0.1:3001,http://localhost:3001",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in allowed_origins.split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _parse_project_json(project_json: str) -> dict[str, Any]:
    try:
        parsed = json.loads(project_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="Project JSON must be an object.")
    return parsed


async def _store_upload(file: UploadFile) -> Path:
    filename = Path(file.filename or "uploaded_file").name
    target = UPLOAD_DIR / f"{uuid.uuid4()}_{filename}"
    content = await file.read()
    target.write_bytes(content)
    return target


def _resolve_pdb_path(project: dict[str, Any], pdb_path: str | None, uploaded: Path | None) -> Path | None:
    if uploaded is not None:
        return uploaded
    if pdb_path:
        try:
            return resolve_structure_path(pdb_path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    configured = project.get("pdb")
    if configured:
        try:
            return resolve_structure_path(configured)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "app": "quest", "version": "19.8.13"}


@app.get("/api/parameter-catalog")
def get_parameter_catalog(locale: str | None = None) -> dict[str, Any]:
    """Parameter labels and help, in the requested language.

    The translations live in `quest/settings/locales/`, beside the catalog they
    translate — so the browser, the desktop form and `doc/parameters.md` all say
    the same thing in the same language.
    """
    try:
        return quest_i18n.translate_catalog(locale or quest_i18n.DEFAULT_LOCALE)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/i18n")
def get_i18n(locale: str | None = None) -> dict[str, Any]:
    """The available languages, and the UI strings for the requested one."""
    response = locales_handler(locale)
    if not response.get("ok"):
        raise HTTPException(status_code=500, detail=response.get("error"))
    return response["result"]


@app.get("/api/dyes")
def get_dyes() -> dict[str, Any]:
    try:
        return quest_settings.dye_repository()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/quenching-defaults")
def get_quenching_defaults() -> dict[str, Any]:
    """Residue-type chemistry used to make quenching amino-acid specific."""
    return quest_settings.quenching_defaults()


@app.get("/api/template")
def template(with_fret: bool = True) -> dict[str, Any]:
    return {"project": api.template_project(with_fret=with_fret)}


@app.post("/api/validate")
async def validate(
    project_json: str = Form(...),
    pdb_path: str | None = Form(default=None),
    pdb_file: UploadFile | None = File(default=None),
) -> dict[str, Any]:
    project = _parse_project_json(project_json)
    uploaded = await _store_upload(pdb_file) if pdb_file is not None else None
    resolved_pdb = _resolve_pdb_path(project, pdb_path, uploaded)
    if resolved_pdb is not None:
        project["pdb"] = str(resolved_pdb)
    try:
        api.validate_project(project)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"valid": True, "project": project}


@app.post("/api/structure/metadata")
async def structure_metadata_route(
    pdb_path: str | None = Form(default=None),
    pdb_file: UploadFile | None = File(default=None),
) -> dict[str, Any]:
    uploaded = await _store_upload(pdb_file) if pdb_file is not None else None
    resolved_pdb = _resolve_pdb_path({}, pdb_path, uploaded)
    if resolved_pdb is None:
        raise HTTPException(status_code=400, detail="Provide a PDB file upload or local pdb_path.")
    if resolved_pdb.suffix.lower() not in {".pdb", ".ent", ".txt", ".cif", ""}:
        raise HTTPException(status_code=400, detail=f"Unsupported structure file type: {resolved_pdb.suffix}")
    response = structure_metadata_handler(str(resolved_pdb))
    if not response["ok"]:
        raise HTTPException(status_code=400, detail=response["error"])

    metadata = response["result"]
    if pdb_file is not None and pdb_file.filename:
        # Uploads are stored as "<uuid>_<name>" so two users cannot collide.
        # That is a storage detail: the UI should show what was uploaded, not
        # the bookkeeping around it.
        metadata["filename"] = Path(pdb_file.filename).name
    return metadata


@app.post("/api/simulate")
async def simulate(
    project_json: str = Form(...),
    pdb_path: str | None = Form(default=None),
    pdb_file: UploadFile | None = File(default=None),
    nbins: int | None = Form(default=None),
    tac_start: float = Form(default=0.0),
    tac_stop: float = Form(default=50.0),
) -> dict[str, Any]:
    project = _parse_project_json(project_json)
    uploaded = await _store_upload(pdb_file) if pdb_file is not None else None
    resolved_pdb = _resolve_pdb_path(project, pdb_path, uploaded)
    if resolved_pdb is not None:
        project["pdb"] = str(resolved_pdb)
    try:
        api.validate_project(project)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        result = api.simulate_project(
            project,
            pdb_path=resolved_pdb,
            nbins=nbins,
            tac_range=(tac_start, tac_stop),
            # No `project_dir`: runs go to the per-user artefact root rather
            # than the checkout. Passing REPO_ROOT here is what put 39 GB of
            # `jobs/` into the working tree, and it also made the run list
            # depend on where uvicorn was started.
            save_outputs=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {exc}") from exc

    # One definition of what a simulation result looks like on the wire, shared
    # with the RPC surface (quest.simulate).
    return result_payload(result)


@app.post("/api/scan")
async def scan(
    request: Request,
    project_json: str = Form(...),
    pdb_path: str | None = Form(default=None),
    pdb_file: UploadFile | None = File(default=None),
    nbins: int | None = Form(default=None),
    tac_start: float = Form(default=0.0),
    tac_stop: float = Form(default=50.0),
    sites_json: str = Form(...),
) -> StreamingResponse:
    project = _parse_project_json(project_json)
    uploaded = await _store_upload(pdb_file) if pdb_file is not None else None
    resolved_pdb = _resolve_pdb_path(project, pdb_path, uploaded)
    if resolved_pdb is not None:
        project["pdb"] = str(resolved_pdb)

    try:
        sites_parsed = json.loads(sites_json)
        fps_data = api.normalize_to_fps_positions(sites_parsed, project)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid sites JSON: {exc}") from exc

    scan_temp_dir = Path(tempfile.gettempdir())
    positions = fps_data.get("Positions", {})
    total_positions = len(positions)

    def simulate_one(name: str, pos: dict[str, Any]) -> dict[str, Any]:
        """Simulate one labelling site. Runs in a worker thread; never raises."""
        return simulate_site(
            project,
            pos,
            name=name,
            pdb_path=resolved_pdb,
            nbins=nbins,
            tac_range=(tac_start, tac_stop),
            project_dir=scan_temp_dir,
        )

    async def generate_scan():
        loop = asyncio.get_running_loop()
        items = list(positions.items())
        executor = ThreadPoolExecutor(
            max_workers=SCAN_WORKERS, thread_name_prefix="quest-scan"
        )
        # Keep at most SCAN_WORKERS sites in flight and pop them in submission
        # order, so the stream still reports sites in the order the client
        # listed them while the pool stays saturated.
        inflight: deque = deque()
        submitted = 0
        try:
            while submitted < len(items) and len(inflight) < SCAN_WORKERS:
                inflight.append(loop.run_in_executor(executor, simulate_one, *items[submitted]))
                submitted += 1

            for idx in range(len(items)):
                if await request.is_disconnected():
                    break
                res = await inflight.popleft()
                if submitted < len(items):
                    inflight.append(
                        loop.run_in_executor(executor, simulate_one, *items[submitted])
                    )
                    submitted += 1
                yield json.dumps(
                    {"current": idx + 1, "total": total_positions, "result": res}
                ) + "\n"
        finally:
            for future in inflight:
                future.cancel()
            executor.shutdown(wait=False)

    return StreamingResponse(generate_scan(), media_type="application/x-ndjson")


@app.get("/api/jobs")
async def list_jobs(
    limit: int = quest_jobs.DEFAULT_JOB_LIST_LIMIT, offset: int = 0
) -> dict[str, Any]:
    """Return the most recent job summaries, newest first."""
    return quest_jobs.list_jobs(None, limit=limit, offset=offset)


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    try:
        return quest_jobs.read_job(job_id, None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read job result: {exc}") from exc


@app.post("/api/dyes")
async def save_dye(dye: dict[str, Any]):
    """Add or replace a dye preset and return the library."""
    try:
        quest_settings.save_dye(dye)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save dye catalogue: {exc}") from exc
    return quest_settings.dye_repository()


@app.get("/api/jobs/{job_id}/download")
async def download_job_zip(job_id: str):
    try:
        zip_path = quest_jobs.archive_job(job_id, None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to create ZIP archive: {exc}") from exc

    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=zip_path.name,
    )
