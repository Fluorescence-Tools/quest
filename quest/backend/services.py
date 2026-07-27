"""QuEst's RPC surface: the operations every front end shares.

Each handler takes plain data, calls the domain, and returns the envelope
described in :mod:`quest.backend.dispatcher`. Nothing here computes anything —
if a handler grows logic, that logic belongs in the domain where the CLI and
notebooks can reach it too.

Registering with a host application's dispatcher is the whole point of the
split: :func:`register_services` only needs an object with
``register(name, handler)``, so ChiSurf (or any other host) can expose QuEst
without QuEst importing the host.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from .. import jobs, settings
from .dispatcher import InProcessClient, ServiceDispatcher, fail, ok

__all__ = ["register_services", "local_client", "METHODS", "result_payload"]

#: Method names in registration order, with a one-line summary each. This is
#: the table a manifest declares and a host advertises.
METHODS: dict[str, str] = {
    "quest.template": "Return a minimal valid project, optionally with FRET.",
    "quest.validate": "Check a project and report why it is malformed.",
    "quest.simulate": "Run one simulation and return the decay and derived metrics.",
    "quest.scan": "Simulate a set of labeling sites and return one record each.",
    "quest.parameter_catalog": "Return the parameter catalog (labels, units, help), optionally translated.",
    "quest.locales": "List the languages QuEst's labels and help text are available in.",
    "quest.dyes.list": "Return the dye presets.",
    "quest.dyes.save": "Add or replace a dye preset.",
    "quest.quenching_defaults": "Return the residue-type PET chemistry defaults.",
    "quest.structure.metadata": "Return the chains, residues and atoms of a structure file.",
    "quest.jobs.list": "Return job summaries, newest first.",
    "quest.jobs.get": "Return a finished run's result.",
}


def result_payload(result: Any) -> dict[str, Any]:
    """Convert a :class:`~quest.core.DecaySimulationResult` to plain data.

    Arrays become lists and absent channels stay ``None``. This is the boundary
    shape: whatever crosses to a browser, a host application or a file goes
    through here, so there is one definition of what a simulation result *is*.
    """

    def _list(value: Any) -> Any:
        return None if value is None else value.tolist()

    return {
        "job_id": result.job_id,
        "time_ns": result.time.tolist(),
        "donor_counts": result.donor_counts.tolist(),
        "fret_counts": _list(result.fret_counts),
        "quantum_yield_donor": result.quantum_yield_donor,
        "lifetime_donor": getattr(result, "lifetime_donor", 0.0),
        "quantum_yield_fret": result.quantum_yield_fret,
        "collisions_fraction": result.collisions_fraction,
        "fret_efficiency": result.fret_efficiency,
        "project": result.project,
        "trajectory_distance_to_mean": _list(result.trajectory_distance_to_mean),
        "trajectory_time_ns": _list(result.trajectory_time_ns),
        "trajectory_autocorr": _list(result.trajectory_autocorr),
        "trajectory_autocorr_lag_ns": _list(result.trajectory_autocorr_lag_ns),
        "trajectory_points": _list(result.trajectory_points),
        "av_points": _list(result.av_points),
        "acv_points": _list(result.acv_points),
        "av_volume": result.av_volume,
        "contact_volume": result.contact_volume,
        "contact_to_free_ratio": result.contact_to_free_ratio,
        "aa_residence": getattr(result, "aa_residence", []),
    }


def template_handler(with_fret: bool = True) -> dict[str, Any]:
    from ..project import template_project

    return ok({"project": template_project(with_fret=bool(with_fret))})


def validate_handler(
    project: Mapping[str, Any],
    pdb_path: Optional[str] = None,
) -> dict[str, Any]:
    from ..project import ProjectValidationError, validate_project

    try:
        validate_project(project, pdb_path=pdb_path)
    except ProjectValidationError as exc:
        return fail(str(exc))
    return ok({"valid": True, "project": dict(project)})


def simulate_handler(
    project: Mapping[str, Any],
    pdb_path: Optional[str] = None,
    nbins: Optional[int] = None,
    tac_range: Any = (0.0, 50.0),
    project_dir: Optional[str] = None,
    save_outputs: bool = True,
) -> dict[str, Any]:
    from .. import api
    from ..project import ProjectValidationError, validate_project

    try:
        validate_project(project, pdb_path=pdb_path)
    except ProjectValidationError as exc:
        return fail(str(exc))

    start, stop = tac_range
    try:
        result = api.simulate_project(
            project,
            pdb_path=pdb_path,
            nbins=nbins,
            tac_range=(float(start), float(stop)),
            project_dir=Path(project_dir) if project_dir else None,
            save_outputs=bool(save_outputs),
        )
    except Exception as exc:
        return fail(f"Simulation failed: {exc}")
    return ok(result_payload(result))


def scan_handler(
    project: Mapping[str, Any],
    positions: Mapping[str, Mapping[str, Any]],
    pdb_path: Optional[str] = None,
    nbins: Optional[int] = None,
    tac_range: Any = (0.0, 50.0),
    project_dir: Optional[str] = None,
) -> dict[str, Any]:
    """Simulate every position and return the records.

    Blocking and sequential: a caller that needs progress or cancellation drives
    :func:`quest.scan.scan_positions` itself. Making long-running work uniform
    across the surfaces is tracked as ``LAY-07``.
    """

    from ..scan import scan_positions

    start, stop = tac_range
    records = list(
        scan_positions(
            project,
            positions,
            pdb_path=pdb_path,
            nbins=nbins,
            tac_range=(float(start), float(stop)),
            project_dir=Path(project_dir) if project_dir else None,
        )
    )
    return ok({"results": {record["name"]: record for record in records}})


def locales_handler(locale: str | None = None) -> dict[str, Any]:
    """Which languages exist, which one was resolved, and its UI strings.

    One call, because a client that has the list still needs the strings, and a
    second round trip to get them is a round trip for nothing.
    """
    from .. import i18n

    resolved = i18n.resolve_locale(locale) if locale else i18n.DEFAULT_LOCALE
    return ok(
        {
            "available": i18n.available_locales(),
            "locale": resolved,
            "default": i18n.DEFAULT_LOCALE,
            "strings": i18n.ui_strings(resolved),
        }
    )


def parameter_catalog_handler(locale: str | None = None) -> dict[str, Any]:
    """The catalog, translated when a known locale is asked for.

    Callers that pass nothing get English rather than the server's system
    language: a shared backend must not answer differently depending on how the
    machine it runs on is configured.
    """
    from .. import i18n

    try:
        return ok(i18n.translate_catalog(locale or i18n.DEFAULT_LOCALE))
    except FileNotFoundError as exc:
        return fail(str(exc))


def dyes_list_handler() -> dict[str, Any]:
    try:
        return ok(settings.dye_repository())
    except FileNotFoundError as exc:
        return fail(str(exc))


def dyes_save_handler(dye: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return ok(settings.save_dye(dye))
    except (ValueError, OSError) as exc:
        return fail(str(exc))


def quenching_defaults_handler() -> dict[str, Any]:
    return ok(settings.quenching_defaults())


def structure_metadata_handler(pdb_path: str) -> dict[str, Any]:
    from ..structure_fetch import resolve_structure_path
    from ..structure_info import structure_metadata

    try:
        resolved = resolve_structure_path(pdb_path)
    except Exception as exc:
        return fail(str(exc))
    try:
        metadata = structure_metadata(resolved)
    except Exception as exc:
        return fail(f"Failed to parse structure: {exc}")
    if metadata["atom_count"] == 0:
        return fail("No ATOM/HETATM records found in the structure.")
    return ok(metadata)


def jobs_list_handler(
    project_dir: Optional[str] = None,
    limit: int = jobs.DEFAULT_JOB_LIST_LIMIT,
    offset: int = 0,
) -> dict[str, Any]:
    return ok(jobs.list_jobs(project_dir, limit=limit, offset=offset))


def jobs_get_handler(job_id: str, project_dir: Optional[str] = None) -> dict[str, Any]:
    try:
        return ok(jobs.read_job(job_id, project_dir))
    except (ValueError, FileNotFoundError) as exc:
        return fail(str(exc))


def register_services(dispatcher: Any) -> None:
    """Register QuEst's methods with *dispatcher*.

    Parameters
    ----------
    dispatcher : object
        Anything exposing ``register(name, handler)`` — QuEst's own
        :class:`~quest.backend.dispatcher.ServiceDispatcher`, or a host
        application's.
    """

    dispatcher.register("quest.template", lambda p: template_handler(**(p or {})))
    dispatcher.register("quest.validate", lambda p: validate_handler(**(p or {})))
    dispatcher.register("quest.simulate", lambda p: simulate_handler(**(p or {})))
    dispatcher.register("quest.scan", lambda p: scan_handler(**(p or {})))
    dispatcher.register(
        "quest.parameter_catalog",
        lambda p: parameter_catalog_handler((p or {}).get("locale")),
    )
    dispatcher.register(
        "quest.locales", lambda p: locales_handler((p or {}).get("locale"))
    )
    dispatcher.register("quest.dyes.list", lambda p: dyes_list_handler())
    dispatcher.register("quest.dyes.save", lambda p: dyes_save_handler(**(p or {})))
    dispatcher.register(
        "quest.quenching_defaults", lambda p: quenching_defaults_handler()
    )
    dispatcher.register(
        "quest.structure.metadata", lambda p: structure_metadata_handler(**(p or {}))
    )
    dispatcher.register("quest.jobs.list", lambda p: jobs_list_handler(**(p or {})))
    dispatcher.register("quest.jobs.get", lambda p: jobs_get_handler(**(p or {})))


def local_client() -> InProcessClient:
    """Return a client wired to a fresh dispatcher with QuEst's methods on it."""

    dispatcher = ServiceDispatcher()
    register_services(dispatcher)
    return InProcessClient(dispatcher)
