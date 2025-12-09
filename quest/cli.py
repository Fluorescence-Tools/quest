from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import json

import click

from . import api


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="quest", prog_name="quest")
def cli() -> None:
    """Command line interface for QuEst (Quenching Estimator)."""


def _parse_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered == "null" or lowered == "none":
        return None
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _set_path(target: MutableMapping[str, Any], path: str, value: Any) -> None:
    """Set dotted path inside a mapping, creating nested dicts as needed."""
    parts = path.split(".")
    cursor: MutableMapping[str, Any] = target
    for key in parts[:-1]:
        if key not in cursor or not isinstance(cursor[key], MutableMapping):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[parts[-1]] = value


def _clone_with_overrides(project: MutableMapping[str, Any], setters: Sequence[tuple[str, Any]]) -> MutableMapping[str, Any]:
    from copy import deepcopy

    clone = deepcopy(project)
    for key, val in setters:
        _set_path(clone, key, val)
    return clone


def _expand_grid(setters: Sequence[tuple[str, Any]]) -> Sequence[tuple[str, Any]]:
    """Expand comma-separated values into multiple setter entries."""
    expanded: list[tuple[str, Any]] = []
    for key, raw in setters:
        if isinstance(raw, str) and "," in raw:
            expanded.append((key, [_parse_value(v) for v in raw.split(",")]))
        else:
            expanded.append((key, raw))
    return expanded


def _cartesian(assignments: Sequence[tuple[str, Sequence[Any]]]) -> list[list[tuple[str, Any]]]:
    if not assignments:
        return [[]]
    head_key, head_vals = assignments[0]
    rest = assignments[1:]
    tails = _cartesian(rest)
    combos: list[list[tuple[str, Any]]] = []
    for hv in head_vals:
        for t in tails:
            combos.append([(head_key, hv)] + t)
    return combos


def _build_runs(base_project: MutableMapping[str, Any], setters: Sequence[tuple[str, Any]]) -> list[tuple[MutableMapping[str, Any], dict[str, Any]]]:
    """Create per-run projects for grid sweeping."""
    # Split single vs multi-valued
    multi: list[tuple[str, list[Any]]] = []
    singles: list[tuple[str, Any]] = []
    for key, val in setters:
        if isinstance(val, list):
            multi.append((key, val))
        else:
            singles.append((key, val))

    if not multi:
        proj = _clone_with_overrides(base_project, singles)
        return [(proj, {k: v for k, v in singles})]

    combos = _cartesian(multi)
    runs: list[tuple[MutableMapping[str, Any], dict[str, Any]]] = []
    for combo in combos:
        effective = singles + combo
        proj = _clone_with_overrides(base_project, effective)
        tag = {k: v for k, v in combo}
        tag.update({k: v for k, v in singles})
        runs.append((proj, tag))
    return runs


def _sanitize_tag_value(val: Any) -> str:
    txt = str(val)
    return txt.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _resolve_output_path(
    *,
    output_path: Path | None,
    project_path: Path,
    idx: int,
    tags: Mapping[str, Any],
    multiple: bool,
) -> Path:
    base = output_path
    if base is None:
        base = project_path.with_suffix(".csv")
    stem = base.stem
    suffix = base.suffix or ".csv"

    if multiple:
        tag_parts = [f"{k}-{_sanitize_tag_value(v)}" for k, v in sorted(tags.items())]
        tag_block = "_".join(tag_parts) if tag_parts else f"run{idx}"
        stem = f"{stem}_{tag_block}"
    return base.with_name(stem + suffix)


@cli.command()
@click.option(
    "-p",
    "--project",
    "project_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a QuEst project JSON file.",
)
@click.option(
    "--pdb",
    "pdb_path",
    required=False,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Override PDB path instead of project['pdb'].",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    required=False,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output path. Uses extension to choose format (csv|json).",
)
@click.option(
    "--nbins",
    type=int,
    default=None,
    help="Number of histogram bins. Defaults to project value or 4096.",
)
@click.option(
    "--tac-start",
    type=float,
    default=0.0,
    show_default=True,
    help="Start of TAC range in ns.",
)
@click.option(
    "--tac-stop",
    type=float,
    default=50.0,
    show_default=True,
    help="End of TAC range in ns.",
)
@click.option(
    "--set",
    "set_param",
    multiple=True,
    help="Override project value, dotted path (e.g., fret.enabled=false or attachment.residue=42).",
)
@click.option(
    "--grid",
    "grid_param",
    multiple=True,
    help="Sweep multiple values, dotted path with comma list (e.g., attachment.residue=42,43,44).",
)
def simulate(
    project_path: Path,
    pdb_path: Path | None,
    output_path: Path | None,
    nbins: int | None,
    tac_start: float,
    tac_stop: float,
    set_param: tuple[str, ...] = (),
    grid_param: tuple[str, ...] = (),
) -> None:
    """Run a QuEst project headlessly and write decay results."""

    try:
        base_project = api.load_project(project_path)
    except Exception as exc:  # pragma: no cover - defensive
        raise click.ClickException(f"Failed to load project: {exc}")

    setters: list[tuple[str, Any]] = []
    for raw in set_param:
        if "=" not in raw:
            raise click.ClickException(f"--set expects key=value, got: {raw}")
        key, value = raw.split("=", 1)
        setters.append((key, _parse_value(value)))
    for raw in grid_param:
        if "=" not in raw:
            raise click.ClickException(f"--grid expects key=val1,val2, got: {raw}")
        key, value = raw.split("=", 1)
        setters.append((key, [_parse_value(v) for v in value.split(",")]))

    runs = _build_runs(base_project, setters)
    results_written = []

    for idx, (project, tags) in enumerate(runs):
        _validate_project(project, project_path)

        try:
            result = api.simulate_project(
                project,
                pdb_path=pdb_path,
                nbins=nbins,
                tac_range=(tac_start, tac_stop),
            )
        except Exception as exc:  # pragma: no cover - defensive
            raise click.ClickException(f"Simulation failed: {exc}")

        out_path = _resolve_output_path(
            output_path=output_path,
            project_path=project_path,
            idx=idx,
            tags=tags,
            multiple=len(runs) > 1,
        )

        suffix = out_path.suffix.lower()
        if suffix == ".csv":
            _write_csv(out_path, result)
        elif suffix == ".json":
            _write_json(out_path, result)
        else:
            raise click.ClickException("Output must end with .csv or .json")

        results_written.append(str(out_path))

    if len(results_written) == 1:
        click.echo(f"Wrote results to {results_written[0]}")
    else:
        click.echo("Wrote results:\n" + "\n".join(f"  - {p}" for p in results_written))


@cli.command()
@click.option(
    "-o",
    "--output",
    "output_path",
    required=False,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write template to file (default: stdout).",
)
@click.option(
    "--with-fret/--no-fret",
    default=True,
    show_default=True,
    help="Include FRET configuration.",
)
def template(output_path: Path | None, with_fret: bool) -> None:
    """Emit a minimal QuEst project template."""

    project = _template_project(with_fret=with_fret)
    payload = json.dumps(project, indent=2)

    if output_path:
        output_path.write_text(payload, encoding="utf8")
        click.echo(f"Wrote template to {output_path}")
    else:
        click.echo(payload)


@cli.command()
@click.option(
    "-p",
    "--project",
    "project_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a QuEst project JSON file.",
)
def validate(project_path: Path) -> None:
    """Validate a project JSON file and report issues."""

    try:
        project = api.load_project(project_path)
    except Exception as exc:  # pragma: no cover - defensive
        raise click.ClickException(f"Failed to load project: {exc}")

    _validate_project(project, project_path)
    click.echo(f"{project_path} is valid.")


@cli.command()
def gui() -> None:
    """Launch the QuEst GUI."""

    try:
        from .quest_gui import start_gui
    except Exception as exc:  # pragma: no cover - defensive
        raise click.ClickException(f"Failed to import GUI: {exc}")

    start_gui()


def _validate_project(project: Mapping[str, Any], project_path: Path | None = None) -> None:
    """Raise ClickException if the project is malformed."""

    required_top = [
        "tau0",
        "kQ",
        "n_photons",
        "t_max",
        "t_step",
        "dg",
        "slow_fact",
        "critical_distance",
        "av_parameter",
        "quencher",
    ]
    missing = [key for key in required_top if key not in project]
    if missing:
        raise click.ClickException(f"Project missing required keys: {', '.join(missing)}")

    pdb = project.get("pdb")
    if pdb is None and project_path is None:
        raise click.ClickException("Project must define 'pdb' path when no override is provided.")
    if pdb is not None:
        pdb_path = Path(pdb)
        if not pdb_path.exists():
            raise click.ClickException(f"PDB path does not exist: {pdb_path}")

    fret_block = project.get("fret") or {}
    dyes: Sequence[Mapping[str, Any]] = fret_block.get("dyes") or []
    if not dyes:
        raise click.ClickException("Project must define at least one dye in fret.dyes with a diffusion coefficient 'D'.")
    donor_dye = dyes[0]
    if donor_dye.get("D") is None:
        raise click.ClickException("Donor dye entry must define diffusion coefficient 'D'.")

    # Basic TAC sanity
    t_start = project.get("t_start", 0.0)
    t_stop = project.get("t_max", None)
    if t_stop is None:
        raise click.ClickException("Project must define 't_max' (simulation duration).")
    if t_stop <= t_start:
        raise click.ClickException("Project 't_max' must be greater than 't_start'.")


def _write_csv(path: Path, result: api.DecaySimulationResult) -> None:
    import csv

    fieldnames = [
        "time_ns",
        "donor_counts",
        "fret_counts",
        "quantum_yield_donor",
        "quantum_yield_fret",
        "collisions_fraction",
        "fret_efficiency",
    ]
    with path.open("w", newline="", encoding="utf8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        fret_counts = result.fret_counts if result.fret_counts is not None else [None] * len(result.time)
        for t, d, f in zip(result.time, result.donor_counts, fret_counts):
            writer.writerow(
                {
                    "time_ns": float(t),
                    "donor_counts": float(d),
                    "fret_counts": None if f is None else float(f),
                    "quantum_yield_donor": result.quantum_yield_donor,
                    "quantum_yield_fret": result.quantum_yield_fret,
                    "collisions_fraction": result.collisions_fraction,
                    "fret_efficiency": result.fret_efficiency,
                }
            )


def _write_json(path: Path, result: api.DecaySimulationResult) -> None:
    payload = {
        "time_ns": result.time.tolist(),
        "donor_counts": result.donor_counts.tolist(),
        "fret_counts": None if result.fret_counts is None else result.fret_counts.tolist(),
        "quantum_yield_donor": result.quantum_yield_donor,
        "quantum_yield_fret": result.quantum_yield_fret,
        "collisions_fraction": result.collisions_fraction,
        "fret_efficiency": result.fret_efficiency,
        "project": result.project,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf8")


def _template_project(*, with_fret: bool = True) -> MutableMapping[str, Any]:
    """Create a minimal valid project dictionary."""

    project: MutableMapping[str, Any] = {
        "tau0": 4.2,
        "kQ": 0.0,
        "n_photons": 500000,
        "t_max": 16000.0,
        "t_step": 0.032,
        "dg": 0.0,
        "slow_fact": 0.1,
        "critical_distance": 8.5,
        "av_parameter": 1000,
        "quencher": "TRY",
        "pdb": "structure.pdb",
        "n_bins": 4096,
        "attachment": {"chain": "A", "residue": 1, "atom": "CB"},
        "save_avs": False,
        "slow_radius": 8.5,
        "all_quencher_atoms": True,
        "output_file": "out",
    }

    dyes = [
        {
            "name": "donor",
            "D": 7.5,
            "attachment": {"chain": "A", "residue": 1, "atom": "CB"},
            "av_parameter": 1000,
        }
    ]

    fret_block: MutableMapping[str, Any] = {"enabled": with_fret, "dyes": dyes}
    if with_fret:
        fret_block["dyes"].append(
            {
                "name": "acceptor",
                "D": 7.5,
                "attachment": {"chain": "A", "residue": 2, "atom": "CB"},
                "av_parameter": 1000,
            }
        )
        fret_block["R0_matrix"] = [[None, 5.5], [5.5, None]]

    project["fret"] = fret_block
    return project


if __name__ == "__main__":
    cli()
