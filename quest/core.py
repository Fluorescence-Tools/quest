"""Headless QuEst backend helpers.

This module provides a small, GUI-free service layer around the legacy
`DonorDecay` simulation so other code (CLI, notebooks, tests) can drive
the core quenching/FRET calculations without depending on any Qt widgets.

It intentionally mirrors the public surface of ``quest.api`` but keeps
the logic in one place so both the CLI and notebook users can share it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Tuple

import json
import numpy as np

from .lib.tools.dye_diffusion.dye_diffusion import DonorDecay
from .lib.tools.dye_diffusion.photon import simulate_decay_quench


@dataclass
class DecaySimulationResult:
    """Container for a single QuEst decay simulation."""

    time: np.ndarray
    donor_counts: np.ndarray
    fret_counts: Optional[np.ndarray]
    quantum_yield_donor: float
    quantum_yield_fret: Optional[float]
    collisions_fraction: float
    fret_efficiency: Optional[float]
    project: Mapping[str, Any]


def load_project(path: Path | str) -> MutableMapping[str, Any]:
    """Load a QuEst project JSON file into a mutable dictionary."""

    p = Path(path)
    with p.open("r", encoding="utf8") as fp:
        return json.load(fp)


def build_donor_from_project(
    project: Mapping[str, Any],
    pdb_path: Optional[Path | str] = None,
) -> DonorDecay:
    """Create and configure a :class:`DonorDecay` from a project mapping."""

    p = project

    attachment = p.get("attachment") or {}

    fret_block = p.get("fret") or {}
    dyes = fret_block.get("dyes") or []
    if not dyes:
        raise KeyError("Project must define at least one dye with a diffusion coefficient 'D'.")
    donor_dye = dyes[0]
    donor_D = donor_dye.get("D")
    if donor_D is None:
        raise KeyError("Donor dye entry must define diffusion coefficient 'D'.")

    d = DonorDecay(
        tau0=p["tau0"],
        kQ=p["kQ"],
        nph=p["n_photons"],
        verbose=False,
        auto_update=False,
        pdb=None,
        attachment_residue=attachment.get("residue"),
        attachment_atom=attachment.get("atom"),
        attachment_chain=attachment.get("chain"),
        dg=p["dg"],
        sticky_mode=p.get("sticky_mode", "quencher"),
        save_avs=p.get("save_avs", False),
        D=donor_D,
        slow_fact=p["slow_fact"],
        critical_distance=p["critical_distance"],
        output_file=p.get("output_file", "out"),
        av_parameter=p["av_parameter"],
        quencher=p["quencher"],
        t_max=p["t_max"],
        t_step=p["t_step"],
        slow_radius=p["slow_radius"],
        all_quencher_atoms=p.get("all_quencher_atoms", True),
    )

    structure_path = pdb_path or p.get("pdb")
    if structure_path is None:
        raise ValueError("Project must define a 'pdb' path or pdb_path must be given.")
    d.structure = str(structure_path)

    fret = p.get("fret") or {}
    if fret.get("enabled"):
        d.fret_enabled = True
        dyes = fret.get("dyes") or []
        if len(dyes) > 1:
            acceptor = dyes[1]
            acc_att = acceptor.get("attachment") or {}
            d.acceptor_chain = acc_att.get("chain")
            d.acceptor_residue = acc_att.get("residue")
            d.acceptor_atom = acc_att.get("atom")
            av_param_acc = acceptor.get("av_parameter")
            if av_param_acc is not None:
                d.fret_av_parameter = av_param_acc

        R0_matrix: Sequence[Sequence[Optional[float]]] = fret.get("R0_matrix") or []
        if R0_matrix and len(R0_matrix[0]) > 1:
            r01 = R0_matrix[0][1]
            if r01 is not None:
                d.fret_R0 = float(r01)

    return d


def _histogram_decay(
    model: DonorDecay,
    nbins: int,
    tac_range: Tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    edges, donor_counts = model.get_histogram(nbins=nbins, range=tac_range)
    time = 0.5 * (edges[:-1] + edges[1:])
    return time, donor_counts


def _curve_decay_from_model(
    model: DonorDecay,
    nbins: int,
    tac_range: Tuple[float, float],
    dt_tac: Optional[float],
    n_curves: Optional[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate legacy curve-mode decay using per-frame quenching rates.

    This uses the existing diffusion trajectory inside ``model`` together with
    a simple frame-wise quenching rate derived from ``model.kQ`` and the
    boolean collision trajectory. It falls back to the standard histogram
    behaviour if anything unexpected happens.
    """

    if nbins <= 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    lo, hi = float(tac_range[0]), float(tac_range[1])
    if dt_tac is not None and dt_tac > 0.0:
        dt = float(dt_tac)
    else:
        span = hi - lo
        if span <= 0.0:
            span = float(nbins)
        dt = span / float(nbins)

    decay = np.zeros(int(nbins), dtype=np.float64)

    try:
        diffusion = model.diffusion
        collided = diffusion.quenched
        t_step = float(diffusion.t_step)
    except Exception:
        # If diffusion is unavailable, reuse the standard histogram path.
        return _histogram_decay(model, nbins, tac_range)

    kQ = float(getattr(model, "kQ", 0.0))
    if kQ <= 0.0:
        # No dynamic quenching; approximate with a pure exponential.
        time = np.linspace(lo, lo + dt * nbins, nbins, endpoint=False, dtype=float)
        decay[:] = np.exp(-time / float(model.tau0))
        return time, decay

    # Build a per-frame quenching rate array from the collision trajectory.
    k_quench = np.where(collided > 0, kQ, 0.0).astype(np.float32)

    curves = int(n_curves) if n_curves is not None else 0
    if curves <= 0:
        try:
            curves = int(getattr(model, "n_photons", 0)) // 1000
        except Exception:
            curves = 0
    if curves <= 0:
        curves = 1024

    simulate_decay_quench(
        n_curves=curves,
        decay=decay,
        dt_tac=dt,
        k_quench=k_quench,
        t_step=t_step,
        tau0=float(model.tau0),
    )

    edges = np.linspace(lo, lo + dt * nbins, nbins + 1, dtype=float)
    time = 0.5 * (edges[:-1] + edges[1:])
    return time, decay


def simulate_project(
    project: Mapping[str, Any],
    *,
    pdb_path: Optional[Path | str] = None,
    nbins: Optional[int] = None,
    tac_range: Tuple[float, float] = (0.0, 50.0),
    mode: Optional[str] = None,
    n_curves: Optional[int] = None,
    dt_tac: Optional[float] = None,
    n_tac: Optional[int] = None,
) -> DecaySimulationResult:
    """Run a full QuEst simulation from a project dictionary.

    Parameters
    ----------
    project:
        QuEst project mapping (same schema as the GUI JSON).
    pdb_path:
        Optional override for the structure path.
    nbins:
        Number of TAC bins for the donor (and FRET) histograms. Defaults to
        ``project['n_bins']`` or 4096 if not present.
    tac_range:
        Time range (ns) for the TAC histograms; defaults to ``(0.0, 50.0)``.
    mode:
        Optional legacy simulation mode. If ``None``, this is read from
        ``project['decay_mode']`` when present, otherwise ``"photon"``.
        Currently recognised modes are:

        - ``"photon"`` (default): standard Monte-Carlo photon trace.
        - ``"curve"``: legacy-style curve mode. This uses the same photon
          backend but allows ``n_curves``, ``dt_tac`` and ``n_tac`` to control
          histogram resolution and effective photon statistics.
    n_curves:
        Optional legacy parameter used in ``"curve"`` mode. If provided (or
        present as ``project['n_curves']``), it will multiply the effective
        ``n_photons`` used in the simulation.
    dt_tac:
        Optional legacy bin width in ns for ``"curve"`` mode. If supplied
        (or present as ``project['dt_tac']``) together with ``n_tac``, it will
        be used to derive ``tac_range`` when the caller did not override it.
    n_tac:
        Optional legacy number of TAC bins for ``"curve"`` mode. If supplied
        (or present as ``project['n_tac']``), it overrides ``nbins``.
    """

    # Resolve decay mode with backwards-compatible precedence:
    # 1) explicit function argument, 2) project['decay_mode'], 3) default.
    if mode is None:
        decay_mode = str(project.get("decay_mode", "photon")).lower()
    else:
        decay_mode = str(mode).lower()

    # Legacy configuration for curve mode
    if n_curves is None:
        legacy_n_curves = project.get("n_curves")
        n_curves = int(legacy_n_curves) if legacy_n_curves is not None else None

    if dt_tac is None:
        legacy_dt = project.get("dt_tac")
        dt_tac = float(legacy_dt) if legacy_dt is not None else None

    if n_tac is None:
        legacy_n_tac = project.get("n_tac")
        n_tac = int(legacy_n_tac) if legacy_n_tac is not None else None

    # Determine histogram bin count, allowing curve-mode overrides.
    if nbins is None:
        nbins = int(project.get("n_bins", 4096))
    effective_nbins = nbins
    if decay_mode == "curve" and n_tac is not None:
        effective_nbins = n_tac

    # Derive tac_range from dt_tac / n_tac when running in curve mode and the
    # caller did not explicitly override tac_range via the function argument.
    if decay_mode == "curve" and dt_tac is not None and n_tac is not None:
        # Only adjust if tac_range still at its default and the project did
        # not provide an explicit tac_range.
        project_has_tac_range = "tac_range" in project
        if tac_range == (0.0, 50.0) and not project_has_tac_range:
            tac_range = (0.0, float(dt_tac) * float(n_tac))

    model = build_donor_from_project(project, pdb_path=pdb_path)

    # In curve mode, optionally scale photon statistics by n_curves to mimic
    # the legacy behaviour where multiple curves were accumulated. This only
    # affects the stochastic photon-based quantities (e.g. quantum yield), not
    # the deterministic curve integration below.
    if decay_mode == "curve" and n_curves is not None:
        try:
            model.n_photons = int(model.n_photons) * int(n_curves)
        except Exception:
            # If anything unexpected happens, fall back to the original count.
            pass

    model.update_all(verbose=False)

    if decay_mode == "curve":
        time, donor_counts = _curve_decay_from_model(
            model,
            effective_nbins,
            tac_range,
            dt_tac,
            n_curves,
        )
    else:
        time, donor_counts = _histogram_decay(model, effective_nbins, tac_range)

    fret_counts: Optional[np.ndarray] = None
    qy_fret: Optional[float] = None
    fret_eff: Optional[float] = None

    if getattr(model, "fret_enabled", False):
        xf, yf = model.get_histogram_fret(nbins=effective_nbins, range=tac_range)
        if xf is not None and yf is not None:
            fret_counts = yf
            qy_fret = model.quantum_yield_fret
            fret_eff = model.fret_efficiency

    return DecaySimulationResult(
        time=time,
        donor_counts=donor_counts,
        fret_counts=fret_counts,
        quantum_yield_donor=model.quantum_yield,
        quantum_yield_fret=qy_fret,
        collisions_fraction=model.collisions,
        fret_efficiency=fret_eff,
        project=project,
    )


__all__ = [
    "DecaySimulationResult",
    "load_project",
    "build_donor_from_project",
    "simulate_project",
]
