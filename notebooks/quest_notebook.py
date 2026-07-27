"""Shared setup for the QuEst notebooks.

Keeps the boilerplate — finding the checkout, the example structure, a small
fast project — in one place instead of repeating it in every notebook.
"""

from __future__ import annotations

import sys
from pathlib import Path

__all__ = ["REPO", "STRUCTURE", "quick_project", "fret_project"]


def _repo() -> Path:
    here = Path.cwd()
    for candidate in (here, *here.parents):
        if (candidate / "quest" / "__init__.py").exists():
            return candidate
    raise RuntimeError("run these notebooks from inside a QuEst checkout")


REPO = _repo()
if str(REPO) not in sys.path:
    # Ahead of any editable install from another checkout.
    sys.path.insert(0, str(REPO))

#: T4 lysozyme — small, well studied, and it has the tryptophan that makes
#: PET quenching unmistakable.
STRUCTURE = REPO / "tests" / "148l.pdb"


def quick_project(residue: int = 100, chain: str = "E", **overrides):
    """A donor-only project that finishes in seconds.

    Production runs use tens of microseconds and millions of photons; these
    settings are chosen so a notebook stays interactive.
    """
    from quest.project import template_project

    project = template_project(with_fret=False)
    project["pdb"] = str(STRUCTURE)
    project["attachment"] = {"chain": chain, "residue": residue, "atom": "CB"}
    project["fret"]["dyes"][0]["attachment"] = dict(project["attachment"])
    project.update(
        t_max=400.0, t_step=0.02, n_photons=200_000, n_bins=256, random_seed=2026
    )
    project.update(overrides)
    return project


def fret_project(donor: int = 55, acceptor: int = 132, r0: float | None = None, **overrides):
    """A donor+acceptor project. ``r0`` is a Forster radius in **Angstrom**."""
    from quest.project import DEFAULT_FORSTER_RADIUS, template_project

    radius = DEFAULT_FORSTER_RADIUS if r0 is None else float(r0)
    project = template_project(with_fret=True)
    project["pdb"] = str(STRUCTURE)
    project["attachment"] = {"chain": "E", "residue": donor, "atom": "CB"}
    project["fret"]["dyes"][0]["attachment"] = dict(project["attachment"])
    project["fret"]["dyes"][1]["attachment"] = {"chain": "E", "residue": acceptor, "atom": "CB"}
    project["fret"]["R0_matrix"] = [[None, radius], [radius, None]]
    project.update(
        t_max=400.0, t_step=0.02, n_photons=200_000, n_bins=256, random_seed=7
    )
    project.update(overrides)
    return project
