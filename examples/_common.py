"""Shared setup for the examples.

The examples are meant to run in a few seconds each, so they use a small test
protein and short trajectories. Real work uses longer ones -- every example says
where to turn the knobs up.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = ["example_structure", "example_project"]

REPO_ROOT = Path(__file__).resolve().parents[1]

# Run *this* checkout, whatever the current directory is. Without this an
# editable install elsewhere on the machine can win -- on this machine the
# `quest` distribution is installed editable from a different checkout, so an
# example run from `examples/` would silently exercise that one instead.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def example_structure() -> Path:
    """Return the structure the examples label: T4 lysozyme (PDB 148L).

    Override with ``QUEST_EXAMPLE_STRUCTURE`` to run any example against your
    own structure.
    """

    override = os.environ.get("QUEST_EXAMPLE_STRUCTURE")
    if override:
        return Path(override).expanduser()

    for candidate in (
        REPO_ROOT / "examples" / "data" / "148l.pdb",
        REPO_ROOT / "tests" / "148l.pdb",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No example structure found. Set QUEST_EXAMPLE_STRUCTURE to a PDB/CIF "
        "file, or run from a source checkout."
    )


def example_project(*, with_fret: bool = False, chain: str = "E", residue: int = 100):
    """A small, fast QuEst project on the example structure.

    Parameters are deliberately modest so an example finishes in seconds:

    ============  ========  ===============================================
    parameter     here      production
    ============  ========  ===============================================
    ``t_max``     400 ns    tens of microseconds, so the dye samples its
                            whole accessible volume many times over
    ``n_photons`` 200 000   millions, for smooth decay statistics
    ``dg``        0.5 Å     0.5 Å is already fine; coarser trades accuracy
    ============  ========  ===============================================
    """

    from quest.project import template_project

    project = template_project(with_fret=with_fret)
    project["pdb"] = str(example_structure())
    project["attachment"] = {"chain": chain, "residue": residue, "atom": "CB"}
    project["fret"]["dyes"][0]["attachment"] = dict(project["attachment"])

    project.update(
        t_max=400.0,
        t_step=0.02,
        n_photons=200_000,
        n_bins=256,
        # Reproducible output: drop this for an independent Monte-Carlo sample.
        random_seed=2026,
    )
    return project
