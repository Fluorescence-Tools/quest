"""High-level Python API for QuEst (Quenching Estimator).

This module provides a small, GUI-independent interface around the core
`DonorDecay` simulation so that you can use QuEst from scripts and
Jupyter notebooks.

Typical usage
-------------

>>> from pathlib import Path
>>> from quest.api import load_project, simulate_project
>>> project = load_project(Path("my_project.quest.json"))
>>> result = simulate_project(project)
>>> result.time.shape == result.donor_counts.shape
True

The API is intentionally minimal and focused on a single labeled donor
(with optional FRET acceptor). The underlying project schema is the same
as the JSON written by the QuEst GUI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Tuple

import numpy as np

# Re-export the headless backend so downstream imports stay stable.
from .core import (
    DecaySimulationResult,
    build_donor_from_project,
    load_project,
    simulate_project,
    top_quenchers,
)
from .project import (
    ProjectValidationError,
    template_project,
    validate_project,
)
from .structure_fetch import (
    download_pdb_id,
    is_pdb_id,
    resolve_structure_path,
)
from .core.fps_json import (
    load_fps_json,
    normalize_to_fps_positions,
    run_fps_scan,
)

__all__ = [
    "DecaySimulationResult",
    "load_project",
    "simulate_project",
    "build_donor_from_project",
    "top_quenchers",
    "ProjectValidationError",
    "template_project",
    "validate_project",
    "download_pdb_id",
    "is_pdb_id",
    "resolve_structure_path",
    "load_fps_json",
    "normalize_to_fps_positions",
    "run_fps_scan",
]
