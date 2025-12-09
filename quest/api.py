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
)

__all__ = [
    "DecaySimulationResult",
    "load_project",
    "simulate_project",
    "build_donor_from_project",
]
