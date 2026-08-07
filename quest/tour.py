"""Shared Tour & Demo Mode Engine for QuEst.

Provides single shared tour definitions (`quest/settings/tour.json`) and demo configuration
for 148L lysozyme, consumed by both the Web UI and PyQt desktop application.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .project import template_project

TOUR_JSON_PATH = Path(__file__).resolve().parent / "settings" / "tour.json"
DEMO_PDB_PATH = Path(__file__).resolve().parents[1] / "tests" / "148l.pdb"

__all__ = ["TOUR_JSON_PATH", "DEMO_PDB_PATH", "get_tour_data", "create_demo_project"]


def get_tour_data() -> dict[str, Any]:
    """Read the shared guided tour JSON definition."""
    if not TOUR_JSON_PATH.is_file():
        return {"steps": []}
    with TOUR_JSON_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
        data["demo_pdb_path"] = str(DEMO_PDB_PATH) if DEMO_PDB_PATH.is_file() else None
        return data


def create_demo_project() -> dict[str, Any]:
    """Create a fully configured demo project dict using 148l.pdb."""
    project = template_project()
    if DEMO_PDB_PATH.is_file():
        project["structure"]["path"] = str(DEMO_PDB_PATH)
        project["structure"]["chain"] = "E"
        project["structure"]["residue"] = 118
        project["structure"]["atom"] = "CB"

        # Set acceptor dye site on chain E
        if "fret" in project:
            project["fret"]["acceptor_chain"] = "E"
            project["fret"]["acceptor_residue"] = 130
            project["fret"]["acceptor_atom"] = "CB"

    return project
