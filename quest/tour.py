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
    project = dict(template_project(with_fret=True))
    if DEMO_PDB_PATH.is_file():
        pdb_str = str(DEMO_PDB_PATH.resolve())
        project["pdb"] = pdb_str
        project["attachment"] = {"chain": "E", "residue": 118, "atom": "CB"}

        if "fret" in project and isinstance(project["fret"], dict):
            dyes = project["fret"].get("dyes", [])
            if len(dyes) > 0:
                dyes[0]["attachment"] = {"chain": "E", "residue": 118, "atom": "CB"}
            if len(dyes) > 1:
                dyes[1]["attachment"] = {"chain": "E", "residue": 130, "atom": "CB"}

    return project
