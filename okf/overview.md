---
type: Codebase
title: QuEst
description: Structure-based simulator of dynamic PET quenching and FRET for dyes tethered to proteins by flexible linkers.
resource: https://github.com/Fluorescence-Tools/quest
tags: [fluorescence, pet-quenching, fret, accessible-volume, brownian-dynamics, python]
timestamp: '2026-07-26T00:00:00Z'
---

# What it is

QuEst (QUenching ESTimation) predicts the fluorescence decay of a dye tethered
to a protein by a flexible linker, from the protein structure alone. It models
photo-induced electron transfer (PET) quenching by nearby amino acids and,
optionally, FRET to a second dye. The method is described in
[Peulen et al., J. Phys. Chem. B 2017](https://doi.org/10.1021/acs.jpcb.7b03441).

The simulation runs in three stages — accessible volume, Brownian diffusion,
photon generation — described in
[architecture/simulation-pipeline.md](/architecture/simulation-pipeline.md).

Its practical uses are choosing labeling positions for FRET experiments, and
calibrating accessible *contact* volumes (ACVs) against a measured donor
lifetime.

# Source layout

| Path | Role |
|------|------|
| `quest/core.py` | The headless service layer: `load_project`, `build_donor_from_project`, `simulate_project`, residue-resolved quenching analysis, MRC/grid writers |
| `quest/api.py` | Thin public re-export of `core` + structure fetch + FPS scan; the stable import surface |
| `quest/cli.py` | The Click CLI (`quest simulate/scan/template/validate/gui`) |
| `quest/structure_fetch.py` | Resolve a path *or* a bare 4-character RCSB PDB ID, with an on-disk cache |
| `quest/quest_gui.py` | Entry point for the legacy Qt GUI |
| `quest/lib/fps/` | Accessible-volume calculation, the AV grid kernels, and the diffusion trajectory kernels (numba) |
| `quest/lib/imp_av.py` | Optional IMP.bff accessible-volume backend |
| `quest/lib/tools/dye_diffusion/` | `SimulateDiffusion` + `DonorDecay` — the simulation objects; also the Qt widget `TransientDecayGenerator` |
| `quest/lib/tools/dye_diffusion/photon.py` | Numba photon-trace and decay-histogram kernels |
| `quest/lib/io/`, `quest/lib/structure/` | PDB/CIF/PQR parsing and the `Structure` object (incl. coarse-graining) |
| `quest/settings/` | `parameter_catalog.json` (parameter help, one source of truth) and `dye_repository.json` (dye presets) |
| `webui/backend/` | FastAPI service wrapping the same `quest.api` calls |
| `webui/frontend/` | Next.js UI (Node toolchain via pixi) |
| `webui/desktop/` | Electron shell around the local backend + frontend |
| `doc/` | User documentation; `doc/parameters.md` is **generated** from the parameter catalog |
| `notebooks/` | Library-usage examples |
| `okf/` | This knowledge bundle — the durable memory layer |

# Entry points

- `quest simulate|scan|template|validate|gui` — the CLI (`quest.cli:cli`).
- `from quest.api import load_project, simulate_project` — the Python API.
- `cs.quest` / `quest gui` — the legacy Qt GUI (`quest.quest_gui:start_gui`).
- `uvicorn webui.backend.app.main:app` + `pixi run -e frontend dev` — the web UI.

# Relationship to imp-tricks

General fluorescence and structural machinery — accessible volumes, dye
distributions, FRET distance metrics and rates — belongs in the sibling
**imp-tricks** repository under `IMP.bff`, and QuEst depends on it rather than
carrying its own copy. QuEst today duplicates a good deal of it; the direction
and the migration map are in
[specs/imp-tricks-dedup.md](/specs/imp-tricks-dedup.md).

# Where to go next

- The [simulation pipeline](/architecture/simulation-pipeline.md) — what actually
  runs when a project is simulated.
- The [project schema](/architecture/project-schema.md) — the JSON contract every
  surface reads and writes.
- The [subsystems](/subsystems/index.md) — the code areas.
- The [workflows](/workflows/index.md) — environment, testing, and how changes
  get recorded.

# Citations

[1] Peulen, Opanasyuk & Seidel, *J. Phys. Chem. B* **121**, 8211–8241 (2017), [doi:10.1021/acs.jpcb.7b03441](https://doi.org/10.1021/acs.jpcb.7b03441)
[2] Dimura et al., *Curr. Opin. Struct. Biol.* **40**, 163–185 (2016), [doi:10.1016/j.sbi.2016.11.012](https://doi.org/10.1016/j.sbi.2016.11.012)
