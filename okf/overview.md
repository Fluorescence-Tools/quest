---
type: Codebase
title: QuEst
description: Structure-based simulator of dynamic PET quenching and FRET for dyes tethered to proteins by flexible linkers.
resource: https://github.com/Fluorescence-Tools/quest
tags: [fluorescence, pet-quenching, fret, accessible-volume, brownian-dynamics, python]
status: stable
generated: { by: process:okf-v0.1-import, at: 2026-07-26T00:00:00Z }
stale_after: 2027-01-26
sources:
  - id: peulen-2017
    resource: https://doi.org/10.1021/acs.jpcb.7b03441
    title: Peulen, Opanasyuk & Seidel, J. Phys. Chem. B 121, 8211-8241 (2017) -- the QuEst method paper
  - id: dimura-2016
    resource: https://doi.org/10.1016/j.sbi.2016.11.012
    title: Dimura et al., Curr. Opin. Struct. Biol. 40, 163-185 (2016) -- accessible contact volumes
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

Rebuilt from the tree on 2026-08-06. The old `quest/lib/` toolkit — which mixed
the PET workflow with a general-purpose structure library shadowing imp-tricks
and ChiSurf — no longer exists; its contents live under `quest/core/`, whose own
`__init__` carries the authoritative module table.

| Path | Role |
|------|------|
| `quest/api.py` | The facade: the verbs, resolved to a **local or a remote** backend. The stable import surface |
| `quest/core/` | The domain — Qt-free and host-free. Everything the simulation needs and nothing about who is driving it |
| `quest/core/simulation.py` | The pipeline: `simulate_project`, `load_project`, `build_donor_from_project`, and the residue-resolved quenching analysis |
| `quest/core/dye_diffusion.py` | `SimulateDiffusion`, `DonorDecay`, and the PET chemistry |
| `quest/core/av.py` | Accessible volumes, the grid kernels, FRET rate traces |
| `quest/core/photon.py` | Photon-trace and decay-curve kernels (numba) |
| `quest/core/structure/` | The `Structure` object (`model.py`), reading through IMP (`reader.py`), labeling-site metadata, and PDB-ID fetch with an on-disk cache (`fetch.py`) |
| `quest/core/fps_json.py`, `quest/core/io.py` | `fps.json` labeling files; point-cloud xyz output |
| `quest/project.py` | The project document: creation and validation |
| `quest/scan.py` | Scanning labeling sites — one site simulated, one record reported |
| `quest/runs.py` | Reading back what a run *wrote*: the artefacts, not the running work |
| `quest/tasks.py` | The one mechanism for long-running work: observable, cooperatively cancelable ([subsystems/tasks.md](subsystems/tasks.md)) |
| `quest/rpc/` | `contract.py` (names, envelopes, error codes), `dispatcher.py`, `services.py` — the operations every front end shares |
| `quest/hosts/` | Optional bridges to a host checkout: `chisurf.py`, `imp_tricks.py` |
| `quest/cli.py` | The Click CLI (`quest simulate/scan/template/validate/gui`) |
| `quest/gui/` | The Qt GUI: `app.py` (`start_gui`), the autoform panel, the dye widget, and the generated `quest.view*.json` |
| `quest/i18n.py` | Translations, from one place, for every surface |
| `quest/settings/` | `parameter_catalog.json` (parameter help, one source of truth), `dye_repository.json` (dye presets), and the locale files |
| `webui/backend/` | FastAPI service wrapping the same `quest.api` calls |
| `webui/frontend/` | Next.js UI (Node toolchain via pixi) |
| `webui/desktop/` | Electron shell around the local backend + frontend |
| `doc/` | User documentation; the parameter pages are **generated** from the parameter catalog |
| `notebooks/` | Library-usage examples |
| `okf/` | This knowledge bundle — the durable memory layer |

The one-word summary of the refactor: **`core/` is what the science is, and
everything beside it — `cli`, `gui`, `rpc`, `hosts`, `webui` — is who is asking.**

# Entry points

- `quest simulate|scan|template|validate|gui` — the CLI (`quest.cli:cli`).
- `from quest.api import load_project, simulate_project` — the Python API.
- `cs.quest` / `quest gui` — the Qt GUI (`quest.gui:start_gui`).
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
