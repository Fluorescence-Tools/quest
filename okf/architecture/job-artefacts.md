---
type: Architecture
title: Job artefacts
description: What a run writes to jobs/<uuid>/, and why bulk work turns it off.
resource: quest/core/
tags: [outputs, mrc, csv, jobs, performance]
status: stable
generated: { by: process:okf-v0.1-import, at: 2026-07-26T00:00:00Z }
stale_after: 2027-01-26
sources:
  - id: quest-core
    resource: ../../quest/core/
    title: quest/core/
    author: human:tpeulen
    last_modified: 2026-07-28
---

# A run is a directory

`simulate_project` mints a UUID per run and writes everything under
`<project_dir>/jobs/<uuid>/`. `project_dir` defaults to the current working
directory, which is why a repo-root `jobs/` folder accumulates during
development (it is git-ignored; the web backend keeps its own under
`QUEST_WEBUI_DATA_DIR`).

| File | Content |
|---|---|
| `project.json` | The exact project dict that produced the run — the run is reproducible from it |
| `result.json` | Scalars and curves: quantum yields, lifetime, contact fraction, AV/contact volumes, FRET efficiency, `aa_residence` |
| `decay.csv` | Time axis and donor (and FRET) counts |
| `results.csv` | The scalar metrics in tabular form |
| `aa_residence.csv` | Per-residue-type contact and quenching breakdown |
| `av.mrc`, `acv.mrc` | Accessible volume and accessible **contact** volume as density grids |
| `diffusion_coefficients.mrc`, `quenching_rates.mrc` | The per-voxel physics grids, when present |
| `sampled_dye_density.mrc` | Where the trajectory actually went, voxelised |
| `out.xyz`, `out_slow.xyz` | AV / slow-AV point clouds |
| `trajectory.npy` | The raw walk |

The `.mrc` writer is hand-rolled in `quest/core/simulation.py` (`write_mrc`) — a 1024-byte
MODE 2 header plus float32 data, with the origin in words 50–52 so the grid
lands on the structure in a molecular viewer.

# `save_outputs=False` is a real optimisation, not a flag

`simulate_project(..., save_outputs=False)` skips the whole directory *and* the
thinned point clouds. For a residue scan — which reads only the scalar metrics
and `aa_residence` — the artefacts dominate the runtime and nothing ever reads
them. Both `quest scan` and the web `/api/scan` use it. Anything that loops over
sites should too.

Note that `save_outputs` also drives `save_avs` in the project dict, so the AV
point clouds are not written either.

# Consequence for the viewer

Because the full clouds stay on disk, what travels through `result.json` to a 3D
viewer is *thinned* (`trajectory_display_points`, default 2000). A viewer that
needs the full cloud reads the `.xyz`/`.mrc` files from the job directory.
