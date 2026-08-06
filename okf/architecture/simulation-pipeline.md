---
type: Architecture
title: Simulation pipeline
description: The three stages of a QuEst run — accessible volume, Brownian diffusion, photon generation — and where each lives.
resource: quest/core/
tags: [simulation, accessible-volume, brownian-dynamics, monte-carlo]
status: stable
generated: { by: process:okf-v0.1-import, at: 2026-07-26T00:00:00Z }
stale_after: 2026-08-06
sources:
  - id: quest-core
    resource: ../../quest/core/
    title: quest/core/
    author: human:tpeulen
    last_modified: 2026-07-28
---

> **Marked stale on 2026-08-06.** The file paths in this concept were
> corrected against the tree, but its *descriptions* were not re-read --
> it says where each stage lives, and every stage moved. Treat the mechanism it describes as unverified until someone
> reads it against the code. See
> [references/known-issues.md](../references/known-issues.md).
# The three stages

Every QuEst run — from the CLI, the API, the web UI or the GUI — is the same
three-stage pipeline, orchestrated by
[`simulate_project`](/subsystems/core-api.md) in `quest/core/simulation.py`.

## 1. Accessible volume (where the dye can be)

The dye is a sphere on a flexible linker. Given the attachment atom
(chain/residue/atom) and the linker geometry (`linker_length`, `linker_width`,
`radius1..3`), the set of sterically reachable dye-centre positions is
enumerated on a cubic grid of spacing `dg`.

Two further grids are derived from it, both in `AV.calc_slow_av`:

- a **slow-factor grid** — near-surface voxels where the dye diffuses more
  slowly because of unspecific stickiness, and
- a **k_quench grid** — the per-voxel PET quenching rate, summed over every
  residue whose contact sphere covers that voxel.

Code: [subsystems/accessible-volume.md](/subsystems/accessible-volume.md)
(`quest/core/av.py`, `quest/core/av.py`).

## 2. Brownian diffusion (where the dye goes)

A single long random walk of `t_max / t_step` steps is run *inside* the AV, with
the local diffusion coefficient taken from the slow-factor grid, so the dye
lingers where it sticks. The walk is a numba kernel (`_simulate_traj_grid` /
`_simulate_traj`) and reflects at the AV boundary.

The trajectory is the object everything downstream reads: quenching is evaluated
per frame, and the residue-resolved contact statistics are run-length analyses
over the same frames.

Code: [subsystems/dye-diffusion.md](/subsystems/dye-diffusion.md).

## 3. Photons and the decay (what the experiment would see)

Walking the trajectory with an excited-state clock produces the fluorescence
decay. Each frame contributes a de-excitation rate `1/tau0 + k_quench(frame)`;
photons are drawn until `n_photons` are collected (photon mode) or the decay
histogram is integrated directly over many curves (curve mode).

Code: [subsystems/photon-decay.md](/subsystems/photon-decay.md).

# What comes out

A [`DecaySimulationResult`](/subsystems/core-api.md): the decay curve, the donor
quantum yield and mean lifetime, the contact fraction, AV/contact volumes, the
optional FRET channel, the thinned trajectory and point clouds a 3D viewer
needs, and `aa_residence` — the per-residue-type breakdown of contact time and
quenching contribution described in
[subsystems/quenching-model.md](/subsystems/quenching-model.md).

Two flags control how much of that a caller gets, and they are **separate on
purpose**:

- `save_outputs` — write the job directory (`result.json`, `.mrc` volumes,
  `.xyz` clouds, CSV tables) under `jobs/<uuid>/`.
- `render_data` — build the thinned trajectory, autocorrelation and point
  clouds that a plot or a 3D viewer consumes.

They used to be one flag, which conflated *write files?* with *return something
to draw?*. The Qt form passes `save_outputs=False` so an interactive run does not
litter the working directory, and silently lost its trajectory, AV and
autocorrelation plots along with it. A residue scan wants neither and passes
both `False`: thinning point clouds nobody reads, once per residue, is the
overhead the fast path existed to avoid.

# Two invariants worth keeping

- **One trajectory, many read-outs.** The decay, the contact fractions, the
  residence statistics and the autocorrelation all derive from the *same* walk.
  Anything that re-simulates to answer a second question is a bug waiting to
  disagree with itself.
- **The grids carry the physics.** Stickiness and quenching are properties of
  *voxels*, not of a per-frame distance loop. A new interaction term belongs in
  `calc_slow_av`, next to the others.
