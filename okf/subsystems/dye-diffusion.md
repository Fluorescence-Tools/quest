---
type: Subsystem
title: Dye diffusion
description: SimulateDiffusion and DonorDecay — the Brownian walk inside the AV and the model object that owns a run.
resource: quest/core/dye_diffusion.py
tags: [brownian-dynamics, trajectory, donor-decay, numba]
status: stable
generated: { by: process:okf-v0.1-import, at: 2026-07-26T00:00:00Z }
stale_after: 2027-01-26
sources:
  - id: quest-core-dye-diffusion-py
    resource: ../../quest/core/dye_diffusion.py
    title: quest/core/dye_diffusion.py
    author: human:tpeulen
    last_modified: 2026-07-28
---

# `DonorDecay` — the model object

One instance owns a whole simulation: structure, AV, diffusion, photons, and the
optional FRET channel. `update_all()` runs the sequence:

1. `calc_av()` — the accessible volume
2. `_calc_slow_av()` — the ACV, the slow-factor grid, the k_quench grid
3. `simulate_diffusion()` — the Brownian walk
4. `diffusion.critical_distance = …` — resolve which frames count as contact
5. `calc_photons()` — the decay

If a Qt application happens to be running it drives a progress dialog; headless
it is a plain sequence. `update_all` deliberately does **not** call
`gc.collect()`: the grids it replaces are refcounted NumPy buffers freed on
rebind, and a forced collection cost ~20 ms per run — which a residue scan pays
hundreds of times.

`TransientDecayGenerator` (the Qt widget) *inherits* `DonorDecay`, so the GUI and
the headless path share one model rather than two implementations.

# `SimulateDiffusion` — the walk

Constructed from the `AV`, it runs `fps.simulate_traj` and keeps:

- `_traj` — the (n_frames, 3) walk;
- `k_quench` — the per-frame quenching rate, read out of the AV's `k_quench`
  grid along the trajectory;
- `quenched` / `_is_quenched` — the boolean contact trace;
- `distance_to_mean`, `mean` — the observables the trajectory plots use.

Two kernels back it: `_simulate_traj_grid` (a spatial slow-factor grid, the
amino-acid stickiness mode) and `_simulate_traj` (a scalar slow factor over a
binary slow-AV, the older mode). `DonorDecay.simulate_diffusion` picks the grid
variant whenever `stickiness_mode == 'amino_acid'` and the AV carries a
`slow_factor_grid`.

When `k_quench` is available the contact trace is derived from it
(`k_quench > 0`) rather than from a distance loop, which keeps the decay and the
contact statistics consistent with the grids by construction.

# Reproducibility and parallelism

- `random_seed` in the project (or on the donor dye) seeds the walk. It is
  honoured by both trajectory kernels.
- The photon trace uses a **different but derived** seed
  (`_photon_seed`, base + 7919 mod 2³¹−1) so the photon draws do not replay the
  same sequence as the trajectory they score. With no seed set it returns
  `None`, which keeps the faster multi-threaded photon kernel in play.
- `parallel_trajectories` (project key, `-1` = all cores, capped by
  `MAX_PARALLEL_TRAJECTORIES = 8`) runs independent walks concurrently.
- `_preserved_numba_environment()` exists because numba resolves settings such
  as `NUMBA_NUM_THREADS` **once at import time** — the context manager keeps a
  temporary change from leaking into the rest of the process.

`tests/test_simulation_performance.py` covers the scan fast path and seeded
reproducibility; treat it as the guard when touching any of the above.

# Coarse graining

`coarse_grained: true` (project or donor dye) makes `DonorDecay.structure` build
its `Structure` with `make_coarse=True` — fewer atoms in the steric test, faster
AV, coarser contacts. Covered by `tests/test_coarse_grained.py`.
