---
type: Subsystem
title: Photons & decay
description: The numba photon-trace kernels and the two decay modes (photon, curve).
resource: quest/core/photon.py
tags: [monte-carlo, photons, decay, tcspc, numba]
status: stable
generated: { by: process:okf-v0.1-import, at: 2026-07-26T00:00:00Z }
stale_after: 2027-01-26
sources:
  - id: quest-core-photon-py
    resource: ../../quest/core/photon.py
    title: quest/core/photon.py
    author: human:tpeulen
    last_modified: 2026-07-27
---

# From trajectory to decay curve

The excited dye de-excites at rate `1/tau0 + k_quench(t)`, where `k_quench` is
read frame by frame off the trajectory. Everything in
`quest/core/photon.py` is a numba kernel over that array.

| Function | Use |
|---|---|
| `simulate_photon_trace_rate` | The production path — a per-frame **rate** trace; used by `DonorDecay.calc_photons` |
| `simulate_photon_trace_kQ` | Boolean collision trace × a single `kQ` |
| `simulate_photon_trace` | Legacy binary-quenching variant |
| `simulate_decay_quench` | Integrates a decay histogram over `n_curves` — the "curve" mode |

`_simulate_trace_rate` is `parallel=True`; `_simulate_trace_rate_seeded` is the
sequential, reproducible variant, chosen when the project sets a `random_seed`.
The walk starts at a random frame offset (`_photon_rate_walk`) so photons sample
the trajectory rather than always its beginning.

# Two decay modes

`simulate_project(mode=…)`, or `project["decay_mode"]`:

- **`"photon"`** (default) — Monte-Carlo photon trace, histogrammed into
  `n_bins` over `tac_range`. The donor quantum yield is the fluorescent fraction
  of the absorbed photons.
- **`"curve"`** — legacy accumulation. `_curve_decay_from_model` integrates the
  decay over `n_curves` curves at bin width `dt_tac` with `n_tac` bins; when
  `dt_tac` and `n_tac` are both given and the caller did not override
  `tac_range`, the range is derived as `(0, dt_tac × n_tac)`. `n_curves` also
  scales the effective `n_photons`.

Curve mode falls back to the plain histogram if the model has no usable
diffusion object — a deliberate degradation, not an error path.

## Only emitted photons are histogrammed

`_photon_rate_walk` returns `dt = 0.0` with its emitted flag **clear** for a
photon lost to quenching or to transfer, so `dts` and `phs` are the same length
whatever the quantum yield. `get_histogram` and `get_histogram_fret` therefore
count `dts[phs == 1]` only. Histogramming the whole trace — which both did until
2026-07-28 — piles every non-emitted photon into the first bin: a spike of
photons that were never emitted, and a curve whose total is the *excitation*
count rather than `QY × n_photons`. On the D–A channel that is the larger half
of the trace (`1 − QY_DA`), so the transfer efficiency was invisible in the
amplitude of a decay that always summed to `n_photons`.

The curve therefore carries amplitude information: `donor_counts.sum() ≈
QY × n_photons`, short only by the photons whose lifetime falls outside
`tac_range`. This is the same rule `simulation.py` already applied to the mean
lifetime (`dts[phs == 1]`).

# FRET channel

When `fret.enabled`, `DonorDecay` computes the acceptor AV and then a **rate per
trajectory frame** (`calc_fret_rate_trace`):

```
k_FRET(t) = (1/tau0) · ⟨ (R0 / |r_D(t) − r_A|)^6 ⟩_A
```

The donor is resolved in time — which is the reason its diffusion is simulated
at all — and the acceptor is averaged over its accessible volume, i.e. the
fast-acceptor limit (not valid for an immobilised acceptor). Quenching and
transfer are parallel de-excitation channels, so `calc_photons_fret` adds
`k_quench + k_fret` frame by frame and feeds the same photon kernel as the donor
channel.

Two things this fixed, both silent before: the rate was a **single scalar** from
⟨R_DA⟩_E over the static clouds, making the FRET channel independent of the
diffusion; and a missing acceptor produced a donor-only result inside a bare
`except`. It now raises.

`fret_efficiency` has **one** definition, `1 − QY_DA/QY_D`, taken from the two
simulated decays. `rda_mean_e()` remains available as a *diagnostic* distance —
what a FRET-efficiency measurement would report — but nothing in the simulation
consumes it.

Both clouds hold dye *centres* and can overlap, where `(R0/r)^6` would diverge;
the distance is floored at the sum of the two dye radii, since the spheres
cannot interpenetrate.

The kernel (`quest/core/av.py fret_rate_trace`) is general fluorescence machinery
and is slated to move into imp-tricks —
[specs/imp-tricks-dedup.md](/specs/imp-tricks-dedup.md), `DUP-03`.
