---
type: Subsystem
title: Accessible volume
description: AV enumeration, the IMP.bff/LabelLib backend split, and the derived slow-factor and quenching-rate grids.
resource: quest/lib/fps/__init__.py
tags: [accessible-volume, av, grid, numba, labellib, imp]
timestamp: '2026-07-26T00:00:00Z'
---

# What the AV is

The set of positions the dye **centre** can occupy: reachable from the
attachment atom along a linker of `linker_length` × `linker_width`, without the
dye sphere (`radius1`, or `radius1..3` for AV3) overlapping protein atoms.
Represented as a cubic density grid of spacing `dg` plus the corresponding point
cloud. `quest/lib/fps/__init__.py::AV` owns it.

# Backend selection

`resolve_av_backend(preference)` decides, from the `av_backend` argument, else
the `QUEST_AV_BACKEND` environment variable, else `"auto"`:

| Value | Behaviour |
|---|---|
| `"labellib"` | `calculate1R` (AV1) / `calculate3R` (AV3), thin wrappers over `ll.dyeDensityAV*` |
| `"imp_bff"` | `quest/lib/imp_av.py::build_imp_accessible_volume` — **currently broken**, see below |
| `"auto"` (default) | LabelLib if available, else IMP.bff, else `RuntimeError` |

An unavailable or misspelled backend raises immediately, naming what *is*
usable, rather than failing later with an empty volume. `AV.av_backend` records
which one ran, and `verbose=True` prints it.

`simulation_type='AV3'` only takes effect on the LabelLib path.

**LabelLib is preferred on purpose.** It is the implementation QuEst has always
actually run — every environment without a compiled `IMP.bff` fell back to it —
and the one the tests pin. Preferring IMP.bff silently meant that in the one
environment where it *is* present (`arm64`), QuEst selected a backend that
returns an empty accessible volume; 34 tests failed the first time the suite ran
there. See [references/known-issues.md](/references/known-issues.md).

> The whole dance already exists as `IMP.bff.av.compute_av`, with the same
> ordering, an explicit `backend=` override, and maintenance against IMP's
> current API. QuEst should call it rather than carry a second copy —
> [specs/imp-tricks-dedup.md](/specs/imp-tricks-dedup.md).

An empty AV raises with a diagnostic naming the residue ("might be buried inside
the protein"), because a zero-point volume otherwise fails much later and much
less legibly.

# The three grids

`AV.calc_slow_av` derives everything the diffusion stage needs, in one place:

| Grid | Built by | Meaning |
|---|---|---|
| `density_fast` | AV backend | the accessible volume itself |
| `density_slow` | `subav` | the accessible **contact** volume (ACV) — voxels within `slow_radius` of a slow centre |
| `slow_factor_grid` | `_slow_factor_grid` | per-voxel diffusion scaling; overlapping centres **multiply** |
| `k_quench_grid` | `_additive_factor_grid` | per-voxel PET rate; overlapping centres **add** |

`slow_centers` and `quench_centers` are passed separately, with their own radii,
so stickiness can sit on the residue while quenching sits on its redox-active
moiety ([quenching-model.md](quenching-model.md)).

# Kernels

All hot loops are `@njit` in `quest/lib/fps/__init__.py`, most with
`parallel=True`: `_density2points`, `_subav`, `_slow_factor_grid`,
`_additive_factor_grid`, `_center_grid_indices`, `_ran_dist` (RDA sampling),
`_asa` (solvent accessibility), and the trajectory kernels `_simulate_traj` /
`_simulate_traj_grid`. `tests/test_grid_kernels.py` pins their equivalence and
behaviour — that is the file to extend when a kernel changes.

# FRET distance observables

`RDAMeanE` (⟨R_DA⟩_E via a Förster-averaged sample), `RDAMean` and `dRmp`
(mean-position distance) sample pairs of AV point clouds. `R0` comes from the
project's `fret.R0_matrix`.
