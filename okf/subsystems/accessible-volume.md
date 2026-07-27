---
type: Subsystem
title: Accessible volume
description: AV enumeration, the IMP.bff/LabelLib backend split, and the derived slow-factor and quenching-rate grids.
resource: quest/core/av.py
tags: [accessible-volume, av, grid, numba, labellib, imp]
timestamp: '2026-07-27T00:00:00Z'
---

# What the AV is

The set of positions the dye **centre** can occupy: reachable from the
attachment atom along a linker of `linker_length` × `linker_width`, without the
dye sphere (`radius1`, or `radius1..3` for AV3) overlapping protein atoms.
Represented as a cubic density grid of spacing `dg` plus the corresponding point
cloud. `quest/core/av.py::AV` owns it.

# Backend selection

`resolve_av_backend(preference)` decides, from the `av_backend` argument, else
the `QUEST_AV_BACKEND` environment variable, else `"auto"`:

| Value | Behaviour |
|---|---|
| `"imp_bff"` | `quest/core/imp_av.py::build_imp_accessible_volume` — **the default** |
| `"labellib"` | `calculate1R` (AV1) / `calculate3R` (AV3), thin wrappers over `ll.dyeDensityAV*` |
| `"auto"` (default) | IMP.bff if a *compiled* one is present, else LabelLib, else `RuntimeError` |

An unavailable or misspelled backend raises immediately, naming what *is*
usable, rather than failing later with an empty volume. `AV.av_backend` records
which one ran, and `verbose=True` prints it.

`simulation_type='AV3'` only takes effect on the LabelLib path.

**IMP.bff is primary as of 2026-07-27**; LabelLib is the fallback for platforms
without a compiled `IMP.bff`, Windows above all. Before that flip the IMP path
re-read structures from disk, which discarded coarse-graining and broke mmCIF;
it now takes the atom array
([specs/structure-io.md](/specs/structure-io.md)).

## The two backends do not agree, and the choice is visible in the results

Fed the same atom array with the same radii they differ by algorithm alone:
**136 707 (IMP.bff) against 151 869 (LabelLib) points on 148l E36 — a 0.90
ratio, centroids 0.63 Å apart.**

That 10 % is the small part. Measured over all 151 CB sites of 148l chain E
under the reference project:

| | labelable sites | quenching (0 < QY < 1) |
|---|---|---|
| LabelLib | 43 / 151 | 21 |
| IMP.bff, `allowed_sphere_radius` 2.0 | 147 / 151 | 60 |
| IMP.bff, `allowed_sphere_radius` 2.1 (current) | **151 / 151** | **67** |

So the flip does not merely shift volumes by 10 % — it more than triples how
many sites can carry a dye at all.

**Why, and a correction.** When IMP.bff was promoted this was attributed mostly
to the two backends receiving different `allowed_sphere_radius` values. That was
wrong, and measuring it is what showed it: raising the project default from
0.5 Å to 2.1 Å left LabelLib's count at exactly 43/151, because
`calculate1R`/`calculate3R` accept a `linkersphere` argument and **never pass it
on** — `ll.dyeDensityAV1` has no such parameter. The real cause is how each
backend frees the attachment site: LabelLib zeroes the attachment atom's own vdW
radius and nothing else, so a CB hemmed in by CHARMM-sized neighbours has no
route out; IMP ignores *every* obstacle inside the sphere. Fatter radii made
that difference decisive.

**Results are not comparable across backends**; anything pinned to a number has
to say which one produced it.

> The whole dance already exists as `IMP.bff.av.compute_av`, with the same
> ordering, an explicit `backend=` override, and maintenance against IMP's
> current API. QuEst should call it rather than carry a second copy —
> [specs/imp-tricks-dedup.md](/specs/imp-tricks-dedup.md). Its LabelLib branch
> is verified point-for-point identical to QuEst's; its `imp_bff` branch does
> not run at all, so that deletion is still blocked.

An empty AV raises with a diagnostic naming the residue ("might be buried inside
the protein"), because a zero-point volume otherwise fails much later and much
less legibly.

## Grid registration: the two ways to get it silently wrong

Both were live in the IMP path until 2026-07-27, and both were invisible while
that path was not the default. They are recorded because the failure mode is
the dangerous kind — a plausible volume in the wrong place, not an error.

1. **Axis order.** IMP numbers voxels with *x* fastest
   (`i = x + nx*y + nx*ny*z`), so reshaping the flat tile values C-order into
   `(nx, ny, nz)` transposes the volume, exchanging x and z. An accessible
   volume is globular enough that the transpose still overlapped the truth by
   **83 %** of its voxels: the point count, the bounding box and the total
   volume were all right, and only the *shape* was mirrored. Downstream, the
   contact volume then landed on the wrong side of the protein, so quenchers in
   reach were not seen and donors came out unquenched.
2. **Grid origin.** IMP sizes the map itself, and the attachment atom is on the
   middle voxel only when the edge length is odd — it is 86 on 148l E36. The
   origin must be read from the PathMap header
   (`IMPAccessibleVolumeResult.origin`), never reconstructed from the
   attachment atom. Relatedly, `_points_from_centred_grid` uses the *integer*
   offset `(ng - 1) // 2` so it agrees with `_center_grid_indices`; a float
   `(ng - 1) / 2` puts the point cloud half a voxel off the grid the quenching
   kernels index into, for even `ng` only.

The test that catches both: `AV(av_backend="imp_bff").points` must equal
`build_imp_accessible_volume(...).points` **exactly**, as a set. Comparing
counts or centroids alone passes with the grid transposed.

## Threading

`IMP` builds its decorators through SWIG, which is not safe from several
threads at once — `quest/core/pdb.py` already serialises the parse for this
reason, and the AV path hits the identical failure one layer down
(`'XYZR_setup_particle'` receiving a particle *name*) as soon as a scan runs
sites in a pool, which the web backend does. `_IMP_BUILD_LOCK` guards object
construction and density extraction but **not** `av.resample()`: measured on
148l E36, resample is 636 ms of a 662 ms build (96 %) and the SWIG construction
is 22 ms (3 %), so locking the compute would cost a threaded scan nearly all
its parallelism to guard the phase that is not failing.

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

All hot loops are `@njit` in `quest/core/av.py`, most with
`parallel=True`: `_density2points`, `_subav`, `_slow_factor_grid`,
`_additive_factor_grid`, `_center_grid_indices`, `_ran_dist` (RDA sampling),
`_asa` (solvent accessibility), and the trajectory kernels `_simulate_traj` /
`_simulate_traj_grid`. `tests/test_grid_kernels.py` pins their equivalence and
behaviour — that is the file to extend when a kernel changes.

# FRET distance observables

`RDAMeanE` (⟨R_DA⟩_E via a Förster-averaged sample), `RDAMean` and `dRmp`
(mean-position distance) sample pairs of AV point clouds. `R0` comes from the
project's `fret.R0_matrix`.
