---
type: Subsystem
title: Accessible volume
description: AV enumeration, the IMP.bff/LabelLib backend split, and the derived slow-factor and quenching-rate grids.
resource: quest/core/av.py
tags: [accessible-volume, av, grid, numba, labellib, imp]
status: stable
generated: { by: process:okf-v0.1-import, at: 2026-07-27T00:00:00Z }
stale_after: 2027-01-27
sources:
  - id: quest-core-av-py
    resource: ../../quest/core/av.py
    title: quest/core/av.py
    author: human:tpeulen
    last_modified: 2026-07-28
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
| `"imp_bff"` | `quest/core/av.py::build_imp_accessible_volume` — **the default** |
| `"labellib"` | `calculate1R` (AV1) / `calculate3R` (AV3), thin wrappers over `ll.dyeDensityAV*` |
| `"auto"` (default) | IMP.bff if a *compiled* one is present, else LabelLib, else `RuntimeError` |

An unavailable or misspelled backend raises immediately, naming what *is*
usable, rather than failing later with an empty volume. `AV.av_backend` records
which one ran, and `verbose=True` prints it.

**Both rows in that table go through imp-tricks, so "LabelLib is the fallback"
does not mean QuEst can compute an AV without imp-tricks.** The single call is
`IMP.bff.av.compute_av(..., backend=…)` — the dispatch is *upstream's*, and
`density2points` comes from `IMP.bff.av._kernels` too. Measured 2026-08-10 with
`IMP_TRICKS_SRC` pointed at nothing: 98 tests fail and 8 error, including the
`labellib`-parametrised baselines. The choice in the table is a choice *inside*
imp-tricks, not a way to do without it. `imp_tricks_av_available()` is the
predicate for "can this interpreter compute an accessible volume at all", and
what `tests/conftest.py` uses to skip rather than fail
([workflows/testing.md](/workflows/testing.md)).

**`simulation_type` must be `"AV1"`; anything else raises.** This concept
previously said AV3 "only takes effect on the LabelLib path" — it took effect on
neither. The LabelLib branch called `calculate3R`, which was deleted in
`7ecf209`, so `simulation_type="AV3"` raised `NameError: name 'calculate3R' is
not defined` from inside a running simulation; the IMP.bff branch ignores
`simulation_type` outright. Since `simulation_type` is a project field a scan can
set, it is now validated in `AV.__init__` **before the structure is read**, and
names AV1 as the only supported value. QuEst has no three-radius dye model, and
`IMP.bff.av.compute_av` has none either — its LabelLib branch also drives
`dyeDensityAV1` and uses `dye_radii[0]` alone.

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

## It is `compute_av` now (`DUP-01`, 2026-07-28)

**QuEst no longer implements either backend.** `AV.__init__` makes one call to
`IMP.bff.av.compute_av`, passing `backend=` explicitly so the choice is never
implicit, and both in-tree implementations are deleted: `quest/core/av.py`
(329 lines) and `calculate1R` (69). QuEst keeps the site selection, the cubic-grid
and binarisation adapter, and `AV`'s grids — the application, not the algorithm.

Verified before the swap and again after, on 148l E36/E55/E118 under **both**
backends: identical grid shape, identical binary density voxel-for-voxel,
identical point counts — 147 011 / 164 783 / 67 889 (IMP.bff) and
172 964 / 200 283 / 91 585 (LabelLib), unchanged to the last point. A green suite
was not treated as evidence here; the volumes were compared directly.

Two things had to go upstream first, both contributed rather than worked around:

- **A build lock.** `compute_av` had no synchronisation, so a threaded scan died
  in `XYZR_setup_particle` exactly as QuEst's own backend used to. imp-tricks
  `9cd5c13` adds `_IMP_BUILD_LOCK` with the same construction/resample split
  described below, so QuEst's lock moved rather than disappeared.
- **A test that reproduces the race.** The first version used the 165-atom slab
  and four workers and **passed with the lock defeated** — the construction
  window is too short to overlap. It takes ~2000 atoms and 8 workers.

`simulation_type` is validated separately; upstream has no AV3 either.

An empty AV raises with a diagnostic naming the residue ("might be buried inside
the protein"), because a zero-point volume otherwise fails much later and much
less legibly.

## Grid registration: the four ways to get it silently wrong

The first two were live in the IMP path until 2026-07-27; the second two on
**both** paths until 2026-07-28. All four were invisible for the same reason —
the failure mode is a plausible volume in the wrong place, not an error.

**The rule that resolves all of them: there is one map between Ångström and a
voxel index, `quest.core.av.grid_center_index`.** Voxel *i* of a grid anchored
at `x0` sits at `x0 + (i − grid_center_index(ng)) · dg`, and the inverse is
`floor((p − x0)/dg) + grid_center_index(ng)`. Anything that converts between the
two and does not use that map is a bug, not a variant.

The measurement that decides it, and the one to repeat rather than argue:
run every point of `AV.points` through the *kernels'* index formula and check it
lands on a voxel `AV.density` calls occupied. It must be 100 %. On 2026-07-28,
before the fix, LabelLib gave **37478/37859 = 98.99 %**.

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
   attachment atom.
3. **Two offset conventions** (fixed 2026-07-28, both backends). The kernels and
   the point cloud used the integer `(ng - 1) // 2`; the two trajectory kernels
   and `_sample_grid_at_trajectory` used the float corner `(ng - 1) / 2`; and
   the **LabelLib** branch anchored `x0` on the attachment atom, which is on a
   node only for odd `ng`. For even `ng` — 86 and 80 at the reference site, so
   the normal case — the walk therefore read the quenching and stickiness grids
   half a voxel from where they had been stamped. The trajectory pair was
   *self*-consistent, which is exactly why nothing caught it.
4. **Truncation toward zero** (fixed 2026-07-28, both backends, larger than 3).
   `_center_grid_indices` used `int((p - r0)/dg)`. `int()` rounds toward zero,
   so a centre on the **negative** side of the anchor was rounded *up* while
   every other map rounds down — a one-voxel error per axis for half the grid,
   independent of parity. `DonorDecay._sample_grid_at_trajectory` had the mirror
   of it: `np.trunc` maps `[-1, 0)` to `0`, so a dye up to one voxel *below* the
   grid was treated as inside and read voxel 0's quenching rate.

The tests that catch these: `tests/test_grid_registration.py` (every cloud point
indexes to an occupied voxel; the two index maps agree probe-for-probe; the
anchor is on a lattice node; and an assertion that `ng` is *even* here, since
none of it is detectable on an odd edge), plus
`AV(av_backend="imp_bff").points` equalling `build_imp_accessible_volume(...)
.points` **exactly**, as a set — comparing counts or centroids alone passes with
the grid transposed.

**A caution about "reference implementations".** `tests/test_grid_kernels.py`
compared the kernels against a hand-written reference that had *copied* the
kernels' `int()`. It could not detect defect 4 at all: it was a self-comparison
wearing the costume of an independent check. If you write a reference, derive it
from the specification, not from the code under test.

## Threading — the lock lives upstream now

`IMP` builds its decorators through SWIG, which is not safe from several
threads at once — `quest/core/structure/reader.py` already serialises the parse for this
reason, and the AV path hits the identical failure one layer down
(`'XYZR_setup_particle'` receiving a particle *name*) as soon as a scan runs
sites in a pool, which the web backend does.

`_IMP_BUILD_LOCK` is in **`IMP.bff.av.compute`** as of `DUP-01`; QuEst's copy went
with `imp_av.py`. It guards object construction and density extraction but
**not** `av.resample()`: measured on 148l E36, resample is 636 ms of a 662 ms
build (96 %) and the SWIG construction is 22 ms (3 %), so locking the compute
would cost a threaded scan nearly all its parallelism to guard the phase that is
not failing.

**Do not assume a thread-safety test is testing anything.** The upstream one
first used a 165-atom system over four workers and passed with the lock replaced
by a no-op — the construction window was too short for two threads to collide.
It needs ~2000 atoms over 8 workers to fail reliably, and the docstring says so.
Same trap as the vacuous grid test of 2026-07-27: a concurrency test that has
never been seen to fail is not evidence.

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
`parallel=True`: `_subav`, `_slow_factor_grid`, `_additive_factor_grid`,
`_center_grid_indices`, `_ran_dist` (RDA sampling), `_asa` (solvent
accessibility), and the trajectory kernels `_simulate_traj` /
`_simulate_traj_grid`. `tests/test_grid_kernels.py` pins their equivalence and
behaviour — that is the file to extend when a kernel changes.

## `density2points` is upstream's now (`DUP-02`, 2026-07-28)

`quest.core.av.density2points` is a thin adapter over
`IMP.bff.av._kernels.density2points`; QuEst's own `_density2points` is deleted.
The two select exactly the same voxels — verified on 5×7×9, 8×8×8 and 11×11×11
grids — and upstream's is a strict superset, keeping the per-voxel weight
QuEst's signature discards. The adapter drops that column and the oversized
buffer's slack.

Two things had to be true first, and neither was:

1. **Upstream's kernels were bare `@nb.njit`.** QuEst's carried
   `cache=True, nogil=True`, and both are load-bearing: this loop visits every
   voxel and is the most expensive step of a site computation, so a threaded
   scan — which the web backend runs — would have re-serialised on the GIL
   exactly where the work is. Contributed upstream (imp-tricks `a49e17c`) rather
   than worked around, so the swap is not a regression.
2. **The enumeration order differs** — upstream runs `ix` outermost, QuEst's ran
   `iz` outermost — so the rows come back permuted. Safe *only* because nothing
   reads `AV.points` positionally: the walk runs on the density grid,
   `_ran_dist` samples indices at random, `Rmp` averages, and the acceptor cloud
   is averaged over in full. The single visible effect is the row order of a
   saved `.xyz`. `TestDensityToPoints` was the only thing asserting a sequence
   and now compares sorted.

**The import is lazy, and that is not an optimisation.** `_density2points_kernel()`
imports on first use and caches in a module global, because importing
`IMP.bff.av` pulls in `IMP`, and `import quest.core` must not. A module-scope
import failed `test_importing_the_domain_does_not_import_qt` and
`test_the_model_needs_no_qt` — not with an `ImportError`, but with the dylib
conflict in [references/known-issues.md](/references/known-issues.md), because
both probe the domain layer in a subprocess. It would have promoted a documented
latent environment fault into a hard failure of the service layer.

# FRET distance observables

`RDAMeanE` (⟨R_DA⟩_E via a Förster-averaged sample), `RDAMean` and `dRmp`
(mean-position distance) sample pairs of AV point clouds. `R0` comes from the
project's `fret.R0_matrix`.

## The sampling is upstream's (`DUP-03`, 2026-07-28)

`RDAMean` and `RDAMeanE` are aggregations over
`IMP.bff.av._kernels.random_distances` + `IMP.bff.distance_metrics.av_pair_statistics`;
QuEst's `_ran_dist` is deleted. Measured on 148l E15/E90 (64 969 × 71 873
points, 50 000 samples), the two agree inside their own sampling noise:
⟨R_DA⟩ 55.164 ± 0.037 Å against 55.171 ± 0.032, ⟨R_DA⟩_E 54.467 ± 0.022 against
54.473 ± 0.029.

Two things changed, and both are deliberate:

- **It is ~5× slower** — 3.4 ms a call against 0.68 ms. QuEst's kernel was
  `parallel=True`; upstream's is sequential because it seeds explicitly, and
  numba's per-thread RNG cannot be seeded reproducibly. A further 0.87 ms is the
  `(N, 3) → (N, 4)` widening upstream's signature requires; QuEst's grids are
  binary, so that weight column is all ones and the weighted mean reduces to the
  plain one. Acceptable **only because this runs once per simulation** — ~0.4 s
  across a 151-site scan. Do not adopt this kernel on a per-frame path.
- **Results are reproducible, which they were not.** The old parallel kernel drew
  from an unseeded per-thread RNG, so two calls on the same clouds disagreed. The
  seed is now a parameter (`RDA_SAMPLING_SEED`, default 0).

**`dRmp` stays local, and index 1 of `av_pair_statistics` must never be read.**
Upstream returns `r_mp` as a copy of `r_da_mean` — its own docstring concedes it
— and R_mp = |⟨r_D⟩ − ⟨r_A⟩| cannot be recovered from a sample of pair distances
at all, because the sampling destroys exactly that information. 47.65 Å (QuEst)
against 51.53 Å (upstream) on 148l E15/E90. QuEst reads indices 0 and 2 only.

`TestTheAdoptedDistanceKernelsBehave` pins the three properties the swap could
have broken. It uses **E55/E124**: a test spanning both environments must pick
sites labelable under *both* backends, and E15 — the pair the spec quotes — is
not among LabelLib's 43 of 151.
