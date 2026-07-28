---
type: Reference
title: Known issues
description: Defects and environment problems that are real, understood, and not yet fixed.
resource: okf/log.md
tags: [known-issues, environment, technical-debt]
timestamp: '2026-07-26T00:00:00Z'
---

Each entry states what is wrong, how it shows up, and what the fix would be.
Verified on 2026-07-26 unless noted. Remove an entry when it is fixed, and say
so in [okf/log.md](../log.md).

# Environment

## ~~QuEst's IMP.bff accessible-volume backend does not work~~ — fixed

**Resolved 2026-07-27. IMP.bff is now the primary backend**
([subsystems/accessible-volume.md](/subsystems/accessible-volume.md)). Kept
here because the sequence of faults is the record of how a backend with no test
coverage fails: each fix revealed the next, and the last two produced *plausible
wrong answers* rather than errors.

1. `path_map.get_path_map_header()` raises `AttributeError` — IMP 2.24 binds it
   to a C symbol (`DensityMap_get_path_map_header`) absent from the compiled
   `_IMP_bff`. Fixed by falling back to the writable accessor.
2. `path_map.get_xyz_density()` returns a NumPy array, and the code truth-tested
   it (`if xyz_density:`) → `ValueError`. Fixed.
3. The backend returned an *empty* volume. Cause: `allowed_sphere_radius` was
   left at QuEst's 0.5 Å default, and below ~2 Å IMP's path search starts
   nowhere and reports nothing. Fixed by the
   `MIN_IMP_ALLOWED_SPHERE_RADIUS = 2.0` floor. It was originally written up
   here as "the path search is not producing a reachable region", which was the
   symptom, not the cause.
4. The backend re-read the structure from disk, discarding coarse-graining and
   failing on mmCIF. Fixed by taking the atom array
   ([specs/structure-io.md](/specs/structure-io.md)).
5. It shipped a `Model.get_particles()` call that does not exist. Fixed, and
   `TestTheImpAvBackendRuns` now exercises the path.
6. The density grid was **transposed** (x↔z) and its origin was one voxel out.
   Found only when the backend became the default and quenching quietly
   vanished from the reference sites. See the registration section of
   [subsystems/accessible-volume.md](/subsystems/accessible-volume.md).
7. It was not thread-safe, so a pooled scan died in SWIG. Fixed with
   `_IMP_BUILD_LOCK`.

*Still true:* the goal remains to delete this integration in favour of
`IMP.bff.av.compute_av` rather than maintain a second copy — `DUP-01`/`DUP-02`
in [specs/assessment.md](/specs/assessment.md), direction in
[specs/imp-tricks-dedup.md](/specs/imp-tricks-dedup.md). That is blocked on the
upstream backend below.

## ~~imp-tricks' own AV backends do not run in `arm64` either~~ — fixed

Verified 2026-07-27 by calling `IMP.bff.av.compute_av` from the checkout:

- `backend="labellib"` — its guard requires `LabelLib.AV`; the installed
  LabelLib has only `dyeDensityAV1`, so the backend reports itself unavailable
  (and would call a missing function if forced). **Fixed** in the local
  imp-tricks checkout, along with the attachment atom being left in its own
  obstacle list — which made that backend return an empty volume with no error.
  It is now verified point-for-point identical to QuEst's LabelLib path
  (`TestTheUpstreamAvMatchesQuEsts`).
- `backend="imp_bff"` — **had never produced an accessible volume in its life.**
  `'Particle' object has no attribute 'get_particle_index'` (IMP 2.24 spells it
  `get_index()`) was only the first layer. Underneath: it read its density by
  feeding the source particle to an `IMP.em.SampledDensityMap` requiring an
  `IMP.atom.Mass` it never set, so every call raised `UsageException` — and with
  a mass it would have sampled a Gaussian blob around the attachment atom rather
  than the path map's accessible density. It never passed
  `allowed_sphere_radius`, leaving it below the ~2 Å cliff; it decorated the AV
  onto the source particle and resampled with `shift_xyz=False`, leaving the map
  at the coordinate origin; and it reshaped the flat tile values in the wrong
  axis order. **All fixed** (imp-tricks `b95c000`), with `TestComputeAvBackends`
  added upstream — there had been no test touching `compute_av` at all, which is
  why a backend that could not work looked fine.

Both were version skew and defects in **imp-tricks**, not in QuEst, and both are
now verified point-for-point identical to QuEst's on 148l E118, E36 and E90.
`DUP-01` is no longer blocked on correctness — only on packaging, since QuEst
still reaches imp-tricks through a path shim rather than a declared dependency.

*Not* an issue: reaching imp-tricks from source. `quest.imp_tricks` splices the
checkout in automatically, deliberately **without installing** it, since the
source tree is what is changing
([workflows/environment.md](/workflows/environment.md)).

## `pytest -m notebooks` fails as one selection; the two files pass separately

Run apart, `tests/test_notebooks.py` gives **9 passed** and
`tests/test_examples.py` gives **7 passed** — the 16 the handover records. Run
as one selection, `pytest tests -m notebooks` gives **6 failed, 10 passed**.

The failure is a dylib conflict in the child process, not anything about the
science: an example launched as a subprocess (or a notebook kernel) resolves
`libimp_kernel.2.24.0.dylib` against `/Users/tpeulen/mambaforge/lib/libboost_filesystem.dylib`
— the **base** environment's boost, not `arm64`'s — and dies with
`Symbol not found: ..._ZN5boost10filesystem6detail15path_algorithms11filename_v3E...`.
Whichever of the two files runs second is the one that fails.

**It is not confined to the notebooks.** Any QuEst test that probes the domain
layer in a subprocess hits the same fault the moment `import quest.core` reaches
IMP at module scope — `test_importing_the_domain_does_not_import_qt` and
`test_the_model_needs_no_qt` both did, when `DUP-02` first imported
`IMP.bff.av._kernels` at the top of `quest/core/av.py`. The import was made lazy
instead ([subsystems/accessible-volume.md](/subsystems/accessible-volume.md)),
which is the right shape anyway, but it means **this environment fault is being
routed around rather than fixed**, and the constraint is now load-bearing: the
domain layer must not import IMP until something actually needs it.

**Pre-existing**, verified 2026-07-28 by running the same selection in a
worktree at `1807e72`, before the AV decisions: identical 6 failed / 10 passed.
It is the same shape as the browser suite's exclusion already documented in
`pyproject.toml` — a process that has loaded IMP is not a clean parent for a
child that also needs it. *Fix:* give each file its own process (as the
`test-notebooks` pixi task should), or scrub the inherited dynamic-loader
environment before spawning. Until then **do not report "16 passed" from a
single command** — it has never produced that.

## `pyproject.toml` requires `numpy>=2.0`; the working env has numpy 1.26.4

The declared floor is not met by the interpreter QuEst actually runs on, and
that is currently the right way round: `tests/sim_test.log` records the failure
mode under numpy 2.2.6 — the installed IMP extension modules are compiled
against numpy 1.x and abort with "A module that was compiled using NumPy 1.x
cannot be run in NumPy 2.2.6". *Fix:* either rebuild IMP against numpy 2, or
relax the declared floor to match what is supported. Right now the metadata
promises something the stack cannot deliver.

## `pytest` cannot collect without `-p no:pytest-qt`

No Qt binding is installed, and pytest-qt aborts the entire session at
collection. Every test invocation in this repo must pass `-p no:pytest-qt`
(the plugin name; `-p no:qt` does not match). *Fix:* install a Qt binding, or
make the plugin a conditional dev dependency.

## On the LabelLib path the point cloud sits half a voxel off its own grid

`AV.points` and `AV.density` describe the same volume, and on the LabelLib
backend they are **offset from each other by half a voxel on every axis** —
measured on 148l E55 at `dg` 0.5 Å: the centroid of `AV.points` is
`(−0.25, −0.25, −0.25)` Å from the centroid of the cloud rebuilt out of
`AV.density` with `AV.x0`. On the IMP.bff path the two are identical.

The cause is two conventions for the same offset. Every grid kernel —
`_center_grid_indices`, and therefore `subav`, `_slow_factor_grid`,
`_additive_factor_grid` and `calc_slow_av`'s `points_slow` — uses the **integer**
`(ng - 1) // 2`. The LabelLib point cloud instead comes from the true geometric
corner, `origin = source − (ng − 1) / 2 · dg`. For odd `ng` the two coincide;
LabelLib's grid is **92** on this structure, and even.

The consequence is that on the fallback backend the fast cloud (`AV.points`,
which feeds ⟨R_DA⟩, `Rmp` and the saved `.xyz`) is shifted relative to the
contact cloud and the grid the walk actually runs on. It is small — 0.43 Å on a
distance — but systematic, and it is the same *class* of defect as the IMP grid
registration bug fixed on 2026-07-27, which is worth noting because that one
looked equally harmless and silently removed all quenching.

Found 2026-07-28 while verifying `DUP-01`, and **deliberately not fixed there**:
routing through `compute_av` reproduces the existing behaviour on both backends
exactly, and unifying the conventions moves LabelLib results. Which convention
is right is also a genuine judgement call — the float corner is geometrically
true for the grid, the integer one is what the kernels index with — so it wants
its own change, its own re-baseline, and probably its own decision. *Fix:* pick
one offset and use it in both places.

## Coarse-graining matches residues by `res_id` alone, ignoring the chain

`move_center_of_mass` in `quest/core/structure.py` selects a residue's atoms
with `all_atoms['res_id'] == res`, with no chain term. Residue numbering restarts
per chain, so in any multi-chain structure — `tests/data/mc4r_dimer.cif`, every
dimer — the side-chain atoms of chain B's residue 40 are folded into the CB
pseudo-atom of chain A's residue 40. The CB then sits at a centre of mass
averaged over two different residues, which moves both the stickiness centre and
the quenching centre.

Found 2026-07-27 while admitting HETATM records: it surfaced as
`KeyError: 'MUB'`, because a ligand residue shares a `res_id` with a protein
residue. The `KeyError` is fixed (untemplated residues are skipped) but **the
underlying mismatch is not** — it was left alone deliberately, since fixing it
changes coarse-grained numbers and that belongs in its own change with its own
re-baseline. *Fix:* match on `(chain, res_id)`.

# Repository hygiene

## Simulation output accumulates in the working tree

`jobs/` at the repo root had grown to **2292** run directories and **32 GB** by
the time it was measured, alongside the `out.xyz`, `out_slow.xyz`, `test.pdb`,
`test.cif`, `tests/sim_test.log` and `webui-viewer-testpdb.png` leftovers.
`simulate_project` writes to `Path.cwd()/jobs/<uuid>` whenever `save_outputs` is
true, so any run or test started from the repo root leaves one behind.

**Half fixed.** All of those paths are now in `.gitignore`, so they can no
longer be committed by accident — but the accumulation itself is unchanged, and
the directory still has to be emptied by hand. The real fix is on the writing
side: tests and bulk work must pass `save_outputs=False` or a temporary
`project_dir` ([workflows/testing.md](/workflows/testing.md)).

## 30 `.coverage.*` files in the repo root

`quest/utils.py` installs a `sitecustomize.py` coverage hook when `coverage` is
already imported, writing its data file into `os.getcwd()`. They are git-ignored
but never cleaned up. *Fix:* point the hook at a temp directory, or drop it —
it is Python-2-era plumbing (the same module still patches
`unittest.TestCase` for "Python 2.6").

# Documentation

## README points at a vendored path that does not exist here

The notebook links read `modules/quest/notebooks/…`, which is the path QuEst has
when it is vendored inside the sibling ChiSurf tree, not in this standalone
repository (`notebooks/…`). The Travis badge at the top is also dead and still
references the pre-rename `master` branch.
