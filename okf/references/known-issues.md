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

## imp-tricks' own AV backends do not run in `arm64` either

Verified 2026-07-27 by calling `IMP.bff.av.compute_av` from the checkout:

- `backend="labellib"` — its guard requires `LabelLib.AV`; the installed
  LabelLib has only `dyeDensityAV1`, so the backend reports itself unavailable
  (and would call a missing function if forced). **Fixed** in the local
  imp-tricks checkout, along with the attachment atom being left in its own
  obstacle list — which made that backend return an empty volume with no error.
  It is now verified point-for-point identical to QuEst's LabelLib path
  (`TestTheUpstreamAvMatchesQuEsts`).
- `backend="imp_bff"` — `'Particle' object has no attribute
  'get_particle_index'`; IMP 2.24 spells it `get_index()`. Fixed by a shim, but
  the backend **still does not run**: it raises `UsageException: Can't get
  attribute that is not there: mass on particle 1`, because it extracts its
  density by feeding the source particle to an `IMP.em.SampledDensityMap` that
  requires `IMP.atom.Mass`, which it never sets. Even with a mass, that would
  sample a Gaussian blob around the attachment atom rather than reading the
  path map's accessible density — so this branch has never produced an
  accessible volume at all. It also never passes `allowed_sphere_radius`,
  leaving it below the ~2 Å cliff.

Both are version skew and defects in **imp-tricks**, not in QuEst. The
`imp_bff` branch must be rewritten before `DUP-01` (delegating QuEst's AV to
`compute_av`) can land.

*Not* an issue: reaching imp-tricks from source. `quest.imp_tricks` splices the
checkout in automatically, deliberately **without installing** it, since the
source tree is what is changing
([workflows/environment.md](/workflows/environment.md)).

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
