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

## QuEst's IMP.bff accessible-volume backend does not work

Discovered on 2026-07-27, the first time the suite ran in the canonical `arm64`
environment — which has a **compiled `IMP.bff`** (IMP 2.24), so
`HAS_IMP_BFF` is true there and QuEst selected the IMP path. **34 tests
failed.** Three distinct faults, in order of discovery:

1. `path_map.get_path_map_header()` raises `AttributeError` — IMP 2.24 binds it
   to a C symbol (`DensityMap_get_path_map_header`) that is absent from the
   compiled `_IMP_bff`. **Fixed** by falling back to the writable accessor,
   which is bound to a different symbol.
2. `path_map.get_xyz_density()` returns a NumPy array, and the code truth-tested
   it (`if xyz_density:`) → `ValueError`. **Fixed.**
3. **Not fixed:** with both API faults out of the way the backend returns an
   *empty* accessible volume. `PM_TILE_ACCESSIBLE_DENSITY` filtered by the
   documented value range yields zero tiles, and every tile's
   `PM_TILE_PATH_LENGTH` is the 50000 "unreachable" sentinel — the path search
   is not producing a reachable region at all. Reverse-engineering the current
   contract was stopped deliberately.

Because the backend is unusable, **selection no longer prefers it silently**:
`resolve_av_backend()` defaults to LabelLib — the implementation QuEst has
always actually run and the one its tests pin — and IMP.bff is opt-in via
`QUEST_AV_BACKEND=imp_bff`, which now fails loudly with the list of usable
backends instead of producing an empty volume.

*Fix:* do not repair QuEst's hand-rolled integration. Call
`IMP.bff.av.compute_av` from imp-tricks, which performs the same backend
dispatch and is maintained against IMP's API. `DUP-01`/`DUP-02` in
[specs/assessment.md](/specs/assessment.md);
direction in [specs/imp-tricks-dedup.md](/specs/imp-tricks-dedup.md).

## imp-tricks' own AV backends do not run in `arm64` either

Verified 2026-07-27 by calling `IMP.bff.av.compute_av` from the checkout:

- `backend="labellib"` — its guard requires `LabelLib.AV`; the installed
  LabelLib has only `dyeDensityAV1`, so the backend reports itself unavailable
  (and would call a missing function if forced).
- `backend="imp_bff"` — `'Particle' object has no attribute
  'get_particle_index'`; IMP 2.24 spells it `get_index()`.

Both are version skew in **imp-tricks**, not in QuEst, and both must be fixed
there before `DUP-01` (delegating QuEst's AV to `compute_av`) can land. Until
then QuEst's LabelLib path — which matches the installed LabelLib — is the only
working accessible-volume implementation on this machine.

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
