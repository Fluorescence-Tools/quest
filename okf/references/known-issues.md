---
type: Reference
title: Known issues
description: Defects and environment problems that are real and understood - the open ones, and the closed ones whose *failure mode* is worth keeping.
resource: okf/log.md
tags: [known-issues, environment, technical-debt]
status: stable
generated: { by: process:okf-v0.1-import, at: 2026-07-26T00:00:00Z }
stale_after: 2026-10-26
okf_check_paths: false   # this file lists paths that are gone, on purpose
---

Each entry states what is wrong, how it shows up, and what the fix would be.
Remove an entry when it is fixed **and its failure mode teaches nothing** — and
say so in [okf/log.md](../log.md).

Several entries here are struck through rather than deleted. They are kept for
one reason each, and it is never sentiment: a backend whose faults each hid the
next, a diagnosis that was confidently wrong twice, a "genuine judgement call"
that turned out to have a right answer, a severity rating that said "has not
bitten yet" about something biting every multi-chain structure. **The rule
they add up to: a plausible explanation is not a diagnosis until it has been
defeated and seen to restore the failure.**

Pruned on 2026-07-28 — the superseded original text of every fixed entry was
removed, leaving the correction. **Every entry in this file is now closed**;
what remains open is in [specs/assessment.md](/specs/assessment.md), and both
items there are decisions rather than defects.

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

*Not* an issue: reaching imp-tricks from source. `quest.hosts.imp_tricks` splices the
checkout in automatically, deliberately **without installing** it, since the
source tree is what is changing
([workflows/environment.md](/workflows/environment.md)).

## ~~`pytest -m notebooks` fails as one selection~~ — fixed 2026-07-28, and the diagnosis below was wrong

**Resolved.** `pytest tests -m notebooks` under `arm64` now gives **16 passed**
as one command. Kept here because the recorded cause was wrong in two separate
ways, and the way it was wrong is the lesson.

The real cause: importing ChiSurf sets **`DYLD_LIBRARY_PATH=<mambaforge>/lib`**
— the *base* prefix — into `os.environ`. In-process that is inert, since dyld
reads the variable once at exec. But **every subprocess inherits it**, so an
`arm64` child resolves `libboost_filesystem.dylib` against base's copy and
`import IMP` dies with `Symbol not found: …filename_v3E`. It is set during
*collection*, by `tests/test_chisurf_autoform.py` calling `autoform_available()`
at module scope — which is why the failure depended on what else was collected
and looked like an ordering effect.

Bisected 2026-07-28: `test_examples.py` + `test_notebooks.py` together give 16
passed; adding `test_pdb_id_support.py`, `test_mc4r.py`, `test_use_cases.py`,
`test_imp_tricks_bridge.py`, `test_service_layer.py` or `test_grid_kernels.py`
still gives 16 passed; adding **`test_chisurf_autoform.py`** gives 6 failed.
`CHISURF_ROOT=/nonexistent` restores 16.

Fixed in `tests/conftest.py` with a `pytest_collection_finish` hook that drops
loader paths pointing outside `sys.prefix`. **That is a containment, not the
fix** — ChiSurf should not export a foreign prefix's `DYLD_LIBRARY_PATH` into
`os.environ` at all, and the proper fix belongs there.

What the old entry got wrong:

- It blamed **base-versus-`arm64` skew**. It is not: with base retired and
  everything running under `arm64`, the failure reproduced identically
  (6 failed / 10 passed).
- It blamed **ordering between the two files** ("whichever runs second fails").
  It is not: the two files together pass.

Both readings were consistent with the symptom, and neither was tested by
defeating it. **That is the lesson worth keeping**: a diagnosis that explains
the symptom is not a diagnosis until it has been defeated and seen to restore
the failure.

## ~~`pyproject.toml` requires `numpy>=2.0`; the working env has numpy 1.26.4~~ — stale

**Measured 2026-07-28:** the canonical `arm64` environment has **numpy 2.4.6**
with **IMP 2.24.0**, and both import and run. The declared floor is therefore
met, and the entry below described the *base* environment (numpy 1.26.4,
IMP 2.17.0) — which is being retired. The `tests/sim_test.log` failure it cites
was an older IMP build compiled against numpy 1.x, not a property of numpy 2.

## ~~`pytest` cannot collect without `-p no:pytest-qt`~~ — inverted 2026-07-28

**The opposite is now true, and passing the flag is harmful.** The entry was
about the base environment, which had no Qt binding, so pytest-qt aborted the
session at collection. Base is retired; `arm64` *has* pytest-qt, and the flag
removes the `qtbot` fixture — which is what made ChiSurf's own widget tests
error when it was passed reflexively to another project's suite.

Kept, inverted rather than deleted, because the flag is still written into
`pixi.toml`'s tasks and into older notes, and a reader who finds it there needs
to know it is now wrong here. See [workflows/testing.md](/workflows/testing.md).

## ~~On the LabelLib path the point cloud sits half a voxel off its own grid~~ — fixed 2026-07-28, and it was two defects, on both backends

**Fixed**, and the entry below understated it in three ways. There is now one
index map everywhere — `quest.core.av.grid_center_index` — and
`tests/test_grid_registration.py` holds it.

1. **It was not confined to LabelLib.** The float corner was also used by
   `_simulate_traj`, `_simulate_traj_grid` and
   `_sample_grid_at_trajectory`, so on **both** backends the walk read the
   quenching and stickiness grids at indices half a voxel from where
   `_center_grid_indices` had stamped them, whenever `ng` was even — which is
   the normal case (86 and 80 at the reference site).
2. **There was a second, larger defect underneath it.** `_center_grid_indices`
   computed `int((p - r0) / dg)`, and `int()` truncates **toward zero**: for any
   centre on the negative side of the anchor it rounded *up*, while every other
   map in the project rounds down. That is a one-voxel error per axis for half
   the grid, on both backends, independent of parity. It was found by asserting
   that the two index maps agree — not by a number looking wrong.
3. **The "genuine judgement call" was not one.** On the IMP path
   `x0 = grid_origin + ((ng-1)//2)·dg`, so voxel *i* is at `grid_origin + i·dg`
   *by construction*; the integer offset is the map that inverts that, and
   integer voxel indexing cannot express the float corner. LabelLib anchored
   `x0` on the **attachment atom**, which lands on a voxel node only for odd
   `ng`.

The decisive measurement, which is the one to repeat if this is ever
re-litigated: run every point of `AV.points` through the *kernels'* index
formula and ask whether it lands on a voxel `AV.density` calls occupied.
Before: **37478/37859 = 98.99 %** on LabelLib, 100 % on IMP. After: **100 % on
both**.

**Numbers moved on both backends**, and are re-baselined in
`tests/baselines/`. Corroboration rather than proof, but worth recording: the
two independently-computed backends agree far better than they did — donor
quantum yield at 148l E96 was 0.089 (LabelLib) against 0.458 (IMP.bff), a
factor of 5.1; it is now 0.073 against 0.085, a factor of 1.2.

## ~~Coarse-graining matches residues by `res_id` alone, ignoring the chain~~ — fixed 2026-07-28, and it was not one function

**Fixed**, and the proposed fix below (*"match on `(chain, res_id)`"* in
`move_center_of_mass`) would have left the defect standing. That function was a
symptom. **`Structure.residue_dict` was itself keyed on `res_id`**, as was
`residue_ids`, so *every* consumer inherited the collapse — the CA/CB/C/N/H
lookup tables built from it, `sequence`, the dihedrals, and which CB a labelling
site attaches to. `l_residue` was also filled in `residue_dict` order while
`move_center_of_mass` indexed it by position in `residue_ids`: two
independently constructed orderings that agreed only because both iterated a
`set()` of small integers.

Residue identity now comes from **IMP's hierarchy, at the reader**:
`quest/core/structure/reader.py` assigns a `residue_index` per distinct
`(chain, res_id, insertion_code)` in file order, and adds an `insertion_code`
column it had been discarding. Everything groups on `residue_index`;
`residue_ids` survives for *display* with a docstring saying so.

The domain still passes plain numpy arrays — no IMP object crosses into it,
which is what keeps the optional-dependency rule and the lazy-import constraint
intact.

**What moved, and only this:** the CB centroid of `tests/data/mc4r_dimer.cif`,
`[-1.400, 2.220, 1.264] → [0.733, -0.419, -0.851]`. `148l.pdb` is
byte-identical — it has three chains but no colliding residue *number*, which
is what makes it the control. The whole baseline diff for this change is those
three numbers.

Note that `n_residue_ids_with_more_than_one_cb` stays **534** on the dimer. That
counts how ambiguous `res_id` is in the *structure*, which the fix does not
change; the CB coordinates are what the fix corrects.

This closes `DUP-06` in [specs/assessment.md](/specs/assessment.md).

## ~~Simulation output accumulates in the working tree~~ — fixed 2026-07-28

**Fixed at the cause, not with a broom.** `simulate_project`'s `save_outputs`
now defaults to **False**, and when a caller does ask for artefacts and gives no
`project_dir` they go to `quest.jobs.user_jobs_directory()` — the per-user data
directory — never `Path.cwd()`. The web backend stopped passing `REPO_ROOT`,
which is what had been steering the browser's runs into the checkout, and its
job list now reads the same per-user root, so a run list no longer depends on
where uvicorn was started. The CLI passes `save_outputs=True` explicitly, since
its user asked for a run record.

`quest.jobs.purge_jobs(project_dir, older_than_days=…, dry_run=True)` is the
domain operation behind `quest jobs clean` and `pixi run clean`; it deletes only
UUID-named directories and reports without deleting unless told otherwise.
`QUEST_JOBS_DIR` overrides the root for tests and bulk work.

The accumulated **2870 directories / 39 GB** were deleted on 2026-07-28.

## ~~30 `.coverage.*` files in the repo root~~ — fixed 2026-07-28

`quest/utils.py` is **deleted**. Nothing in the tree imported it: it was a
`sitecustomize.py` coverage hook writing into `os.getcwd()`, a `set_search_paths`
helper left over from the `sys.path` injection `LAY-08` removed, and
`unittest.TestCase` monkey-patching for "Python 2.6". The stale claim in
[subsystems/structure-io.md](/subsystems/structure-io.md) that
`quest/__init__.py` calls `utils.set_search_paths` is corrected there.

# Documentation

## ~~README points at a vendored path that does not exist here~~ — fixed 2026-07-28

The notebook links pointed at `modules/quest/notebooks/…`, the path QuEst has
when vendored inside the sibling ChiSurf tree, not in this standalone
repository. They now name the four notebooks that exist and are executed by
`tests/test_notebooks.py`, plus the scripted equivalents in `examples/`. The
dead Travis badge — still referencing the pre-rename `master` branch — is
replaced by the GitHub Actions one.

# Still open

## Upstream: `IMP.bff`'s C++ fps.json reader collapses a 3-radius AV to 1 radius — logged 2026-08-10

**Not QuEst's bug, and it does not change a QuEst number** — recorded because
QuEst *writes* the affected fields, so a consumer of a QuEst fps.json inherits
it.

ChiSurf found (`chisurf 47eb871cf`, PRD-95) that `AV::set_av_parameter`
(`AV.cpp:208-210`) does `set_radius2(r[0]); set_radius3(r[0])` — radius2 and
radius3 are parsed from the file into `r[1]`/`r[2]` and then overwritten with
radius1. **Every three-radius AV built from an fps.json through that reader is
silently a one-radius AV.** The fix belongs in imp.bff.

Why QuEst is unaffected, checked rather than assumed:

- QuEst never calls that setter. `quest/core/av.py` passes
  `dye_radii=(radius1, radius2, radius3)` straight into
  `IMP.bff.av.compute_av` from Python; the C++ fps.json path is not on its
  route.
- QuEst is **AV1-only** — `simulation_type` is validated in `AV.__init__` and
  anything but `"AV1"` raises, so `radius2`/`radius3` reach no kernel that
  reads them ([subsystems/accessible-volume.md](/subsystems/accessible-volume.md)).

**What to watch**: `quest/core/fps_json.py` writes `radius2` and `radius3` into
the fps.json it emits (defaults 4.5 and 3.5). Those values are honest, and a
reader with this bug will discard them without saying so. If an fps.json round
trip ever disagrees about dye size, this is the first thing to check — and it
will look like a QuEst export bug.

## This bundle points at files that no longer exist — found 2026-08-06

**The concepts describe the pre-refactor tree.** `quest/lib/` is gone entirely,
`quest/core.py` became the package `quest/core/`, `quest/jobs.py` became
`quest/runs.py`, `quest/quest_gui.py` became `quest/gui/app.py`, `imp_av.py`
disappeared into `IMP.bff.av.compute_av`, and `quest/api/contract.py` is now
`quest/rpc/contract.py`. The frontmatter parses, every link resolves, the prose
reads well — nothing *looks* broken, which is why it drifted silently.

**How it was found**, and how to re-check:

```bash
python3 okf/references/tools/okf_validate.py okf --version 0.2 \
        --check-paths "quest/,webui/,tests/,doc/"
```

Every path a concept names is tested for existence. `log.md` is exempt —
history is supposed to mention paths that are gone.

**Fixed on 2026-08-06 — paths, not descriptions.** Every substitution below was
made only after locating the symbol with `grep`; nothing was guessed:

| Was | Is | Verified by |
|---|---|---|
| `quest/core.py` (`write_mrc`, `autocorr`, the trajectory walk) | `quest/core/simulation.py` | `def write_mrc`, `calculate_aa_quenching` |
| `quest/lib/tools/dye_diffusion/dye_diffusion.py` | `quest/core/dye_diffusion.py` | `core/__init__` module table |
| `quest/lib/tools/dye_diffusion/photon.py` | `quest/core/photon.py` | module docstring |
| `quest/lib/fps/__init__.py`, `quest/lib/imp_av.py`, `quest/core/imp_av.py` | `quest/core/av.py` | `fret_rate_trace`, `compute_av` |
| `quest/lib/structure/Structure.py` | `quest/core/structure/model.py` | `class Structure` |
| `quest/lib/io/PDB.py` | `quest/core/structure/reader.py` | `core/__init` table |
| `quest/lib/fps_json.py` | `quest/core/fps_json.py` | file exists |
| `quest/jobs.py` (`list_jobs`, `job_summary`, `read_job`, `archive_job`, `is_job_id`) | `quest/runs.py` (`list_runs`, `run_summary`, `read_run`, `archive_run`, `purge_runs`, `is_run_id`) | `grep '^def ' quest/runs.py` |
| `quest/quest_gui.py:start_gui` | `quest/gui:start_gui` | `pyproject.toml [project.gui-scripts]` |
| `quest/api/contract.py` | `quest/rpc/contract.py` | file exists; the spec row now records both |

`overview.md`'s source-layout table was rebuilt from the tree in the same pass.

**Marked stale, then re-read -- all four cleared on 2026-08-06.** Each was
checked name by name against the tree, and one of them was not merely
path-stale but wrong:

| Concept | Verdict |
|---|---|
| `subsystems/core-api.md` | **Rewritten.** It described `quest/api.py` as a re-export shim. Since `LAY-04` it is a verb facade over an RPC client -- `simulate`, `scan`, `template`, `validate`, `describe` going through `quest.rpc`, results staying native with serialization at the transport. The domain re-exports still work, and are documented as such. |
| `architecture/surfaces.md` | **Rewritten.** Three claims were false: the backend no longer borrows the CLI's `_validate_project` (rules live in `quest/project.py`), `TransientDecayGenerator` is a `QWidget` around a `ProjectFormModel` rather than inheriting `DonorDecay`, and `core/av.py` and `core/dye_diffusion.py` no longer import `qtpy` at all. Qt now lives in exactly three files, all under `quest/gui/` -- a structural guarantee, stronger than the try/except one it replaced. |
| `subsystems/structure-io.md` | **Rewritten, and it was the worst.** `Structure` had **none** of the nine geometry methods it documented (`rmsd`, `super_impose`, `cluster`, `onRMSF` ...), and `reader.py` was missing four of seven entry points. IMP is the parser now; `Structure` is a thin accessor over the atom array; the alignment and clustering machinery is gone by design. The concept now names the removals rather than dropping them silently. |
| `architecture/simulation-pipeline.md` | **Verified, barely changed.** Every kernel it names is where it says: `calc_slow_av` and the `_simulate_traj` kernels in `core/av.py`, `_photon_rate_walk` in `core/photon.py`, `get_histogram` in `core/dye_diffusion.py`. The three-stage structure survived the move; only file names changed. |

The physics concepts — `photon-decay`, `quenching-model`, `dye-diffusion`,
`accessible-volume` — were **not** marked stale: the refactor relocated them
(`core/__init__`: "Relocated from the old in-tree `quest/lib` toolkit") without
changing the behaviour they describe, and marking everything would make the
signal worthless.

**The remaining 21 `--check-paths` warnings are the baseline, and all are
correct prose:** sentences that name a path *because it is gone*
("`quest/lib` was removed outright on 2026-07-27", "`quest/utils.py` deleted",
"`elements.py` **deleted** (271 lines)"), a completed task ("Delete the `.ui`
files"), the `doc/parameters*.md` glob, and imp-tricks' own `tests/bff`. A run that reports
more than 21 has found something new.

**Exempt by frontmatter:** this file and `architecture/package-layout.md` carry
`okf_check_paths: false` — one is an audit of what rotted, the other cites names
that were deliberately never created. Flagging either trains readers to ignore
the warnings.

The lesson is one this bundle already states and did not survive: a concept is
updated **in the same change as the code**, or it quietly becomes fiction.

## Decisions, not defects

The two remaining defects live in
[specs/assessment.md](/specs/assessment.md) as `DUP-05`'s `asa` remainder and
`DUP-07`'s `payload.py` destination, and both are decisions rather than bugs.
