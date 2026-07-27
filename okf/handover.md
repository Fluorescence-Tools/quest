---
type: Handover
title: Session handover
description: Where the tree stands, what is uncommitted, which decisions are open, and the traps that cost time — written to be the first thing a fresh session reads after the index.
resource: okf/log.md
tags: [handover, state, open-questions]
timestamp: '2026-07-28T00:00:00Z'
---

# Read this, then [`log.md`](log.md)

**Both repositories are committed and clean, and nothing has been pushed.**
QuEst: the 138 accumulated paths landed as nine commits (`5962ce3`…`1fa4226`),
then decisions 1-3 as four more (`1b997ec`, `5b8fc2a`, `4f3dad2`, `ea74ca9`).
imp-tricks: two commits (`936a17c`, `b95c000`) fixing both of its AV backends.

Decision 4 is the one thing left from the previous session's list, and it is
blocked on **publishing imp-tricks**, not on any remaining code.

# State

| | tests | note |
|---|---|---|
| `arm64` (Qt + compiled IMP.bff) | **353 passed, 4 skipped** | canonical interpreter |
| base + web backend | **314 passed, 27 skipped** | no Qt, no compiled `IMP.bff` |
| notebooks | **9 passed** | `tests/test_notebooks.py` alone — see below |
| examples | **7 passed** | `tests/test_examples.py` alone — see below |
| imp-tricks `tests/bff` | **197 passed, 5 failed** | +6 new AV tests; the 5 predate this work |
| browser e2e | 26 passed | *not re-run since the AV changes* |
| ChiSurf `quenching_estimator` | 1 passed | *not re-run since the AV changes* |

The notebooks and examples pass **one file per process**. Run as a single
`pytest tests -m notebooks` they give 6 failed / 10 passed, from a dynamic
loader conflict in the child process — pre-existing, verified at `1807e72`,
and written up in [references/known-issues.md](references/known-issues.md).
The previous handover's "16 passed" came from separate runs; that one command
has never produced it.

The two italicised rows are the previous session's numbers, carried forward.
They were **not** re-run after decisions 1-3, which changed every accessible
volume — treat them as unverified until they are.

All runs above were on a **cold numba cache**. That matters; see Traps.

## Committed in `quest` (138 paths, nine commits)

Five threads landed this session, in order — see [`log.md`](log.md) for which
commit carries which:

1. **AutoForm migration.** Every `.ui` file is gone. The Qt form is
   `quest/gui/quest.view.json`, *generated* from the parameter catalog by
   `quest/gui/generate_view_spec.py`, rendered by ChiSurf's AutoForm, bound to
   `quest/gui/form_model.py::ProjectFormModel` — whose state *is* a project
   dict, so the GUI can no longer drift from the CLI.
   [subsystems/gui.md](subsystems/gui.md).
2. **i18n — English, German, French.** `quest/i18n.py` plus
   `quest/settings/locales/*.json`. Because the catalog is the single
   description, translating it reaches the docs, the Qt form and the web UI at
   once. [subsystems/i18n.md](subsystems/i18n.md).
3. **FRET: the acceptor diffuses and sticks**, with its own `D`, `slow_radius`
   and interaction table; `k_FRET(t)` uses the instantaneous pair separation.
   The old fast-acceptor limit remains as `fret.acceptor_dynamics: "averaged"`.
4. **`render_data` split from `save_outputs`** — they were one flag doing two
   jobs, which cost the Qt form its plots.
5. **Structure I/O reduced to one reader**, taking IMP's radii.
   [specs/structure-io.md](specs/structure-io.md). This is the one that changes
   numbers; see Decisions.

## Committed in `~/dev/imp-tricks` (two commits, unpushed)

Contributed upstream while unblocking QuEst:

- `src/IMP/bff/av/compute.py` — three fixes: `get_particle_index()` →
  a shim accepting `get_index()`; the LabelLib probe accepting `dyeDensityAV1`
  (current builds have no `LabelLib.AV`, so a working install was reported
  missing); and the attachment atom's own vdW sphere left in the obstacle list,
  which made the LabelLib backend return an **empty volume with no error**.
- `src/sitecustomize.py` — extends the `__path__` of subpackages that are
  themselves compiled. Without it `import IMP.bff.av` fails however `sys.path`
  is arranged, and imp-tricks' own suite could not collect (28 errors).
- `tests/bff/test_label.py`, `tests/bff/test_restraints.py` — two tests
  asserted `ImportError` "without a backend" and passed only *because* of the
  LabelLib probe bug. They now monkeypatch the flags.

The 5 remaining imp-tricks failures (`cgdye/rotamer` R0-vs-FRETpredict,
`dye_io` PDB→mmCIF, `polymer` Gaussian-chain PDF) reference none of the changed
code and had never been collected before. Newly *visible*, not newly broken.

# Decisions taken 2026-07-27 — **1-3 implemented, 4 blocked**

All four were answered in the previous session. Three are done; the fourth
cannot be done here and the reason is packaging, not correctness.

1. **`IMP.bff` is the primary AV backend.** Done (`1b997ec`). Promoting it
   surfaced three defects that had been invisible for as long as nothing entered
   the branch — a **transposed** density grid (x↔z, which still overlapped the
   truth by 83 % of its voxels, so counts and bounding boxes looked right while
   quenching silently vanished), a grid origin one voxel out on even edge
   lengths, and no thread safety at all. Details in
   [subsystems/accessible-volume.md](subsystems/accessible-volume.md).

2. **HETATM is no longer stripped; water still is.** Done (`5b8fc2a`). Both
   candidates were measured over all 151 CB sites rather than argued: keeping
   waters as obstacles costs **17 %** of every accessible volume, which is a
   stronger claim than mobile solvent supports. Admitting ligands broke two
   things that had only ever seen amino acids — `to_coarse` and hetero atom
   names. [specs/structure-io.md](specs/structure-io.md).

3. **`allowed_sphere_radius` defaults to 2.1 Å.** Done (`4f3dad2`). It also
   **corrected a claim made in decision 1**: the labelability gap between the
   backends is not mostly this parameter. Raising the default left LabelLib at
   exactly 43/151, because `calculate1R`/`calculate3R` accept `linkersphere` and
   never pass it on — LabelLib has no such knob and frees only the attachment
   atom instead. Only the IMP.bff backend honours it.

4. **imp-tricks as a declared dependency — BLOCKED, and not on correctness.**
   The upstream work is done: both of its AV backends now run and both are
   verified point-for-point identical to QuEst's, pinned by
   `TestTheUpstreamAvMatchesQuEsts` (imp-tricks `936a17c`, `b95c000`; QuEst
   `ea74ca9`). Its `imp_bff` branch had **never produced a volume in its life**
   — see [specs/imp-tricks-dedup.md](specs/imp-tricks-dedup.md).

   What blocks it is that **imp-tricks is not installable**. It is `imp-tricks`
   0.1.0 on a private GitLab (`gitlab.peulen.xyz/tpeulen/imp-tricks`), absent
   from PyPI and conda-forge, and the two fixes above are committed **locally
   and unpushed**. Adding it to `pyproject.toml` and
   `rattler-recipe/recipe.yaml` now would make QuEst uninstallable for anyone
   who cannot resolve that name, and would fail the packaging job in CI. The
   path shim in `quest/imp_tricks.py` stays until imp-tricks is published
   somewhere a resolver can reach.

   *Order of operations when unblocking it:* push imp-tricks → publish it (or
   vendor it, or pin a VCS URL if a private index is acceptable) → then declare
   the dependency → then `DUP-01`-`DUP-03`, which is the largest deletion QuEst
   has left.

# Next steps, in order

1. **Publish imp-tricks**, then decision 4. Nothing else unblocks the largest
   deletion QuEst has left, and the code side of it is already finished and
   pinned.
2. **Fix the coarse-grain chain bug** in
   [references/known-issues.md](references/known-issues.md):
   `move_center_of_mass` matches residues by `res_id` alone, so in any
   multi-chain structure one chain's side chains are folded into another
   chain's CB. Found while implementing decision 2 and deliberately left alone,
   because fixing it moves coarse-grained numbers and deserves its own
   re-baseline.
3. `DUP-02`/`DUP-03` (after decision 4): adopt `IMP.bff.av._kernels.density2points` (verified
   numerically identical, and a superset — it keeps voxel weights) and
   `random_distances` + `av_pair_statistics`.
4. **Do not** adopt upstream's `r_mp`: `av_pair_statistics` returns ⟨R_DA⟩ under
   that name (its docstring admits "same as `r_da_mean` here"), which is a
   different quantity — 47.65 vs 51.53 Å on 148l E15/E90. QuEst's `dRmp` is
   correct and should go *upstream*.
5. Remaining backlog rows: `LAY-04` (`quest.api` is still a re-export shim),
   `LAY-07` (three unrelated long-running mechanisms, none cancelable),
   `OBJ-01` item 6 (κ² unmodelled), `OBJ-02` (fitting — not started, by
   instruction). [specs/assessment.md](specs/assessment.md).
6. Web UI chrome is still partly English — plot card titles, metric row names,
   empty states, toolbar. `t()` is wired; it is one key at a time.

# Traps that cost time here

- **A warm numba cache hides compile-time failures.** `@njit(cache=True)`
  kernels only re-read configuration when they actually compile, so a green
  suite proves nothing after touching a jitted module. Clear it:
  `find quest -name '__pycache__' -type d -exec sh -c 'rm -f "$1"/*.nbi "$1"/*.nbc' _ {} \;`
  `tests/conftest.py` now pins `NUMBA_NUM_THREADS` and `NUMBA_THREADING_LAYER`
  for exactly this reason.
- **`Structure(path)` defaults to `make_coarse=True`.** A scratch script
  comparing coordinates against a PDB file will look 1.5 Å wrong. Pass
  `make_coarse=False`.
- **uvicorn does not reload.** Restart it before any browser run. Locale files
  are `lru_cache`d per process, so editing one has no effect until restart.
- **The frontend needs Node ≥ 20**; `/opt/homebrew/bin/node` is 26, the one on
  `PATH` is 18.15 and Next refuses to start.
- **`-p no:pytest-qt` only where there is no Qt binding.** The plugin aborts
  collection in the base environment, so the flag is required there. In `arm64`
  pytest-qt *is* installed and the flag is harmful: it removes the `qtbot`
  fixture, which is what made ChiSurf's own chimol widget tests error. Passing
  it reflexively to another project's suite will look like that project is
  broken.
- **`import IMP.bff` succeeding does not mean the compiled AV exists.** In the
  base environment the *source* `IMP.bff` imports (nothing shadows it) while the
  C++ `AV` class is absent. Guard on `quest.core.imp_av.HAS_IMP_BFF`.
- **Verify that an edit actually applied.** Three `str.replace` calls in a row
  silently no-oped here because the anchor text had already changed, and it only
  surfaced as a test failing on a stale value.
- **A backend with no coverage will ship broken code.** `imp_av.py` carried a
  `Model.get_particles()` call that does not exist; nothing noticed because
  `AV_BACKENDS` starts with LabelLib and no test entered that branch.
