---
type: Handover
title: Session handover
description: Where the tree stands, what is uncommitted, which decisions are open, and the traps that cost time — written to be the first thing a fresh session reads after the index.
resource: okf/log.md
tags: [handover, state, open-questions]
timestamp: '2026-07-27T00:00:00Z'
---

# Read this, then [`log.md`](log.md)

Everything below is **uncommitted**. Nothing has been pushed. Two repositories
are dirty; both are green.

# State

| | tests | note |
|---|---|---|
| `arm64` (Qt + compiled IMP.bff) | **350 passed, 4 skipped** | canonical interpreter |
| base + web backend | **313 passed, 25 skipped** | no Qt, no compiled `IMP.bff` |
| browser e2e | **26 passed** | needs both servers up |
| notebooks + examples | **16 passed** | `-m notebooks` |
| ChiSurf `quenching_estimator` | **1 passed** | plugin still loads |
| imp-tricks `tests/bff` | **191 passed, 5 failed** | the 5 predate this work — see below |

All runs above were on a **cold numba cache**. That matters; see Traps.

## Uncommitted in `quest` (138 paths)

Five threads landed this session, in order:

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

## Uncommitted in `~/dev/imp-tricks` (4 files)

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

# Decisions taken (2026-07-27) — **not yet implemented**

The four open questions were answered. Nothing below has been coded; this is
the work list for the next session, in the order it should be done.

1. **`IMP.bff` becomes the primary AV backend.** Flip `AV_BACKENDS` in
   `quest/core/av.py` to `("imp_bff", "labellib")`. LabelLib stays as the
   fallback for platforms without a compiled `IMP.bff` — Windows above all.
   Expect the suite to move: the two backends agree to a 0.90 ratio on 148l E36
   (136 707 vs 151 869 points, centroids within 0.5 Å), so contact fractions and
   quenching numbers shift by roughly that much and every pinned value needs
   regenerating. `quest/core/imp_av.py` is already array-based and now has
   coverage (`TestTheImpAvBackendRuns`), so the path is exercised — but it has
   never been the *default*, and `tests/test_imp_tricks_bridge.py::
   TestTheImpBackendIsNotReadyToBePrimary` exists solely to assert LabelLib is
   first. **Delete that class as part of the flip**; it has done its job.

2. **Do not strip HETATM or water.** Keep IMP's full atom set as steric
   obstacles rather than QuEst's ATOM-only selector — `_atom_record_selector`
   in `quest/core/pdb.py` goes, and the reader takes what IMP gives it
   (1363 atoms on 148l against the current 1322; the extra 63 are `FGA`, `API`,
   `DAL`, `MUB`, `NAG`, `BME`).
   *Check while implementing:* IMP's own default already excludes water and
   hydrogens — `NonWaterNonHydrogenPDBSelector` returns the identical 1363 — so
   "don't strip" is satisfied by passing **no** selector. If crystallographic
   waters are genuinely wanted as obstacles, that needs
   `IMP.atom.NonHydrogenPDBSelector` (1503 atoms) and is a different, stronger
   choice: mobile solvent occluding a dye is unusual, and it will shrink the
   accessible volume further on top of decision 1. Measure both before picking.

3. **`allowed_sphere_radius` defaults to 2.1 Å.** Replace
   `MIN_IMP_ALLOWED_SPHERE_RADIUS = 2.0` and QuEst's own `0.5` default with a
   single value of **2.1**. This is the radius around the attachment atom inside
   which obstacles are ignored so the linker can leave the atom it is tied to.
   Below ~2 Å the path search starts nowhere and returns an **empty volume with
   no error** — imp.bff's own reference test uses 2.0 and expects zero points at
   1.0. 2.1 sits just above that cliff and matches carbon's radius.

4. **imp-tricks becomes a declared dependency.** Add it to `pyproject.toml` and
   `rattler-recipe/recipe.yaml` so it is installed rather than reached through
   the `quest/imp_tricks.py` path shim. That unblocks `DUP-01`–`DUP-03`:
   QuEst's `AV` class and `imp_av.py` both collapse into
   `IMP.bff.av.compute.compute_av`, and `density2points` /
   `random_distances` + `av_pair_statistics` can replace the in-tree copies.
   The upstream fixes this depends on are in `~/dev/imp-tricks`, **uncommitted**
   — they must be committed and released before QuEst can depend on them.

Sequencing note: 2 and 3 both change the accessible volume, and 1 changes which
code computes it. Do them **one at a time**, re-running and re-baselining
between each, or the resulting number changes cannot be attributed. Decision 4
is independent and can go first.

# Next steps, in order

1. Implement decisions 1–4 above, one at a time, re-baselining between each.
2. `DUP-02`/`DUP-03` (after decision 4): adopt `IMP.bff.av._kernels.density2points` (verified
   numerically identical, and a superset — it keeps voxel weights) and
   `random_distances` + `av_pair_statistics`.
3. **Do not** adopt upstream's `r_mp`: `av_pair_statistics` returns ⟨R_DA⟩ under
   that name (its docstring admits "same as `r_da_mean` here"), which is a
   different quantity — 47.65 vs 51.53 Å on 148l E15/E90. QuEst's `dRmp` is
   correct and should go *upstream*.
4. Remaining backlog rows: `LAY-04` (`quest.api` is still a re-export shim),
   `LAY-07` (three unrelated long-running mechanisms, none cancelable),
   `OBJ-01` item 6 (κ² unmodelled), `OBJ-02` (fitting — not started, by
   instruction). [specs/assessment.md](specs/assessment.md).
5. Web UI chrome is still partly English — plot card titles, metric row names,
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
