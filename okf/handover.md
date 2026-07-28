---
type: Handover
title: Session handover
description: Where the tree stands, what the 2026-07 programme has closed, what is left, and the traps that cost time — written to be the first thing a fresh session reads after the index.
resource: okf/log.md
tags: [handover, state, open-questions]
timestamp: '2026-07-28T18:00:00Z'
---

# Read this, then [`specs/programme-2026-07.md`](specs/programme-2026-07.md), then [`log.md`](log.md)

**Everything is committed and clean; nothing is pushed.** A thirteen-phase
programme was agreed by interview on 2026-07-28 and **phases 0–7 are done**
(eight commits in `quest`, one in `imp-tricks`). The decisions behind it — and
the three places they *reverse* what this bundle used to assert — are in
[specs/programme-2026-07.md](specs/programme-2026-07.md). Read that before
proposing work in these areas; several "obvious" next steps there have already
been evaluated and rejected with reasons.

# State

All in `arm64`. **The base environment is retired** — one interpreter now runs
everything ([workflows/environment.md](workflows/environment.md)).

| | tests |
|---|---|
| library + Qt GUI + web backend, one command | **461 passed, 1 skipped** |
| notebooks + examples (`-m notebooks`) | **16 passed, 2 skipped** |
| browser e2e (Playwright, both servers up) | **26 passed** |
| ChiSurf `quenching_estimator` | **1 passed** |
| imp-tricks `tests/bff` | **201 passed, 24 skipped, 5 failed** — the 5 predate all of this |

Nothing in that table is stale: every row was measured on 2026-07-28.

**Numbers moved twice, and both are re-baselined in `tests/baselines/`.** If you
are comparing against anything computed before 2026-07-28, read the phase 3 and
phase 4 entries in [log.md](log.md) first — the grid registration and the
residue identity were both wrong, on both backends.

# Done: phases 0–7

| # | Commit | What |
|---|---|---|
| 0 | `b979744` | Run artefacts opt-in and out of the checkout (39 GB / 2870 dirs deleted); `-m notebooks` fixed; `quest/utils.py` deleted |
| 1 | `d394efc` | One `arm64` environment; base retired; optional-dependency fallbacks tested by simulated absence |
| 2 | `40743de` | Numeric baselines; browser e2e and the ChiSurf plugin verified for the first time in three sessions |
| 3 | `fc80e9c` | **One Ångström↔voxel map.** Two registration defects, both backends |
| 4 | `83bac32` | **Residue identity from IMP** — `(chain, res_id, insertion_code)`. Closes `DUP-06` |
| 5 | `d9ba940` | κ² stated (`fret.kappa2`, default 2/3, no numeric change); `DUP-03` rejected; `OBJ-02` decided |
| 6+7 | `dae3763`, imp-tricks `3dbefc8` | Upstream `r_mp` and `split_av_acv` fixed; `subav` and `dRmp` adopted; `DUP-02` closed |

# Left: phases 8–13

In dependency order. Nothing here is started.

1. **`LAY-04` — the facade.** `quest.api` becomes verb-oriented over a client
   (`InProcessClient` by default), and **`result_payload` moves out of the
   handlers to the transport**, so an in-process call returns native arrays.
   That rule is not tidiness: `OBJ-02`'s host optimiser calls `simulate`
   thousands of times.
2. **The contract.** `quest/api/contract.py` on ChiSurf's pattern —
   `PLUGIN_ID`, `CONTRACT_VERSION`, `service_success`/`service_error` **with
   `error_code`**, `contract_descriptor()`, a `quest.contract.describe` method,
   and a stable `error_code` taxonomy. Guardrail: a per-method round-trip test
   asserting the payload validates against the manifest schema. Reference:
   `chisurf/plugins/modelling/fps_json_editor/api/contract.py`.
3. **`LAY-07` — cancellation.** One manager mirroring ChiSurf's
   `chisurf/server/jobs.py` contract (same statuses, `progress`, cooperative
   `should_cancel`), implemented in QuEst so it still runs standalone. **Rename
   the artefact side to `runs`** and free `jobs` for the async sense, matching
   the host. Document the limitation: `_simulate_traj` computes a whole
   trajectory in one `njit` call, so cancel means "at the next checkpoint" and
   a single-site simulate may have none.
4. **The ChiSurf plugin, rewritten from scratch** on the `fps_json_editor`
   template, reusing `cs` machinery. `quest/gui/` stays in QuEst; the plugin
   embeds it as a dockable widget rather than a `QMainWindow`, imports lazily
   (it currently pulls QuEst → IMP at ChiSurf startup) and registers QuEst's
   RPC methods into the host dispatcher. Closes the ChiSurf half of `LAY-05`.
5. **i18n `P1`** — a coverage test across en/de/fr *and* the catalog, then the
   remaining web-UI chrome. Note the asymmetry: `en.json` carries only `ui` and
   `categories` because English lives in the catalog, while de/fr carry a full
   `parameters` block, and nothing currently checks they still match the
   catalog's keys.
6. **Publish imp-tricks 0.2.0** to anaconda.org (`tpeulen`, noarch) **and**
   PyPI; then pin `imp-tricks >=0.2.0` in `pixi.toml` and the recipe. **This
   needs credentials and is the one phase that cannot be done unattended.** The
   recipe is written (`~/dev/imp-tricks/rattler-recipe/recipe.yaml`).
   Publication is what turns CI green — the `*` constraint makes all four jobs
   red today, deliberately.
7. **OKF closeout** — totals in [specs/assessment.md](specs/assessment.md),
   prune [references/known-issues.md](references/known-issues.md), rewrite this
   file.

# Still open, needing a decision

- **Where ChiSurf's fps.json `payload.py` belongs** (`DUP-07` remainder). The
  owner is not convinced `chisurf/core` is right. Nothing depends on it.
- **`asa`/`spherePoints` → `IMP.cgmol.sterics.asa`** (`DUP-05` remainder). Cold
  path; deliberately not rushed.
- **The scratch files in the tree**: `out.xyz`, `out_slow.xyz`, `test.pdb`,
  `test.cif`, `webui-viewer-testpdb.png`, `tests/sim_test.log`. All gitignored.
  `sim_test.log` is cited by a known-issue that has since been corrected.

# Traps that cost time here

Ordered by how much. The first four were all learned or corrected on
2026-07-28.

- **Clear *both* numba caches.** `tests/conftest.py` imports ChiSurf, which sets
  `NUMBA_CACHE_DIR=~/.chisurf/cache` into `os.environ`, so under pytest numba
  reads and writes *there* — not beside the source. The command recorded here
  for two days cleared only `quest/**/__pycache__`, and 193 kernels were sitting
  in the other directory. Both, every time:

      find quest -name '__pycache__' -type d -exec sh -c 'rm -f "$1"/*.nbi "$1"/*.nbc' _ {} \;
      find ~/.chisurf/cache \( -name '*.nbi' -o -name '*.nbc' \) -delete

- **A reference implementation that copies the code under test is not a
  reference.** `tests/test_grid_kernels.py` compared the kernels against a
  hand-written version that had copied their `int()`, so it was structurally
  unable to see a one-voxel error affecting half the grid. Derive a reference
  from the specification.
- **A test that has never been *seen* to fail is not evidence.** Five times now.
  The most recent: an upstream test asserted `r_mp == 50.0` and passed only
  because the function returned the wrong quantity under that name.
- **Check the bridge was invoked before blaming the other repository.** A probe
  written `import quest.imp_tricks` (the module, without calling
  `enable_imp_tricks()`) reported `IMP.bff.av` and `IMP.cgmol` missing, and
  "fix the `IMP.cgmol` import" was carried into a work plan as an upstream
  defect. There was none. `import quest` alone does not splice; `quest.core`,
  `quest.core.av` and `quest.api` do.
- **Do not pass `-p no:pytest-qt`.** It was mandatory while base had no Qt
  binding. `arm64` has pytest-qt, and the flag removes the `qtbot` fixture.
- **`Structure(path)` defaults to `make_coarse=True`.** A scratch script
  comparing coordinates against a PDB file will look 1.5 Å wrong.
- **uvicorn does not reload.** Restart it before any browser run. Locale files
  are `lru_cache`d per process, so editing one has no effect until restart.
- **The frontend needs Node ≥ 20**; `/opt/homebrew/bin/node` is 26, the one on
  `PATH` is 18.15 and Next refuses to start.
- **`playwright` must come from pip.** conda-forge's `playwright` is the *Node*
  package; installing it leaves `import playwright` failing.
- **Verify that an edit actually applied.** A slice-based deletion here removed
  93 lines of dead kernel *and* the 40-line adapter written minutes earlier;
  only the next test run caught it.
