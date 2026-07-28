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
programme was agreed by interview on 2026-07-28 and **twelve of its thirteen
phases are done** (seventeen commits in `quest`, two in `imp-tricks`, one in
`chisurf`). **Every `LAY-*` row is closed, on both sides, and
`known-issues.md` has no open entry left.**

**Phase 12 is closed as "not doing it": imp-tricks stays local.** The owner
ruled publication out on 2026-07-28, reversing the interview decision. Nothing
in the code is waiting on it — the consequence is CI, below. The decisions behind it — and
the three places they *reverse* what this bundle used to assert — are in
[specs/programme-2026-07.md](specs/programme-2026-07.md). Read that before
proposing work in these areas; several "obvious" next steps there have already
been evaluated and rejected with reasons.

# State

All in `arm64`. **The base environment is retired** — one interpreter now runs
everything ([workflows/environment.md](workflows/environment.md)).

| | tests |
|---|---|
| library + Qt GUI + web backend, one command | **531 passed, 1 skipped** |
| notebooks + examples (`-m notebooks`) | **16 passed, 2 skipped** |
| browser e2e (Playwright, both servers up) | **27 passed** |
| ChiSurf `quenching_estimator` | **14 passed** |
| notebooks + examples (`-m notebooks`, cold cache) | **16 passed** |
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
| 8 | `4539b22` | **`quest.api` is a facade**; serialization moved to the transport; contract, `error_code` taxonomy, `quest.contract.describe`. Closes `LAY-04` |
| 9 | `82b75d4`, `8218c22` | **One mechanism for long-running work**, cooperatively cancelable; `jobs` (running) split from `runs` (artefacts). Closes `LAY-07` |
| 10 | chisurf `669fce499` | **The plugin declares itself**: manifest, `api`/`rpc`/`cli`/`gui`, no `core/`, a dockable widget, and no `quest` import at ChiSurf startup. Closes `LAY-05`'s ChiSurf half |
| 11 | `0dd0ac5` | i18n gets an end condition: frontend keys must resolve, and the 90-string chrome backlog is a ratchet |
| 13 | `a78709a` | `known-issues.md` pruned 358 → 263 lines; every entry closed, two that asserted the opposite of the truth inverted |

# Left: phases 12 and 13

1. ~~**Phase 12 — publish imp-tricks**~~ — **not doing it.** Everything stays
   local by owner ruling (2026-07-28). The recipe is written
   (`~/dev/imp-tricks/rattler-recipe/recipe.yaml`, `noarch: python`) and the
   version is deliberately still `0.1.0`, to be decided at release. If that ever
   changes: publish to anaconda.org (`tpeulen`) and PyPI, then pin the version
   in `pixi.toml` and `rattler-recipe/recipe.yaml` in place of the current `*`.

   **The standing consequence is that two CI jobs cannot solve.** `pixi.toml`
   declares `imp-tricks = "*"` on the `py` feature, and no public channel has
   it, so the `default` and `desktop` environments are unsolvable:

   | CI job | environment | state |
   |---|---|---|
   | `test` | `default` (`py` + web + test) | **red — cannot solve** |
   | `gui` | `desktop` (`py` + gui + web + test) | **red — cannot solve** |
   | `frontend` | `frontend` (separate solve-group) | green |
   | `package` | `build` (`no-default-feature`) | green |

   **Earlier notes said "all four jobs", which is wrong** — `frontend` and
   `build` never carry the `py` feature. Corrected 2026-07-28 by reading
   `pixi.toml`'s `[environments]` against `.github/workflows/ci.yml`.

   A permanently red job is worse than a missing one, because it trains people
   to ignore the light. **This is unresolved and wants a decision** — see the
   next section.
   **This needs credentials and is the one phase that cannot be done
   unattended.** The recipe is written
   (`~/dev/imp-tricks/rattler-recipe/recipe.yaml`) and the version is bumped.
   Publication is what turns CI green — the `*` constraint makes all four jobs
   red today, deliberately.

2. **Phase 13 — the last of the closeout**: rewrite this file once 12 lands.
   The prune of [references/known-issues.md](references/known-issues.md) is
   done (`a78709a`).

# A trap the catalog hides

**A parameter added to `quest/settings/parameter_catalog.json` reaches
`doc/parameters*.md` and the Qt form automatically — and the web UI never.**
`quest/gui/quest.view.json` is *generated* from the catalog;
`webui/frontend/src/components/AppLayout.tsx` is hand-written React. `fret.kappa2`
shipped on 2026-07-28 with two of the three surfaces wired and nothing failing.

Until the web panel is generated from the catalog too, adding a parameter is a
**three-place** edit, and only two of them are automatic.

# Still open, needing a decision

- **What to do about the two permanently red CI jobs**, now that publication is
  off the table indefinitely. Three options, none taken: make `imp-tricks` an
  *optional* pixi feature so `default`/`desktop` solve and the IMP-dependent
  tests skip explicitly; disable the two jobs with a comment pointing here; or
  leave them red and accept that CI signals nothing for the library and GUI.
  The first is the only one that keeps CI meaningful, and it costs an honest
  accounting of what is then untested.
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
