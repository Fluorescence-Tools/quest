---
type: Handover
title: Session handover
description: Where the tree stands, what the 2026-07 programme has closed, what is left, and the traps that cost time — written to be the first thing a fresh session reads after the index.
resource: okf/log.md
tags: [handover, state, open-questions]
status: stable
generated: { by: process:okf-v0.1-import, at: 2026-07-28T18:00:00Z }
stale_after: 2026-10-28
---

# Read this, then [`specs/programme-2026-07.md`](specs/programme-2026-07.md), then [`log.md`](log.md)

**2026-08-10 — everything below is now committed, and `.gitignore` was hiding
three frontend sources.** The AutoForm migration and the desktop viewer work
described here as "landed" were all uncommitted modifications; they are five
commits now (`74d7421`, `c79ec26`, `c31e3d2`, `7170f58`, `30fae2c`), each
re-verified rather than trusted. **`lib/` in `.gitignore` was unanchored and
also matched `webui/frontend/src/lib/`**, so `autoformPaths.ts`,
`parameterCatalog.ts` and `projectHelpers.ts` had never been committed and the
frontend could not be built from a clean clone — fixed by anchoring `/lib/`.
The library baseline is **706 passed, 1 skipped** — 695 after the sweep, plus
the eleven tests that came with the desktop-run and CI work below. Nothing is
pushed.

**2026-08-09 — the web UI is AutoForm-only, and the previous headline
baselines changed.** The legacy hand-written React form was deleted from
`webui/frontend/src/components/AppLayout.tsx`; AutoForm (backed by the same
`/api/autoform-spec` JSON the desktop renders through ChiSurf) is unconditional,
with only a JSON overlay toggle remaining. The scan residue picker moved into
`webui/frontend/src/app/scan/page.tsx`. A real **web AutoForm bug** was fixed
en route: the flat `attr` names in the view spec never matched the nested
project schema, so FRET never seeded its acceptor and several fields wrote keys
the backend rejected — see `webui/frontend/src/lib/autoformPaths.ts`.

Final baselines (measured 2026-08-09): **693 passed, 1 skipped** (library + Qt
+ backend) and **54 passed** (browser e2e, Playwright). Both browser failures
that were open on 2026-08-08 (tour cross-page navigation, FRET simulation 500)
are **closed** — the 500 was the acceptor-seeding bug above.

**2026-08-06 — the bundle is now OKF v0.2, and every path in it was re-checked
against the tree.** The migration added the frontmatter families, `sources`,
`okf_version: "0.2"`, a vendored validator, and two new concepts
([subsystems/tasks.md](subsystems/tasks.md),
[references/okf-format.md](references/okf-format.md)).

It also **reopened `known-issues.md`**, which had been empty since 2026-07-28:
the bundle was still describing the pre-refactor tree — `quest/lib/`,
`quest/core.py`, `quest/jobs.py`. Every path has since been corrected against a
`grep` for the symbol, and `overview.md`'s source-layout table was rebuilt. What
was *not* re-read is the prose, so four concepts were marked unverified on
purpose — `subsystems/core-api.md`, `subsystems/structure-io.md`,
`architecture/surfaces.md`, `architecture/simulation-pipeline.md` — because each
describes *where things live* and every one of those places moved.

> **All four were read against the code on 2026-08-10 and are verified.** Three
> were already correct (every symbol resolved by `hasattr`, including the names
> they claim are *gone*); `architecture/surfaces.md` was the one that had rotted
> — it said Qt was imported in "exactly three files" where there are nine, all
> still under `quest/gui/`, so the rule held and the count did not. The two
> surface concepts, [subsystems/webui.md](subsystems/webui.md) and
> [subsystems/gui.md](subsystems/gui.md), were *not* on that list and were the
> genuinely stale ones; they are corrected too.

Re-check any time, from this repo alone:

```bash
python3 okf/references/tools/okf_validate.py okf --version 0.2 \
        --check-paths "quest/,webui/,tests/,doc/"
```

**The baseline is 0 warnings without `--check-paths`** (no concept is stale)
**and 20 with it** (re-measured 2026-08-10) -- every one a sentence that names a
path *because it is gone*. More than 20 means something new rotted. The sentence
below about `known-issues.md` having no open entry was true on 2026-07-28 and is
not now.

**Nothing is pushed.** A thirteen-phase
programme was agreed by interview on 2026-07-28 and **twelve of its thirteen
phases are done** (seventeen commits in `quest`, two in `imp-tricks`, one in
`chisurf`). **Every `LAY-*` row is closed, on both sides.**

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
| library + Qt GUI + web backend, one command | **706 passed, 1 skipped** |
| notebooks + examples (`-m notebooks`) | **16 passed, 2 skipped** |
| browser e2e (Playwright, both servers up) | **54 passed** |
| ChiSurf `quenching_estimator` | **14 passed** |
| notebooks + examples (`-m notebooks`, cold cache) | **16 passed** |
| imp-tricks `tests/bff` | **201 passed, 24 skipped, 5 failed** — the 5 predate all of this |

Library row measured 2026-08-10 (re-measured after the desktop-run change), browser row 2026-08-09; the rest last measured
on 2026-07-28 and unchanged since.

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

   **The two red CI jobs are fixed (2026-08-10), without publishing anything.**
   `default` and `desktop` were unsolvable because the `py` feature required
   `imp-tricks`, which no channel serves — and, it turned out, also `labellib`,
   whose numpy-2 builds are on the **`dev` label** of the `tpeulen` channel
   rather than its root. `pixi lock` now solves all four environments. What it
   cost is stated rather than hidden: a pixi environment cannot compute an
   accessible volume, so **141 tests skip there with that reason and ~550 still
   run**. Details in [workflows/environment.md](workflows/environment.md) and
   [workflows/testing.md](workflows/testing.md).

   | CI job | environment | state |
   |---|---|---|
   | `test` | `default` (`py` + web + test) | **solves; ~550 tests, 141 skipped** |
   | `gui` | `desktop` (`py` + gui + web + test) | **solves** (the Qt suite skips without ChiSurf, as before) |
   | `frontend` | `frontend` (separate solve-group) | green |
   | `package` | `build` (`no-default-feature`) | green |

   **Earlier notes said "all four jobs", which is wrong** — `frontend` and
   `build` never carry the `py` feature. Corrected 2026-07-28 by reading
   `pixi.toml`'s `[environments]` against `.github/workflows/ci.yml`.

   Publishing imp-tricks remains the thing that would make CI cover the
   simulation path, and it still needs credentials.

2. **Phase 13 — the last of the closeout**: rewrite this file once 12 lands.
   The prune of [references/known-issues.md](references/known-issues.md) is
   done (`a78709a`).

# A solved trap: the catalog drove a three-place edit

> **Closed 2026-08-09.** The web form is AutoForm-only now. Previously the Qt form
> was generated from the catalog but the web form was hand-written React, so
> adding a parameter was a three-place edit and the web was the surface that
> fell behind (`fret.kappa2` shipped there on 2026-07-28 with two of three
> surfaces wired).

**A parameter added to `quest/settings/parameter_catalog.json` now reaches
every surface from one spec.** `quest/gui/quest.view.json` is *generated* from
the catalog; `/api/autoform-spec` serves the same `build_view_spec(locale)`;
the desktop renders it through ChiSurf's AutoForm and the web through
`webui/frontend/src/components/autoform/`. The glue that keeps the web in step
with the project schema is `webui/frontend/src/lib/autoformPaths.ts` (spec
`attr` → nested project path), and `tests/test_autoform_blueprint.py` fails if
the catalog grows a parameter that map does not cover.

Adding a parameter is now a **one-place** edit (the catalog) with the mapping
table the one follow-up, not three hand-synchronised surfaces.

# Still open, needing a decision

- ~~**What to do about the two permanently red CI jobs**~~ — **done
  2026-08-10**, by the first of the three options: `imp-tricks` is a pixi
  feature no environment uses, and the tests that need it skip with a stated
  reason (141 of them). The accounting it was said to cost is in
  [workflows/testing.md](workflows/testing.md).
- **Where ChiSurf's fps.json `payload.py` belongs** (`DUP-07` remainder). The
  owner is not convinced `chisurf/core` is right, and **upstream has since ruled
  against it**: ChiSurf's own `okf/specs/chimol.md` (2026-08-10, `a2ff49665`)
  places the labelling plugin — the `LabelStructure` widget that owns this
  payload — **in ChiMOL**, on the grounds that placing a dye on a structure is
  structural work done while looking at the structure. It sits in `modelling/`
  only until ChiMOL hardens, and **the `fps_json_payload` getter/setter seam is
  what the move rides on, so that seam must stay intact.** So the answer is
  neither `chisurf/core` nor here; the question is now *when*, not *where*.
  Nothing in QuEst depends on it.
- **`asa`/`spherePoints` → `IMP.cgmol.sterics.asa`** (`DUP-05` remainder). Cold
  path; deliberately not rushed.
- **The scratch files in the tree**: `out.xyz`, `out_slow.xyz`, `test.pdb`,
  `test.cif`, `webui-viewer-testpdb.png`, `tests/sim_test.log`. All gitignored.
  `sim_test.log` is cited by a known-issue that has since been corrected.

# Traps that cost time here

Ordered by how much. The first four were all learned or corrected on
2026-07-28.

- **"The tree is clean" is not evidence that the tree is committed.** An
  ignored file never appears in `git status`, and `lib/` in `.gitignore` — the
  unanchored entry from the standard Python block — matched
  `webui/frontend/src/lib/`, so three frontend sources lived only in the
  working copy for weeks. Before trusting such a sentence, run
  `git status --ignored=matching --untracked-files=all` and read past the
  `node_modules`/`.pixi` noise.
- **A view-spec `attr` is not the project path.** The spec names fields by the
  flat `ProjectFormModel` attribute (`attachment_chain`, `fret_enabled`,
  `fret_R0`), and the project JSON is nested (`attachment.chain`,
  `fret.enabled`, `fret.R0_matrix[0][1]`). Reading/writing the flat key silently
  produced projects the backend rejected — FRET never seeded its acceptor and
  "worked" as a one-dye donor run. The web resolves through
  `webui/frontend/src/lib/autoformPaths.ts`; the Qt side has `FIELD_PATHS`.
- **Enabling FRET is not `set fret.enabled = true`.** The acceptor dye and R0
  matrix must be seeded (on the last labelable residue, not a water); a bare
  toggle leaves a project that "validates" but cannot transfer. The web routes
  the `fret.enabled` field through `setFretEnabled` for this reason.
- **The Next.js dev server takes > 3 s on a cold request**, which was silently
  skipping the entire browser suite (the module-scoped `_server_is_up` check).
  The timeout is 10 s now; prime the frontend with one request before a run.
- **A styled toggle's real checkbox is hidden** (`.switch input` is
  `display:none`), so Playwright's `.click()` on the switch times out
  intermittently. Drive the checkbox through its label or `evaluate()`, or use
  `.check()` — not a click on the visible `.slider`.
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
  written `import quest.hosts.imp_tricks` (the module, without calling
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
