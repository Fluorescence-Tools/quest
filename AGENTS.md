# AGENTS.md

This file provides guidance to AI coding agents working in this repository.

QuEst (QUenching ESTimation) predicts the fluorescence decay of a dye tethered
to a protein by a flexible linker, from the structure alone — PET quenching by
nearby amino acids, plus optional FRET.

## Both surfaces must stay in sync

QuEst has **two user-facing surfaces**: the **PyQt desktop GUI** (`quest/gui/`)
and the **Web UI** (`webui/`). Every feature must work on both. A change that
only reaches one surface is a bug. When adding or changing a feature:

1. **Identify both surfaces.** Search for the feature in both `quest/gui/` and
   `webui/frontend/src/`. If it exists in one, it must exist in the other.
2. **Implement on both.** Do not defer the second surface to a later commit.
3. **Test on both.** The PyQt tests (`tests/test_pyqt_*.py`) and the browser
   e2e (`tests/test_webui_e2e.py`) must both pass.
4. **The sequence viewer, 3D structure viewer, and AA selection** are the most
   common divergence points. Click-to-select a residue in the 3D viewer, the
   sequence bar, or the autoform dropdowns must update the same model state on
   both surfaces.

## Read the OKF bundle first — it is this project's memory

`okf/` is an **Open Knowledge Format v0.2** bundle: plain-markdown concepts with
YAML frontmatter, grouped as `architecture/`, `subsystems/`, `specs/`,
`workflows/` and `references/` under a root `okf/index.md`, plus an append-only
`okf/log.md`. It is the durable, agent-readable knowledge layer that sits beside
the code, and it is where what has been learned about this repository lives.

**Start at [`okf/index.md`](okf/index.md)**, and read the concept that owns the
area you are about to touch, before reading the source. On a conflict between a
code comment and an OKF concept, OKF wins — it is newer.

Since 2026-08-06 every concept carries `status`, `generated`, `stale_after` and
`sources`. Three rules follow, and they are not optional:

- **`stale_after` is a promise to re-read.** When it has passed, the concept is
  not wrong — it is unreviewed. Re-read it against the code, then re-date or fix.
- **`verified` is earned.** Add it only when a human or a named process actually
  checked the concept; the trust tier is derived from it mechanically.
- **Validate before finishing:**
  `python3 okf/references/tools/okf_validate.py okf --version 0.2
  --check-paths "quest/,webui/,tests/,doc/"` — the second flag catches the
  bundle drifting behind a refactor, which is exactly what happened to
  `quest/lib/` ([`okf/references/known-issues.md`](okf/references/known-issues.md)).

Format and shared conventions: [`okf/references/okf-format.md`](okf/references/okf-format.md),
and FPSIMP's [`okf/references/okf-conventions.md`](../fpsimp/okf/references/okf-conventions.md),
which is the canonical copy of the rule set the two share.
The rules are shared with the sibling project **FPSIMP** (`../fpsimp`), whose
agent entry point is [`../fpsimp/AGENTS.md`](../fpsimp/AGENTS.md) — same file,
same role. Every repository in this family uses `AGENTS.md` and no per-tool
instruction file; FPSIMP finished that rename on 2026-08-11. It runs the same
taxonomy and the same validator (vendored into each repository, FPSIMP's copy
canonical, `diff` the sync check) — the two are heading for one server, so keep
them shaped alike.

Fast paths:

- What actually runs during a simulation → [`okf/architecture/simulation-pipeline.md`](okf/architecture/simulation-pipeline.md)
- Where the architecture is heading, and the gap backlog → [`okf/specs/plugin-integration.md`](okf/specs/plugin-integration.md), [`okf/specs/assessment.md`](okf/specs/assessment.md)
- The project-JSON contract every surface shares → [`okf/architecture/project-schema.md`](okf/architecture/project-schema.md)
- The physics and its approximations → [`okf/references/pet-quenching-theory.md`](okf/references/pet-quenching-theory.md)
- Why something is broken → [`okf/references/known-issues.md`](okf/references/known-issues.md)
- What changed recently and why → [`okf/log.md`](okf/log.md)

## Working practices

- **Track everything in OKF — no exceptions.** Every change, finding, decision
  and dead end goes into the owning concept *and* `okf/log.md`, in the same
  change that makes it. That includes packaging, CI, environments and
  documentation, not just code: if it is not in OKF it did not happen, because
  the next session starts from OKF and nothing else.
- **Keep OKF up to date — always, not at the end.** `okf/` is only worth reading
  if it matches the tree, so it is updated *in the same change* as the code, not
  in a later cleanup pass. That includes correcting entries that turn out to be
  wrong: a confident but stale concept is worse than a missing one.
- **Use the `arm64` conda environment — it is the only one.**
  `/Users/tpeulen/mambaforge/envs/arm64/bin/python` runs everything: Python
  3.12, numpy 2.4.6, PyQt5, LabelLib, a **compiled `IMP.bff`**, and (since
  2026-07-28) fastapi, uvicorn and playwright. The mambaforge base interpreter
  is **retired**; the two-environment split and the `-p no:pytest-qt` flag that
  went with it are gone. See
  [`okf/workflows/environment.md`](okf/workflows/environment.md).
- **Write to memory as you go.** Every material change updates the OKF concept
  that owns the behaviour **and** appends a dated bullet to `okf/log.md`, in the
  same change. Anything you *learn* that was not obvious from the code — a
  missing dependency, a default chosen by measurement, a kernel that must stay
  sequential, a physical assumption — goes into the owning concept, or into
  `okf/references/known-issues.md` if it is a defect you cannot fix now. Never
  leave a finding only in the chat. Full loop:
  [`okf/workflows/change-tracking.md`](okf/workflows/change-tracking.md).
- **Docs are part of the change.** `okf/` is for agents; `doc/` and `README.md`
  are for people. Parameter help text is edited in
  `quest/settings/parameter_catalog.json` (translations in
  `quest/settings/locales/*.json`) and regenerated with **both**
  `python doc/generate_parameter_docs.py` and
  `python -m quest.gui.generate_view_spec` — the second writes the Qt form's
  view spec from the same catalog. Never hand-edit `doc/parameters*.md` or
  `quest/gui/quest.view*.json`. Locale files are cached per process, so restart
  uvicorn after editing one.
- **Test all three surfaces, every time — library, web UI, and Qt GUI.** They
  have diverged before (the GUI shipped different quenching chemistry than the
  CLI for as long as nobody looked), so a green unit suite is not verification.
  One command covers library, GUI and backend:
  `QT_QPA_PLATFORM=offscreen <arm64>/bin/python -m pytest tests/ webui/backend/tests -q --ignore=tests/test_webui_e2e.py`
  — **do not pass `-p no:pytest-qt`**, it now removes the `qtbot` fixture. Start
  the backend and frontend so `test_webui_e2e.py` runs instead of skipping.
  **Render the GUI and the charts and look at the images yourself** — a missing
  control or an acceptor defaulting to a water molecule is invisible to
  assertions. uvicorn does not reload: restart it before re-running the browser
  suite. Baseline 2026-07-28: **404 passed, 1 skipped**, plus 16 for
  `-m notebooks`. See [`okf/workflows/testing.md`](okf/workflows/testing.md).
- **A number-moving change is verified against `tests/baselines/`, not against a
  green suite** — and every fix must be *seen to fail* before it is believed.
  [`okf/specs/programme-2026-07.md`](okf/specs/programme-2026-07.md), decision 13.
- **General fluorescence machinery belongs in imp-tricks, under `IMP.bff`.**
  QuEst is the application — project schema, PET workflow, surfaces, scanning.
  Accessible volumes, dye distributions and FRET distance/rate math are general;
  before writing any of that here, check whether `IMP.bff` already has it (it
  usually does) and prefer depending on it to adding a second copy. Direction
  and migration map: [`okf/specs/imp-tricks-dedup.md`](okf/specs/imp-tricks-dedup.md).
- **Keep the optional dependencies optional.** Qt, LabelLib and IMP are all
  imported behind `try`/`except` with fallbacks. That is what lets the CLI, the
  API and the web backend run in this environment, which has no Qt bindings and
  no `IMP.bff`. Do not make any of those imports unconditional.
- **Logic goes in the service layer, not in a surface.** The CLI, the web
  backend, the GUI and notebooks are thin wrappers over `quest/core.py` (exposed
  as `quest.api`). A feature that exists only in the web UI is a design error.
  See [`okf/architecture/surfaces.md`](okf/architecture/surfaces.md). The target
  is the ChiSurf plugin split — core / rpc / api / cli / gui, one method
  registry behind every surface — so prefer changes that move the tree toward
  [`okf/specs/plugin-integration.md`](okf/specs/plugin-integration.md) and close
  a row in [`okf/specs/assessment.md`](okf/specs/assessment.md).
- **Do not leave run artefacts in the tree.** `simulate_project` writes
  `jobs/<uuid>/` under the current directory; pass `save_outputs=False` or a
  temp `project_dir` for tests and bulk work.
- **Change policy.** QuEst is **unreleased** — nothing depends on a published
  build, so prefer the clean change over the compatible one: rename freely,
  delete unused shims, reshape file formats. This does *not* extend to silently
  changing what the science computes, to breaking ChiSurf without recording it,
  or to skipping OKF. Full rule:
  [`okf/references/release-state.md`](okf/references/release-state.md).
- **Commit in between, not at the end.** As soon as a coherent piece is made
  *and verified* — suite green, `okf_validate.py` clean — commit it, then start
  the next. Do not batch a session's work into a final commit: an uncommitted
  stretch cannot be reverted selectively, is invisible to anyone else reading
  the tree, and yields messages that describe *what* was done because the *why*
  has already faded. The concept update and the `okf/log.md` bullet ride in that
  same commit. If a piece cannot be committed because it shares a file with
  other in-flight work, say so when you reach it, not in a closing summary.
  Same rule in FPSIMP ([`okf/references/okf-format.md`](okf/references/okf-format.md)).
- **Git.** Commit locally; do not push. No commit trailers of any kind — no
  `Co-Authored-By`, no tool attribution. The working tree carries other
  in-flight work: never `git add -A` / `git commit -a`, and never run
  destructive git (`reset --hard`, `restore`, `clean -f`, `stash`, `--force`)
  that could discard changes you did not make.

## Environment and packaging

**pixi drives everything**, Python included:

| environment | holds | tasks |
|---|---|---|
| `default` | library, CLI, web backend | `test`, `test-web`, `serve` |
| `desktop` | the above plus Qt | `gui`, `test-gui` |
| `frontend` | the Node toolchain | `dev`, `build`, `typecheck` |
| `build` | rattler-build only | `build-pkg` |

The conda package is built by **rattler-build** from
`rattler-recipe/recipe.yaml`; the old `conda-recipe/meta.yaml` and its
`conda mambabuild` workflow are gone. CI (`.github/workflows/ci.yml`) runs the
library, backend, GUI, frontend and packaging jobs from those same pixi tasks.

For day-to-day work on this machine the `arm64` conda environment is still the
canonical interpreter (it has the compiled `IMP.bff`); see
[`okf/workflows/environment.md`](okf/workflows/environment.md).

## Contact

Author: Thomas-Otavio Peulen. Repository:
`github.com/Fluorescence-Tools/quest`.
