---
type: Playbook
title: Environment
description: Which interpreter runs QuEst, which dependencies are optional, and the pixi Node toolchain for the web UI.
resource: pyproject.toml
tags: [environment, conda, pixi, dependencies]
status: stable
generated: { by: process:okf-v0.1-import, at: 2026-07-26T00:00:00Z }
stale_after: 2027-01-26
sources:
  - id: pyproject-toml
    resource: ../../pyproject.toml
    title: pyproject.toml
    author: human:tpeulen
    last_modified: 2026-07-27
---

# pixi drives the whole project

`pixi.toml` declares four environments — `default` (library, CLI, web backend),
`desktop` (adds Qt), `frontend` (the Node toolchain), and `build`
(rattler-build alone, so packaging does not solve the science stack):

```bash
pixi run test        # library
pixi run test-web    # FastAPI backend
pixi run -e desktop test-gui
pixi run -e frontend typecheck
pixi run -e build build-pkg      # conda package via rattler-build
```

The conda package comes from `rattler-recipe/recipe.yaml`; `conda-recipe/` and
its `conda mambabuild` workflow are gone, as is the CI that filtered on
`modules/quest/**` and therefore never ran.

## All four environments solve again (2026-08-10)

`default` and `desktop` were unsolvable — and their two CI jobs permanently red
— because the `py` feature required packages no public channel serves. Both
causes are now handled, and they are different:

- **`labellib` comes from the `dev` label of the owner's channel.** Not the
  channel root: `https://conda.anaconda.org/tpeulen`'s `main` label stops at the
  2020 builds, which are py37-py39 and pin `numpy <2`, so the solve failed with
  *"labellib 1.0.0 would require numpy >=1.22.4,<2.0a0"* against this project's
  `numpy >=2.0`. The `dev` label carries py310-py313 builds against numpy ≥ 2.2.
  It has no `osx-64` or `linux-aarch64` build, so the `py` feature restricts
  itself to `osx-arm64`, `linux-64` and `win-64` — which is exactly the CI
  matrix (ubuntu / macos-arm / windows).
- **`imp-tricks` is a feature no environment uses.** Publication was ruled out
  (2026-07-28, it stays local), so anything that required it would make
  `pixi lock` fail outright. The feature is declared, unreferenced, one word in
  `[environments]` away from being switched on the day it is published.

The consequence is stated rather than glossed: **a pixi environment cannot
compute an accessible volume**, because both AV backends dispatch through
`IMP.bff.av` ([subsystems/accessible-volume.md](/subsystems/accessible-volume.md)).
141 tests skip there with that reason and ~550 still run — the contract, the
project schema, the RPC envelopes, i18n, the view spec, the form model. That is
CI signal where there was a red light nobody could act on.

**For day-to-day work on this machine the `arm64` conda environment is
canonical** — it carries the compiled `IMP.bff`, which a pixi-solved `imp` may
not. The two coexist: pixi for reproducible CI and packaging, `arm64` for
development against the real IMP build.

# `arm64` is the *only* environment — one interpreter, all four surfaces

**Use `/Users/tpeulen/mambaforge/envs/arm64/bin/python`, for everything.**

This changed on **2026-07-28** ([specs/programme-2026-07.md](/specs/programme-2026-07.md),
decision 4). Until then the project ran on two interpreters and the rule was
*"both must stay green"*: `arm64` for the library, the GUI and the science, and
the mambaforge base for the web layer, because fastapi and playwright were only
there. The owner ruled base out of scope, so the web stack moved into `arm64`:

    mamba install -n arm64 -c conda-forge fastapi uvicorn python-multipart pytest-asyncio httpx
    <arm64>/bin/python -m pip install playwright pytest-playwright
    <arm64>/bin/python -m playwright install chromium

**`playwright` must come from pip, not conda-forge.** conda-forge's `playwright`
is the *Node* package; the Python bindings are not in it, and installing it
leaves `import playwright` failing.

| | `arm64` (the environment) | ~~mambaforge base~~ (retired) |
|---|---|---|
| Python / numpy | 3.12 / **2.4.6** | 3.10.14 / 1.26.4 |
| IMP | **2.24.0, compiled `IMP.bff`** | 2.17.0, no `IMP.bff` |
| Qt (PyQt5), pytest-qt | **yes** | no |
| LabelLib, numba | yes | yes |
| fastapi, uvicorn, playwright | **yes** (since 2026-07-28) | yes |

**Base was not measured to be broken.** On 2026-07-28 its web backend suite gave
16 passed and `import quest.core` succeeded. Retiring it is an owner ruling, and
is recorded as one rather than dressed up as a measurement.

Consequences of the consolidation:

- **`-p no:pytest-qt` is no longer used, and passing it is now harmful.** The
  flag existed because base had no Qt binding and the plugin aborted collection.
  `arm64` *has* pytest-qt, and the flag removes the `qtbot` fixture.
- **The optional-dependency rule lost its enforcement and gained a test.** Qt,
  LabelLib and `IMP.bff` stay behind `try`/`except` — a Windows box without a
  compiled `IMP.bff` still needs the LabelLib path — but nothing exercises those
  branches by accident any more, because `arm64` has all three.
  `tests/test_optional_dependencies.py` simulates each absence instead: the
  backend resolution with either flag forced false, and `import quest.core` in a
  subprocess with `PyQt5`, `qtpy`, `LabelLib` or `IMP` blocked at import.
  Without it, a fallback path rots exactly the way `imp_av.py` did.
- `arm64` imports `quest` straight from this working tree, so there is no
  install step.

One interpreter, one command, all four surfaces ([testing.md](testing.md)).

Install for development:

```bash
pip install -e .            # core
pip install -e ".[web]"     # + fastapi / uvicorn / python-multipart
```

Optional dependencies are optional *by design* — every import of Qt, LabelLib
and IMP is guarded ([subsystems/gui.md](/subsystems/gui.md),
[subsystems/accessible-volume.md](/subsystems/accessible-volume.md)). Keep it
that way. Since `arm64` now has all three, the rule is held by
`tests/test_optional_dependencies.py` rather than by an interpreter that lacks
them — if you add a guarded import, add its absence case there too.

# QuEst resolves to *this* checkout, from anywhere

`arm64` imports QuEst from `~/dev/quest` regardless of the working directory: a
`quest-src.pth` in site-packages adds the checkout to the path. It previously
had no QuEst at all and worked only because the shell happened to sit in the
repo.

*(Historical, for anyone who finds an old base install: base's editable install
mapped `quest` to `/Users/tpeulen/dev/chisurf/modules/quest/quest` — a separate,
older checkout — through a meta-path finder that outranks the current directory,
so anything run from outside the repo silently exercised that copy. It was
re-pointed here before base was retired.)

`chisurf/modules/quest` is a **git submodule** with its own `.git`, so it was
left alone rather than replaced by a symlink — it may hold unpushed work. It is
now simply not what gets imported.

Linking the live tree immediately surfaced real breakage that the stale copy had
been hiding: ChiSurf's `quenching_estimator` plugin still imported
`quest.lib.tools.dye_diffusion`, which no longer exists. It now imports
`quest.gui`, and its tests pass again — the kind of thing that stays invisible
while two copies of a package are in play.

# Reaching imp-tricks from a checkout

General fluorescence machinery lives in imp-tricks
([specs/imp-tricks-dedup.md](/specs/imp-tricks-dedup.md)). **Do not install it** —
it is under active development, so the checkout is what has to work; an
installed copy goes stale immediately. (A `pip install --no-deps`, which is what
ChiSurf's installer does for releases, was tried here and reverted.)

Instead `quest.core.av` calls `quest.hosts.imp_tricks.enable_imp_tricks()` **on
import**, so touching the domain layer makes the checkout importable:

```python
import quest.core                       # bridges ~/dev/imp-tricks/src
from IMP.bff.av import compute_av       # from the live source tree
from IMP.cgmol.sterics import asa
```

**`import quest` alone is not enough**, and this section said it was, naming a
module (`quest.lib.imp_av`) that `DUP-01` deleted. Verified 2026-07-28:
`import quest` then `import IMP.cgmol.sterics` raises `ModuleNotFoundError`;
`quest.core`, `quest.core.av` and `quest.api` all splice. Calling
`enable_imp_tricks()` yourself always works.

That error cost real time here: a probe written as `import quest.hosts.imp_tricks`
(the module, without *calling* it) reported `IMP.bff.av` and `IMP.cgmol` as
missing, and "fix the `IMP.cgmol` import" was carried into a work plan as a
defect upstream. There was no defect. **When a checkout-only import fails, check
that the bridge was actually invoked before concluding anything about the other
repository.**

Point it elsewhere with `IMP_TRICKS_SRC`; with no checkout it does nothing.

Adding `src` to `PYTHONPATH` is **not** sufficient in `arm64`: `IMP` and
`IMP.bff` are compiled *regular* packages there and shadow the checkout's
namespace directories entirely. The helper extends the imported package's
`__path__` instead, so the checkout's `IMP.bff.av` sits alongside the compiled
`IMP.bff.AV` rather than replacing it. It is idempotent and returns `[]` rather
than raising when no checkout exists.

# The Node toolchain (web UI)

```bash
pixi install -e frontend      # materialise Node >= 22.11, <25
pixi run -e frontend dev      # Next.js dev server on :3000 (backend on :8000 first)
pixi run -e frontend build    # production build (+ standalone assets)
pixi run -e frontend start
pixi run -e frontend typecheck
pixi run -e frontend node     # print the env's Node version
```

Next.js 16 refuses to start on anything older than Node 20.9, which is the whole
reason the pinned toolchain exists. `pixi.toml` and `pixi.lock` are committed, so
the environment is reproducible. Electron tasks: `desktop-install`, `desktop`,
`desktop-dist`.

# Environment variables

| Variable | Effect |
|---|---|
| `QUEST_PDB_CACHE_DIR` | Where downloaded RCSB structures are cached |
| `QUEST_WEBUI_DATA_DIR` | Web-UI uploads and job directories |
| `QUEST_WEBUI_CORS_ORIGINS` | Comma-separated allowed origins |
| `QUEST_BACKEND_URL` | Backend target for the frontend |
| `QUEST_PYTHON` | Interpreter the Electron shell spawns |
| `NUMBA_NUM_THREADS` | Read by numba **at import time** — see `_preserved_numba_environment` |
| `QUEST_AV_BACKEND` | `labellib` (default), `imp_bff`, or `auto` — see [subsystems/accessible-volume.md](/subsystems/accessible-volume.md) |
| `IMP_TRICKS_SRC` | Where `enable_imp_tricks()` looks for the checkout |
