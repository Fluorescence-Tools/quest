---
type: Playbook
title: Environment
description: Which interpreter runs QuEst, which dependencies are optional, and the pixi Node toolchain for the web UI.
resource: pyproject.toml
tags: [environment, conda, pixi, dependencies]
timestamp: '2026-07-26T00:00:00Z'
---

# pixi drives the whole project

`pixi.toml` declares five environments — `default` (library, CLI, web backend),
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

**For day-to-day work on this machine the `arm64` conda environment is still
canonical** — it carries the compiled `IMP.bff`, which a pixi-solved `imp` may
not. The two coexist: pixi for reproducible CI and packaging, `arm64` for
development against the real IMP build.

# `arm64` is the canonical environment

**Use `/Users/tpeulen/mambaforge/envs/arm64/bin/python`.** It is the only
environment with the full stack: CPython 3.12, numpy 2.4.6 (so
`pyproject.toml`'s `numpy>=2.0` is right for it), **IMP 2.24 with a compiled
`IMP.bff`**, PyQt5, LabelLib and numba. Both other candidates are partial.

| | `arm64` (canonical) | mambaforge base | `chisurf` |
|---|---|---|---|
| Python / numpy | 3.12 / 2.4.6 | 3.10.14 / 1.26.4 | 3.10.13 |
| compiled `IMP.bff` | **yes** | no | not checked |
| Qt (PyQt5) | **yes** | no | yes |
| LabelLib, numba | yes | yes | yes |
| fastapi / uvicorn / playwright | **no** | yes | not checked |

Consequences worth knowing before choosing an interpreter:

- The **web** suites (`tests/test_webui_e2e.py`, `tests/test_pdb_id_support.py`,
  `webui/backend/tests/`) need fastapi/playwright and therefore run in the
  **base** environment today. Installing `".[web]"` plus playwright into `arm64`
  would consolidate that.
- The **GUI** suite needs Qt and runs in `arm64` (or `chisurf`).
- `arm64` is the only place the **IMP.bff AV backend** is even reachable —
  which is how it was discovered to be broken; see
  [references/known-issues.md](/references/known-issues.md).
- All three import `quest` straight from this working tree, so no install step
  is needed to switch between them.

Both interpreters must stay green: `arm64` for the library and GUI, base for the
library and the web layer ([testing.md](testing.md)).

Install for development:

```bash
pip install -e .            # core
pip install -e ".[web]"     # + fastapi / uvicorn / python-multipart
```

Optional dependencies are optional *by design* — every import of Qt, LabelLib
and IMP is guarded ([subsystems/gui.md](/subsystems/gui.md),
[subsystems/accessible-volume.md](/subsystems/accessible-volume.md)). Keep it
that way: the base environment has no Qt and no `IMP.bff`, and everything except
the GUI must still work there.

# QuEst resolves to *this* checkout, from anywhere

Both interpreters now import QuEst from `~/dev/quest` regardless of the working
directory:

- **base** — the editable install was re-pointed here. It previously mapped
  `quest` to `/Users/tpeulen/dev/chisurf/modules/quest/quest`, a **separate,
  older checkout**, through a meta-path finder that outranks the current
  directory. Anything run from outside the repo silently exercised that copy.
- **arm64** — a `quest-src.pth` in site-packages adds the checkout to the path.
  It previously had no QuEst at all and worked only because the shell happened
  to sit in the repo.

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

Instead `quest.lib.imp_av` calls `quest.imp_tricks.enable_imp_tricks()` **on
import**, so simply using QuEst makes the checkout importable:

```python
import quest                            # bridges ~/dev/imp-tricks/src
from IMP.bff.av import compute_av       # from the live source tree
from IMP.cgmol.statpot import _kernels
```

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
