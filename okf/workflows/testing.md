---
type: Playbook
title: Testing
description: How to run the suite (including the -p no:pytest-qt you will need), what is covered, and what skips.
resource: tests/
tags: [testing, pytest, playwright]
timestamp: '2026-07-26T00:00:00Z'
---

# Every change is tested on all three surfaces

A change is not verified until the **library, the web UI and the Qt GUI** have
all been exercised. They have diverged before — the GUI shipped different
quenching chemistry than the CLI for as long as nobody looked — so a green unit
suite proves less than it appears to.

**No single interpreter runs everything.** `arm64` is canonical and has Qt and
a compiled `IMP.bff`; the base environment has the web stack
([environment.md](environment.md)). Both must be green.

```bash
ARM64=/Users/tpeulen/mambaforge/envs/arm64/bin/python

# 1. canonical: library + Qt GUI + the IMP-capable code paths
QT_QPA_PLATFORM=offscreen $ARM64 -m pytest tests/ -q \
    --ignore=tests/test_webui_e2e.py --ignore=tests/test_pdb_id_support.py

# 2. base: library + web backend (no Qt here, hence the flag)
python -m pytest tests/ webui/backend/tests -q -p no:pytest-qt

# 3. web UI: start both servers, then run the browser suite in its OWN process
python -m uvicorn webui.backend.app.main:app --port 8000 &
pixi run -e frontend dev &
pixi run test-e2e          # == pytest tests/test_webui_e2e.py -m browser
```

**The browser suite is deselected by default** (`addopts = "-m 'not browser'"`)
and must run in its own process. Chromium's renderer dies with a bare
"Page crashed" once one pytest process has *also* imported IMP, numba and
FastAPI and executed the notebooks — the suite passes alone and fails at the end
of a combined run. Deselecting says so; silently skipping would not. A combined
run reports `22 deselected`, which is the signal that the browser layer still
needs its own command.

Baseline on 2026-07-27: **170 passed, 1 skipped** for (1) — GUI included — and
**205 passed, 2 skipped** for (2)+(3) with the servers up.

Running only the base environment hides real breakage: the IMP AV backend is
unreachable there, and 34 tests failed the first time the suite was run under
`arm64`.

**uvicorn does not reload.** After changing backend code, restart it before
re-running the browser suite, or you will be testing the old process.

# Running it

```bash
python -m pytest tests/ -q -p no:pytest-qt
```

**The `-p no:pytest-qt` is not optional in this environment.** No Qt binding is
installed, and the plugin aborts the whole run at collection with
"pytest-qt requires either PySide6, PyQt5 or PyQt6 installed" before a single
test executes. `-p no:qt` does *not* work — the plugin's entry-point name is
`pytest-qt`.

Baseline on 2026-07-26: **116 passed, 17 skipped** in ~20 s.

Backend API tests live separately:

```bash
python -m pytest webui/backend/tests -q -p no:pytest-qt
```

# What the suites cover

| File | Covers |
|---|---|
| `test_core_api.py` | The service layer: project loading, `simulate_project`, result fields |
| `test_api.py` | The `quest.api` re-export surface |
| `test_cli.py`, `test_cli_smoke.py` | CLI commands, overrides, grid sweeps |
| `test_aa_specific_quenching.py` | Per-residue PET quenching, contributions split by residue type |
| `test_scan_reporting.py` | What a scan reports: quencher attribution and failure diagnostics |
| `test_grid_kernels.py` | Equivalence and behaviour of the AV grid kernels |
| `test_simulation_performance.py` | The scan fast path (`save_outputs=False`) and seeded reproducibility |
| `test_coarse_grained.py` | The coarse-grained structure representation |
| `test_fps_json.py` | fps.json labeling files |
| `test_pdb_id_support.py` | Bare PDB IDs accepted wherever a path is |
| `test_use_cases.py`, `test_mc4r.py` | End-to-end scientific use cases |
| `test_notebooks.py` | Notebooks are readable/valid |
| `test_webui_e2e.py` | Playwright browser tests against a running UI, including the FRET panel and a full FRET run |
| `test_gui_widget.py` | The Qt widget headlessly: chemistry parity with the domain, the R0 control, that it paints |
| `test_service_layer.py` | The RPC envelope and registry, the scan record shape, and the layering guardrails |
| `test_fret_units.py` | R0 is a Förster radius in Ångström, not nanometres |

# Testing the Qt GUI

The default interpreter has no Qt binding, but two conda environments on this
machine do — `arm64` (Python 3.12) and `chisurf` (3.10). Both import `quest`
straight from this working tree and carry LabelLib, numba and pyqtgraph, so no
install step is needed:

```bash
QT_QPA_PLATFORM=offscreen /Users/tpeulen/mambaforge/envs/arm64/bin/python \
    -m pytest tests/test_gui_widget.py -q
```

`tests/test_gui_widget.py` skips itself where Qt is absent, so it is safe to
include in any run.

**Render it and look at it.** Construction tests only prove the widget did not
crash; clipped labels, a control that is missing entirely, and a table showing
the wrong chemistry are invisible to assertions and obvious in an image:

```python
w = TransientDecayGenerator(tempfile.gettempdir())
w.resize(1400, 900); w.show(); app.processEvents()
w.grab().save("gui.png")
```

Then **read the PNG yourself** rather than handing it to the user. That is how
the missing R0 control and the water-molecule acceptor default were found. Note
that `QOpenGLWidget` is unsupported offscreen, so the 3D tab will not render
under this platform plugin — that warning is expected, not a failure.

The same rule applies to the web UI: Playwright can screenshot any element
(`page.locator("svg.chart").screenshot(...)`), and a chart that plots the right
numbers can still draw them wrong.

# What skips, and why

- **`test_webui_e2e.py`** — "QuEst web UI is not running". Start the backend on
  :8000 and the frontend on :3000 first ([environment.md](environment.md)), then
  rerun; otherwise these prove nothing.
- **`test_gui_widget.py`** — no Qt binding in the default interpreter. Run it in
  the `arm64` environment as above.
- **`test_cli.py::…1anf…`** — an opt-in residence diagnostic that is slow and may
  need the PDB cache or network.

A run whose skips include the browser or GUI files means those layers were
**not** exercised. Say so when reporting, rather than "all tests pass".

# Conventions

- Structures for tests live in `tests/data/` and `tests/148l.pdb` (T4 lysozyme,
  the standard small test protein here).
- Anything touching a numba kernel gets a case in `test_grid_kernels.py` or
  `test_simulation_performance.py` — those are where kernel equivalence and
  seeded reproducibility are pinned.
- Simulations in tests should pass `save_outputs=False` unless the artefacts are
  the thing under test; otherwise every test writes a job directory into the
  working tree.

# A warm numba cache hides compile-time failures

`@njit(cache=True)` kernels only re-read numba's configuration when they
actually compile. A suite that has run once never recompiles, so a class of
failure — numba refusing a changed `NUMBA_NUM_THREADS`, a kernel that no longer
type-infers — is invisible until the cache is cold. It cost one confused round
of "flaky, passes on retry" before the pattern was clear.

Clear it before trusting a green run after touching a jitted module:

    find quest -name '__pycache__' -type d \
      -exec sh -c 'rm -f "$1"/*.nbi "$1"/*.nbc' _ {} \;

`tests/conftest.py` pins `NUMBA_NUM_THREADS` and `NUMBA_THREADING_LAYER` for
this reason; see [okf/log.md](../log.md), 2026-07-27.
