---
type: Specification
title: Plugin integration — target
description: The core / rpc / api / cli / gui split QuEst adopts from the ChiSurf plugin contract, so one implementation serves every surface.
resource: quest/
tags: [target, architecture, plugin, rpc, layering, chisurf]
timestamp: '2026-07-26T00:00:00Z'
---

> The target. Current shape: [architecture/surfaces.md](/architecture/surfaces.md).
> Current-state gaps: [assessment.md](assessment.md).

# Why

Beyond keeping the surfaces in step, this split is the **enabler for
[OBJ-02](objectives.md)**: a host application can only drive QuEst as a fittable
model — varying parameters, scoring the decay against a measurement — if QuEst
exposes named operations over plain data. Today it exposes a window.

QuEst has four surfaces today and is *also* embedded in the sibling ChiSurf
application as its `quenching_estimator` plugin. That plugin is currently a bare
Qt wrapper: it imports `quest.lib.tools.dye_diffusion.TransientDecayGenerator`
and puts it in a `QMainWindow`. Nothing of QuEst's headless capability —
simulate, scan, validate, the parameter catalog, the dye library — is reachable
from ChiSurf's RPC layer, its CLI, or its automation. The plugin also ships no
`manifest.json`, so it loads through ChiSurf's legacy AST-based discovery path.

ChiSurf's own plugins solve this with a layered split that QuEst should adopt.
The payoff is not tidiness: it is that **one implementation serves the QuEst
CLI, the QuEst web UI, the QuEst desktop GUI, and the ChiSurf plugin** — with
the same method names, the same shapes, and the same error contract.

# The split

The reference layout, as used by ChiSurf plugins (e.g.
`chisurf/plugins/fcs/fcs_calculator/`):

```text
manifest.json          identity, entrypoints (gui / cli / services), rpc_methods
core/                  Qt-free, host-free domain code — the algorithms
backend/services.py    RPC handlers: register_services(dispatcher)
backend/state.py       session state owned by the service layer
cli/main.py            CLI entry point -> calls core (or the facade)
gui/client.py          the client wrapper — the ONLY way the GUI reaches the backend
<plugin>.view.json     declarative form/panel description rendered by the host
test/
README.md
```

The layering rule, in one line each:

- **core** knows nothing about Qt, HTTP, ChiSurf, or the CLI. Pure functions and
  domain objects over plain data.
- **backend** exposes core as named RPC methods returning a single shape:
  `{"ok": True, "result": …}` or `{"ok": False, "error": …}`. It is the only
  writer of session state.
- **api / facade** is the one door presentation calls. It resolves in-process or
  remote *internally* — callers never branch on "am I local".
- **cli** parses arguments and formats output. No domain logic.
- **gui** renders and calls `client.call("quest.<verb>", params)`. It never
  imports a domain object.

Everything crossing the boundary is plain serializable data with stable
identifiers; live domain objects do not cross.

# QuEst's method surface

The RPC methods fall out of what the FastAPI backend already exposes — which is
the evidence that this surface is the real one, it is just currently spelled
only as HTTP routes:

| RPC method | Today's spelling | Long-running |
|---|---|---|
| `quest.simulate` | `POST /api/simulate` | yes, cancelable |
| `quest.scan` | `POST /api/scan` | yes, cancelable |
| `quest.validate` | `POST /api/validate` → `cli._validate_project` | no |
| `quest.template` | `GET /api/template` → `cli._template_project` | no |
| `quest.parameter_catalog` | `GET /api/parameter-catalog` | no |
| `quest.dyes.list` / `quest.dyes.save` | `GET`/`POST /api/dyes` | no |
| `quest.quenching_defaults` | `GET /api/quenching-defaults` | no |
| `quest.structure.metadata` | `POST /api/structure/metadata` | no |
| `quest.jobs.list` / `quest.jobs.get` | `GET /api/jobs`, `/api/jobs/{id}` | no |

Each carries documented params/result shapes in the manifest. The web backend
then becomes a **transport over the registry** rather than a parallel
implementation of it — one place to add a method, four places it appears.

`quest.simulate` and `quest.scan` are the long-running pair and need the uniform
start / observe / cancel mechanism, not three ad-hoc ones (a thread pool in the
web backend, a blocking call in the CLI, a `QProgressDialog` in the GUI).

# Target layout for this repository

```text
quest/
  manifest.json            id "quest", entrypoints, rpc_methods, state schema
  core/                    the domain, Qt-free
    simulation.py          simulate_project & the pipeline (today: core.py)
    analysis.py            residue attribution, autocorrelation
    project.py             the project schema: template, validate, normalise
    grids.py               MRC + density writers
  backend/services.py      register_services(dispatcher) -> the table above
  api.py                   the facade: verb-oriented, plain data in and out
  cli/                     Click commands -> facade
  gui/                     widget + client.py
  quest.view.json          declarative parameter form, generated from the catalog
```

Two QuEst-specific notes:

- **The parameter catalog is already half of a view spec.**
  `quest/settings/parameter_catalog.json` carries label, category, type, unit and
  description per dotted parameter id, and both the web UI and the docs render
  from it. Generating `quest.view.json` from the catalog — rather than
  hand-writing a second description of the same fields — keeps the Qt form, the
  browser form, and the docs from drifting.
- **The project JSON is already the boundary contract.**
  [architecture/project-schema.md](/architecture/project-schema.md) is plain
  serializable data with dotted addressing. It crosses the wire unchanged; there
  is no object to marshal. That is why this split is cheap for QuEst.

# The dependency arrow, and where it points

Two different directions, decided deliberately:

- **Fluorescence machinery: QuEst → imp-tricks.** Accessible volumes, distance
  metrics, dye models. QuEst must never be imported by imp-tricks
  ([imp-tricks-dedup.md](imp-tricks-dedup.md)).
- **Presentation: QuEst → ChiSurf.** The host's **AutoForm** renders QuEst's
  control panels, so QuEst stops carrying `.ui` files and hand-built Qt forms.

The second is the reversal of the earlier "QuEst never imports the host" rule,
and it is what makes the UI deduplication possible at all: without it, a
declarative form means QuEst reimplementing a form renderer.

**It is optional, and that is what keeps QuEst standalone.** ChiSurf is reached
through `quest.chisurf_host.enable_chisurf()` — a source checkout, resolved from
`CHISURF_ROOT`, never fatal, and never overriding a ChiSurf already imported by a
host that loaded QuEst as a plugin. Nothing outside `quest.gui` imports it.

## AutoForm renders the manifest QuEst already ships

`AutoForm.from_rpc_method(entry)` takes an `rpc_methods` entry from
`quest/manifest.json` and produces a bound Qt form: labels, types, defaults,
JSON-Schema `description`s as tooltips, and the entered values read back with
`form.model.params()`. QuEst already declares all eleven methods with full
parameter schemas, so **the declaration a host reads to learn what QuEst offers
is also the UI description** — one artefact, not two.

Demonstrated 2026-07-27 in `arm64` with the ChiSurf checkout on the path: every
one of the eleven declarations renders, `quest.simulate` showing
project / pdb_path / nbins / tac_range / project_dir plus a `save_outputs`
checkbox, with no QuEst-side UI code. Pinned by
`tests/test_chisurf_autoform.py` (14 cases), which skips where Qt or the
checkout is absent.

## The `.ui` files are gone (2026-07-27)

The migration ran in the three stages planned here, and finished:

1. Render the parameter groups from the manifest — `quest/gui/autoform_panel.py`,
   still there for rendering one RPC method's parameters.
2. Author the rest as a view scheme — `quest/gui/quest.view.json`, **generated**
   from the parameter catalog by `quest/gui/generate_view_spec.py`. AutoForm's
   own section types cover everything QuEst needed: `value kind=file` for the
   structure picker, an editable `table` for the quencher chemistry, the
   `chimol` custom section for the 3D view, and `plot` sections (chiplot) for
   the decay, trajectory and autocorrelation.
3. Delete the `.ui` files and the hand-built form. Done — 1,899 lines of widget
   became ~130.

There is no in-tree fallback renderer: where ChiSurf is absent, the desktop
surface says so and the CLI, API and web backend carry on unaffected. That is a
deliberate consequence of the reversed presentation arrow — one renderer, not
two that can disagree. See [subsystems/gui.md](/subsystems/gui.md).

## The remaining hand-built pieces, and where each one goes

Three instructions arrived for this migration and are recorded here rather than
started, because each is a substantial change and half of one is worse than
none:

1. **Delete the `.ui` files.** `quest/gui/dye_diffusion2.ui`, `pdb_widget.ui`
   and `ui/*.ui` describe the same parameters the manifest already declares.
   `AutoForm.from_rpc_method` covers the parameter groups today (proven); the
   structure picker, quencher table and plots need a `quest.view.json` scheme
   first — AutoForm has section types for tables, and `path_list`/`state_table`
   look like the right primitives for the picker and the quencher grid.
2. **Plot through `chisurf/gui/chiplot/`.** The widget hand-builds pyqtgraph
   plots for the decay, the trajectory and the autocorrelation; ChiSurf's own
   tools use chiplot, so QuEst carrying a third plotting style is the same
   duplication as the molecule viewer was. Tracked as `LAY-11`.
3. **Check ChiSurf tests run against the current `src`.** QuEst reaches ChiSurf
   through a checkout (`CHISURF_ROOT`), and the `chisurf` conda environment's
   *installed* copy predates AutoForm — so a ChiSurf test run there may be
   exercising a stale package, exactly the trap that hid QuEst's own broken IMP
   backend. Worth verifying on the ChiSurf side before either project relies on
   the other's test results.

# Rules

1. `core/` MUST NOT import Qt, FastAPI, Click, or ChiSurf.
2. Presentation (CLI, web, GUI, plugin) MUST reach the domain through the facade
   or the RPC client — never by importing `quest.lib.*` internals, and never by
   importing another surface's private helpers.
3. Every operation is reachable under exactly one name. A second spelling is a
   second contract to keep in step.
4. Local and remote results for the same method MUST have the same shape, and
   failure MUST be distinguishable from success by the contract, not by sniffing
   fields.
5. The plugin declares itself in a `manifest.json` that validates against the
   host's schema — identity, entry points, RPC methods with their shapes.
6. Long-running work uses one mechanism to start, observe and cancel it.
7. The GUI may own its own windows and widgets; it MUST NOT own domain objects.

# Migration order

Each step is independently valuable and leaves the tree working:

1. **Move project logic out of the CLI.** `_validate_project` and
   `_template_project` become `quest.core.project.validate/template`; the CLI and
   the web backend both call that. Removes the web backend's import of a private
   CLI function (`LAY-01`).
2. **Add `backend/services.py`** with the method table above, wrapping the
   facade. The FastAPI routes become thin adapters onto it (`LAY-03`).
3. **Add `manifest.json`** and point its `gui`/`cli`/`services` entry points at
   the real ones, so ChiSurf discovers QuEst the same way it discovers every
   other plugin (`LAY-05`).
4. **Give the GUI a client.** `TransientDecayGenerator` stops inheriting
   `DonorDecay` and holds a client instead (`LAY-02`) — the largest step, and the
   one that finally makes the Qt widget testable headlessly.
5. **Generate `quest.view.json` from the parameter catalog** (`LAY-06`).

Progress is tracked in [assessment.md](assessment.md).
