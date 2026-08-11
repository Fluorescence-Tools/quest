---
type: Architecture
title: Surfaces
description: The four user-facing surfaces and the rule that each is a thin wrapper over one service layer.
resource: quest/api.py
tags: [api, cli, webui, gui, layering]
status: stable
generated: { by: process:okf-v0.1-import, at: 2026-07-26T00:00:00Z }
stale_after: 2027-02-06
sources:
  - id: quest-api-py
    resource: ../../quest/api.py
    title: quest/api.py
    author: human:tpeulen
    last_modified: 2026-07-28
---

# One service layer, four front ends

```
        CLI            web backend        Qt GUI          notebooks
   (quest/cli.py)   (webui/backend)   (quest/gui/)      (quest.api)
        \                 |                |                 /
         \________________|________________|________________/
                                  |
                     quest.api  (the verbs)
                                  |
                    quest.rpc  (client + registry)
                                  |
                            quest/core/
                                  |
        DonorDecay / SimulateDiffusion / AV / photon kernels
```

`quest/api.py` is a **verb facade over an RPC client**, not a re-export shim:
`simulate`, `scan`, `template`, `validate`, `describe` and the data verbs all go
through `quest.rpc`, so every surface exercises one registry, one validation
path and one set of error codes ([subsystems/core-api.md](/subsystems/core-api.md)).
It still re-exports the domain names for notebooks. `quest/core/` is where the
logic lives.

> This page describes the tree as it is. The **target** is the
> core/rpc/api/cli/gui split adopted from the ChiSurf plugin contract —
> [specs/plugin-integration.md](/specs/plugin-integration.md) — with the current
> divergences itemised in [specs/assessment.md](/specs/assessment.md). QuEst is
> also embedded in ChiSurf as its `quenching_estimator` plugin, which today
> reaches only the Qt widget.

# The layering rule

A surface parses input, calls the service layer, and formats output. Simulation
logic does not live in a surface. Two consequences that have already been paid
for:

- **The validation rules live in `quest/project.py`**, and every surface
  enforces those. The backend imports `quest.api`, `quest.rpc` and
  `quest.runs`; the CLI's `_validate_project` is a thin wrapper that only
  translates `ProjectValidationError` into a `ClickException`. (Until
  2026-07-28 the backend borrowed the CLI's own `_validate_project` and
  `_template_project` -- the same guarantee, reached the wrong way round.)
- **The Qt widget owns no simulation state.**
  `TransientDecayGenerator` is a `QWidget` around a `ProjectFormModel` whose
  `model.project` is a plain project dict at all times, so the form edits the
  same document the CLI and the API consume. It used to *inherit* `DonorDecay`,
  which shared one model at the price of a widget that was also a simulation.

New behaviour therefore lands in `quest/core/` first, and the surfaces expose
it. A feature that exists only in the web UI is a design error.

# The surfaces

| Surface | Entry | Notes |
|---|---|---|
| Python API | `from quest.api import …` | Documented in [subsystems/core-api.md](/subsystems/core-api.md); safe in headless environments |
| CLI | `quest simulate/scan/template/validate/gui` | [subsystems/cli.md](/subsystems/cli.md) |
| Web UI | FastAPI + Next.js (+ Electron shell) | [subsystems/webui.md](/subsystems/webui.md) |
| Qt GUI | `cs.quest`, `quest gui` (`quest.gui:start_gui`) | Needs a Qt binding and a ChiSurf checkout to render the form — [subsystems/gui.md](/subsystems/gui.md) |

# Qt is optional, and the layout is what keeps it that way

Qt is imported in nine files and **every one of them is under `quest/gui/`**:
`app.py`, `autoform_panel.py`, `chimol_widget.py`, `dye_widget.py`,
`form_model.py`, `guided_tour.py`, `plots.py`, `quencher_widget.py`,
`theme.py`. Nothing in `quest/core/`, `quest/rpc/`, the CLI or the web backend
touches it. (This concept said "exactly three files" until 2026-08-10 — the
count grew with the viewer, the tour and the plots; the rule it was standing
for did not change, which is why a *rule* is the durable claim and a count is
not.)

`form_model.py` is on that list only through two function-local
`from qtpy import QtWidgets` calls in its file dialogs, so importing the model
still pulls in no Qt — the subprocess guardrail in `tests/test_form_model.py`
asserts exactly that, and `tests/test_service_layer.py` asserts the same for
`quest.api` / `quest.rpc` / `quest.core`.

That is a stronger guarantee than the one this concept used to describe --
`try`/`except qtpy` with dummy widget bases inside `core/av.py` and
`core/dye_diffusion.py`, which no longer import Qt at all. The rule is now
structural: **Qt lives in `quest/gui/` and nowhere else**, which is what lets
the CLI, the API and the web backend run in an interpreter without a binding.

Verified against the tree on 2026-08-10.
