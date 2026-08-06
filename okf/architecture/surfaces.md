---
type: Architecture
title: Surfaces
description: The four user-facing surfaces and the rule that each is a thin wrapper over one service layer.
resource: quest/api.py
tags: [api, cli, webui, gui, layering]
status: stable
generated: { by: process:okf-v0.1-import, at: 2026-07-26T00:00:00Z }
stale_after: 2026-08-06
sources:
  - id: quest-api-py
    resource: ../../quest/api.py
    title: quest/api.py
    author: human:tpeulen
    last_modified: 2026-07-28
---

> **Marked stale on 2026-08-06.** The file paths in this concept were
> corrected against the tree, but its *descriptions* were not re-read --
> `quest/rpc/`, `quest/hosts/` and `quest/tasks.py` post-date it. Treat the mechanism it describes as unverified until someone
> reads it against the code. See
> [references/known-issues.md](../references/known-issues.md).
# One service layer, four front ends

```
        CLI            web backend         Qt GUI           notebooks
   (quest/cli.py)   (webui/backend)   (quest_gui.py)      (quest.api)
        \                 |                 |                  /
         \________________|_________________|_________________/
                                  |
                       quest.api  ->  quest.core
                                  |
        DonorDecay / SimulateDiffusion / AV / photon kernels
```

`quest/api.py` is a verb facade over the RPC client (plus re-exports from `quest/core/` and
`fps_json`) and is the **stable import surface**: downstream code and the web
backend import `quest.api`, not the internals. `quest/core/` is where the
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

- The web backend (`webui/backend/app/main.py`) calls `quest.api` and even
  reuses the CLI's own `_validate_project` and `_template_project`, so the
  browser and the terminal cannot disagree about what a valid project is.
- The Qt GUI class `TransientDecayGenerator` *inherits* `DonorDecay` rather than
  wrapping a copy of it, so the widget and the headless run share one model.

New behaviour therefore lands in `quest/core/` first, and the surfaces expose
it. A feature that exists only in the web UI is a design error.

# The surfaces

| Surface | Entry | Notes |
|---|---|---|
| Python API | `from quest.api import …` | Documented in [subsystems/core-api.md](/subsystems/core-api.md); safe in headless environments |
| CLI | `quest simulate/scan/template/validate/gui` | [subsystems/cli.md](/subsystems/cli.md) |
| Web UI | FastAPI + Next.js (+ Electron shell) | [subsystems/webui.md](/subsystems/webui.md) |
| Qt GUI | `cs.quest`, `quest gui` | Legacy; needs Qt bindings that are **not** in the current env — [subsystems/gui.md](/subsystems/gui.md) |

# Qt is optional, and the code keeps it that way

`quest/core/av.py` and `dye_diffusion.py` import `qtpy` inside a
`try`/`except` and fall back to dummy widget bases when no Qt bindings exist.
That is what lets the CLI, the API and the web backend run in an environment
without PyQt — which is exactly the current one. Do not turn those imports into
hard dependencies.
