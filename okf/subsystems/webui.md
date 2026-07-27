---
type: Subsystem
title: Web UI
description: The FastAPI backend, the Next.js frontend, and the Electron shell.
resource: webui/backend/app/main.py
tags: [fastapi, nextjs, electron, webui]
timestamp: '2026-07-26T00:00:00Z'
---

# Three pieces

| Piece | Path | Run |
|---|---|---|
| Backend | `webui/backend/app/main.py` | `python -m uvicorn webui.backend.app.main:app --port 8000` (needs `pip install -e ".[web]"`) |
| Frontend | `webui/frontend` (Next.js 16, React 19, Node ≥ 20.9) | `pixi run -e frontend dev` |
| Desktop | `webui/desktop` (Electron) | `npm --prefix webui/desktop run start` |

The frontend needs Node ≥ 20.9; the pinned toolchain lives in the pixi
`frontend` environment so nothing has to be installed system-wide
([workflows/environment.md](/workflows/environment.md)).

The Electron shell spawns the backend and the frontend's standalone server and
points a `BrowserWindow` at it; `QUEST_PYTHON` overrides the interpreter and
`webui/desktop/scripts/prepare-python-env.mjs` prepares a bundled one for
packaged builds.

# Backend API

```
GET  /health
GET  /api/parameter-catalog     # quest/settings/parameter_catalog.json
GET  /api/dyes                  # dye presets;  POST /api/dyes appends one
GET  /api/quenching-defaults    # REFERENCE_PET_QUENCHING + DEFAULT_QUENCHER_ATOMS
GET  /api/template              # the CLI's own _template_project
POST /api/validate              # the CLI's own _validate_project
POST /api/structure/metadata    # chains/residues parsed from an uploaded structure
POST /api/simulate
POST /api/scan
GET  /api/jobs, /api/jobs/{id}, /api/jobs/{id}/download
```

Importing `quest.cli`'s `_template_project` / `_validate_project` is deliberate:
one definition of "a valid project", shared with the terminal.

# Things that are easy to get wrong

- **`SCAN_WORKERS` is small on purpose** — `max(1, min(4, cpus-1))`. LabelLib's
  AV kernel holds the GIL, so extra workers add contention, not throughput. The
  pool exists to keep the simulation off the event loop so the server stays
  responsive and notices client disconnects during a long scan.
- **State lives on disk, not in the process.** Uploads and job directories go
  under `QUEST_WEBUI_DATA_DIR` (default: a `quest-webui` folder in the system
  temp dir). The server is restartable and holds no session state.
- **CORS origins** come from `QUEST_WEBUI_CORS_ORIGINS` (default: localhost
  :3000/:3001). The frontend targets `QUEST_BACKEND_URL`.
- **Dye defaults are filled in server-side** (`_with_dye_stickiness_defaults`,
  `_with_dye_library_defaults`) so a project posted from the browser is complete
  before it reaches `quest.api`.

# Frontend

App-router pages: `/` , `/simulation`, `/labeling`, `/scan`, `/advanced`, with
shared state in `context/AppContext.tsx`.

The labeling page carries the **FRET acceptor** panel: an enable toggle, the
acceptor chain/residue/atom selectors, `R0 [Å]` and the acceptor diffusion
coefficient. Enabling it seeds the acceptor on the *last labelable* residue of
the current chain — `lastLabelableResidue` skips the waters and ions structures
end with, which a dye cannot be tethered to — so switching FRET on yields a
project that validates. Notable components: `Plots.tsx`,
`ProteinViewer.tsx` (NGL, vendored at `public/vendor/ngl.js` — no CDN),
`QuenchingEditor.tsx` (the per-residue interaction table), and `Field.tsx` +
`HelpBadge.tsx`, which render labels and help from
`lib/parameterCatalog.ts` — i.e. from
[the parameter catalog](/architecture/project-schema.md), not from hard-coded
strings.

# Tests

`webui/backend/tests/test_backend_api.py` covers the API directly.
`tests/test_webui_e2e.py` drives a real browser and **skips unless both servers
are already running** — see [workflows/testing.md](/workflows/testing.md).
