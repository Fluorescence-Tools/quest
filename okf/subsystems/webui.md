---
type: Subsystem
title: Web UI
description: The FastAPI backend, the Next.js frontend, and the Electron shell.
resource: webui/backend/app/main.py
tags: [fastapi, nextjs, electron, webui]
status: stable
generated: { by: process:okf-v0.1-import, at: 2026-07-26T00:00:00Z }
stale_after: 2027-01-26
sources:
  - id: webui-backend-app-main-py
    resource: ../../webui/backend/app/main.py
    title: webui/backend/app/main.py
    author: human:tpeulen
    last_modified: 2026-07-28
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

# Design language

The visual layer lives entirely in `webui/frontend/src/app/globals.css` — no
inline styles, no per-component styling — so the look is changed by editing the
stylesheet and *nothing else*. Class names, ids and visible text are contract:
the browser suite pins them (`tests/test_webui_e2e.py`).

A 2026-08-07 refresh made it denser and slightly more modern without touching
the DOM:

- **Tokens** (`:root` + the three theme blocks): layered shadows
  (`--shadow`, `--shadow-lift`), a shared focus ring `--focus` applied
  `:focus-visible` on buttons/inputs/selects/links, a system font stack
  `--font-ui`, and an accent-tinted hover wash `--accent-ink`. **Accent moved
  from flat teal (`#00706b`) to a modern indigo (`#4f46e5` light /
  `#818cf8` dark) with amber `--accent-2`; neutrals went from warm gray to cool
  slate (`#f5f6fa` bg, `--ink #0f172a`).**
- **Density** (measured at 1440×900): app-shell padding 12→8 px, topbar
  brand 38→30 px and title 22→18 px, control sidebar 320→296 px, panel
  sections 7/9→5/8 px, uppercase section headings 12→10.5 px with
  letter-spacing, tab buttons 4→3 px vertical, workflow/card chrome 8→6 px.
  Body `scrollWidth == innerWidth` (1440) on all four tabs — no horizontal
  overflow — and the panel-overflow guard stays green.
- **Primary action** gets a subtle vertical accent gradient + lift instead of
  the flat fill; active tabs and cards use the shared shadow token.
- **Theme switch** — a sun/moon button in the topbar (`data-testid="theme-toggle"`)
  flips `data-theme` on `<html>` between `light` and `dark` and persists the
  choice in `localStorage["quest-theme"]`. Default is **dark**; a saved choice
  always wins over the OS `prefers-color-scheme` (the `:root[data-theme=…]`
  blocks come after the media query, so they win). Label via the `web.theme_toggle`
  key in all three locales. The `@media (prefers-color-scheme: dark)` rules that
  flip `button.primary`'s label colour have `:root[data-theme=…]` twins so the
  toggle path is styled too.

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

Two of its guards exist because a panel defect had no signal anywhere else:

- **`TestLayoutAndAccessibility::test_no_inner_container_overflows_its_box`**
  parametrises over all four tabs and fails if any element wider than its box
  has `overflow-x: visible`. The page-level check missed this class: the
  advanced card clips with `overflow: hidden`, so the document never scrolled
  while the quenching table was cut off at the card's 560 px body. Fixed by
  `min-width: 0` on `.quench-editor`, which lets the `quench-table-wrap` scroll
  container govern its own width instead of the grid item forcing it wide.
- **`test_the_header_selects_have_accessible_names`** pins an `aria-label` on
  the dye and run pickers, which are bare `<select>`s without a wrapping
  `<label>` (the chain/residue/atom selects are named by their wrapping label,
  so they are excluded). Both labels go through `t()` and the `web.dye` /
  `web.run_select` keys, keeping the translation-backlog ratchet flat.
