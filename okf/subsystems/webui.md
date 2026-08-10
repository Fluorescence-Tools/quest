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
GET  /api/autoform-spec         # build_view_spec(locale) — the form the UI renders
GET  /api/tour                  # quest.tour: the guided-tour steps
GET  /api/demo-project          # quest.tour: the 148L demo project
GET  /api/i18n                  # locales_handler
GET  /api/dyes                  # dye presets;  POST /api/dyes appends one
GET  /api/quenching-defaults    # REFERENCE_PET_QUENCHING + DEFAULT_QUENCHER_ATOMS
GET  /api/template
POST /api/validate
POST /api/structure/metadata    # chains/residues parsed from an uploaded structure
POST /api/simulate
POST /api/scan
GET  /api/jobs, /api/jobs/{id}, /api/jobs/{id}/download
```

**The backend goes through `quest.api`, not through the CLI.** It once imported
`quest.cli`'s private `_template_project` / `_validate_project` at four call
sites (`LAY-01`); since phase 8 it calls the facade and maps `error_code` onto
an HTTP status, so "a valid project" has one definition and the web is not
coupled to a terminal surface ([subsystems/core-api.md](/subsystems/core-api.md)).
`to_payload` is applied here because the transport is the boundary that
serialises — the same rule the [ZMQ server](/subsystems/rpc.md) follows.

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

## The sidebar form is generated, not written (since 2026-08-09)

`AppLayout` fetches `/api/autoform-spec?locale=…` and renders it through
`components/autoform/` (`AutoForm` → `AutoPanel` → `AutoSection` →
`AutoValue`/`AutoChoice`/`AutoToggle`). **No form is hand-written on this
surface any more**; the ~540-line legacy React block and its AutoForm/legacy
toggle are gone, and the only remaining switch is the JSON overlay
(`showJson`). The desktop renders the *same* spec through ChiSurf's AutoForm,
so a parameter added to the catalog reaches both
([specs/autoform-scheme.md](/specs/autoform-scheme.md)).

Two consequences worth knowing before editing anything here:

- **`lib/autoformPaths.ts` is load-bearing.** A spec `attr` is the flat
  `ProjectFormModel` attribute (`attachment_chain`, `fret_enabled`, `fret_R0`),
  not the nested project path (`attachment.chain`, `fret.enabled`,
  `fret.R0_matrix[0][1]`). `AutoSection` resolves every `attr` through
  `attrToPath` before reading or writing; without it the form silently produces
  projects the backend rejects. `tests/test_autoform_blueprint.py` fails if the
  catalog grows a parameter the map does not cover.
- **Two fields do not go through `setNested`.** `fret.enabled` routes through
  `setFretEnabled` (which seeds the acceptor dye and the R0 matrix — a bare
  toggle leaves a one-dye project) and `fret.R0_matrix.0.1` through
  `setForsterRadius`, because a list cell is not a path `setNested` can index.

The scan residue picker and the Calculate Scan button live in a **Scan
Settings** card on `app/scan/page.tsx` — they were in the deleted sidebar
block. `app/advanced/page.tsx` still uses the hand-written `Field.tsx` +
`HelpBadge.tsx` (labels and help from `lib/parameterCatalog.ts`) and
`QuenchingEditor.tsx`, the per-residue interaction table; those are the pieces
the view spec does not describe.

## Click-to-select a labeling site

`ProteinViewer.tsx` (NGL, vendored at `public/vendor/ngl.js` — no CDN) wires
NGL's picking proxy on `stage.signals.clicked` to the same `onResidueSelect`
callback the sequence bar uses, so clicking a residue in the 3D view sets the
attachment. The desktop has the same gesture through ChiSurf's `SequenceDock`
and `MolView` ([subsystems/chisurf-widgets.md](/subsystems/chisurf-widgets.md));
a feature that reaches only one of the two surfaces is a bug, which is why they
land together.

> **NGL is scheduled to be replaced by chimol here.** The two surfaces draw the
> same structure with two different engines — ChiSurf's chimol on the desktop,
> NGL in the browser — which is the same two-implementations shape the AutoForm
> migration removed from the forms. ChiSurf's `chimol → WebGPU` work exists to
> close it: one WGSL source, a desktop driver and a browser driver, so the web
> viewer becomes the *same* renderer rather than a lookalike.
> **Not yet actionable** (checked 2026-08-10): the desktop WGSL renderer runs
> (`CHIMOL_RENDERER=wgpu`), but the browser/JS driver has not been started and
> `rendercanvas`'s `pyodide` backend is still an open question upstream. The
> status lives on ChiSurf's agent board (`~/dev/chisurf/okf/agent-board.md`) and
> in `~/dev/chisurf/okf/plugins/chimol-web.md` — check those, not this file,
> for whether it has landed.

## Guided tour (`components/tour/`)

`Tour.tsx` is a **dependency-free, quest-agnostic onboarding overlay** — it
imports nothing from quest, knows no i18n, and takes `steps` + `labels` as
props. It is meant to be copy-pasted into FPSIMP unchanged (same React tree, no
React DOM). QuEst's shape lives in `tourSteps.ts`: `getTourSteps(t, navigate)`
returns `{ target, title, body, before }[]` spotting `[data-tour="…"]` hooks,
`getTourLabels(t)` builds the control labels, and both pull every string through
`t()` so the translation ratchet stays flat.

How it works: opening locks `body` scroll; each step `scrollIntoView`s the
target, measures it with `getBoundingClientRect`, and draws a transparent
**spotlight** whose huge `box-shadow` (`0 0 0 9999px rgba(…,0.6)`) dims
everything but the target — the one-element cutout, no four-pane mask. The
tooltip card sits below (or above) the box, clamped into the viewport, and is
repositioned on resize/scroll. Keys: `Esc`/click-elsewhere close, `←→` step,
`Enter` advances. Labels (`next`, `prev`, `done`, `close`,
`stepIndicator(c,t)` are passed in already-translated. Spotlight targets live
on `data-tour` attributes, so steps are stable under refactors. The button that
opens it is **`data-testid="guide-btn"`** (it was `tour-btn` until 2026-08-08,
when the control was renamed "Guide" to match ChiSurf's chrome; `.tour-btn` is
still the CSS class on the tour's own next/prev buttons, which is a trap when
grepping). The step copy is the `web.tour.*` keys in all three locales.

Since 2026-08-07 the tour is **exhaustive and cross-page** (24 steps, all three
locales). A step may carry a `before` hook that routes the app first (e.g.
`() => navigate("/simulation")`) — that is how the sidebar panels, which switch
with the active tab, get their own steps. Two consequences:

- **Target measurement retries.** After `before` runs, the effect polls the
  selector for ~3.5 s (`14 × 250 ms`) before falling back to a centred card, so
  a target that only mounts after navigation (or a data round-trip) is caught.
  `AppLayout` passes `navigate` as a guarded `router.push` that is a no-op when
  already on that path (the step effect re-runs on `steps` identity changes).
- **`data-tour` hooks on every surface** — the topbar (language, theme, cite,
  preset), the toolbar (PDB form, upload, download, tabs), the sidebar (runs,
  JSON toggle, structure, labeling, FRET, AV, simulation-settings,
  scan-settings, stickiness) and each page (labeling viewer, simulation
  results, scan results, the two advanced cards). The first step is a centred
  welcome card that routes to `/labeling` so the walk is identical whichever
  page starts it; the last is a centred closing card. `TestTour` walks through
  the `before` navigation and pins the titles via the browser suite.

# Design language

The visual layer lives entirely in `webui/frontend/src/app/globals.css` — no
inline styles, no per-component styling — so the look is changed by editing the
stylesheet and *nothing else*. Class names, ids and visible text are contract:
the browser suite pins them (`tests/test_webui_e2e.py`).

## The CSS refresh

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

**FRET is a sidebar section of the generated form, not a page panel.** It was
a hand-written panel on the labeling page until 2026-08-09; that page is now
just the `ProteinViewer`. Enabling FRET still seeds the acceptor on the *last
labelable* residue of the current chain — `lastLabelableResidue` skips the
waters and ions structures end with, which a dye cannot be tethered to — so
switching it on yields a project that validates. That seeding is why the
`fret.enabled` field is routed through `setFretEnabled` rather than written
straight into the project.

# Tests

`webui/backend/tests/test_backend_api.py` covers the API directly.
`tests/test_webui_e2e.py` drives a real browser and **skips unless both servers
are already running** — see [workflows/testing.md](/workflows/testing.md).

Since the form became generated, the browser suite locates fields *by label*
through `_autoform_field` / `_autoform_input` / `_autoform_check` rather than by
the ids the hand-written inputs used to carry; `_use_legacy_form` is gone. Two
mechanical traps live here: `_server_is_up` waits **10 s** because a cold
Next.js dev server takes >3 s and a shorter timeout silently skipped the whole
suite, and a styled toggle's real checkbox is `display:none`, so a `.click()` on
the visible slider times out intermittently — drive it with `.check()` or
`evaluate()`.

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
