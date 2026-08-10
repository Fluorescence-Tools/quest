---
type: Subsystem
title: ChiSurf widget reuse — what to embed, not rebuild
description: ChiSurf already ships widgets for sequences, code editing, structure viewing and labeling-site tables. QuEst embeds those instead of carrying a second copy.
resource: quest/gui/
tags: [gui, chisurf, dedup, widgets]
status: stable
generated: { by: human:tpeulen, at: 2026-08-07T00:00:00Z }
stale_after: 2027-02-07
sources:
  - id: chisurf-gui
    resource: ../../chisurf/chisurf/gui/
    title: chisurf/gui/ and chisurf/plugins/
    author: human:tpeulen
    last_modified: 2026-08-07
---

# ChiSurf already has these widgets

QuEst's desktop GUI lives inside ChiSurf's host environment. Every Qt widget
below already exists in ChiSurf — QuEst embeds the original, not a re-creation.

## The mapping

| QuEst feature | ChiSurf widget | Import path |
|---|---|---|
| 3D structure viewer | `MolView` | `chisurf.plugins.chimol.chimol.renderer.view.MolView` |
| Sequence bar | `SequenceDock` | `chisurf.plugins.chimol.chimol.app.sequence_dock.SequenceDock` |
| JSON view / editor | `SimpleCodeEditor(language="JSON")` | `chisurf.plugins.core.code_editor.SimpleCodeEditor` |
| Labeling-site table | `PositionPanel` | `chisurf.plugins.modelling.fps_json_editor.gui.position_panel.PositionPanel` |
| Code editor (full) | `CodeEditor` | `chisurf.plugins.core.code_editor.CodeEditor` |
| Decay/trajectory plots | chiplot | `chisurf.gui.widgets.chiplot` |

## Why not a second copy

QuEst shipped its own `QPlainTextEdit` for JSON, a custom `ResidueButton`
sequence bar, and hand-built pyqtgraph plots. Each duplicates something
ChiSurf already provides and tests. Two copies drift — exactly what happened
with the quenching chemistry, where the GUI shipped different `kQ` values than
the CLI for months. The rule is the same one that governs imp-tricks:
**general machinery belongs upstream, the application embeds it.**

## What was dedup'd

- **JSON view**: replaced `QPlainTextEdit` with `SimpleCodeEditor(language="JSON")`
  — gives syntax highlighting and line numbers. The old code is gone.
- **Sequence bar**: replaced custom `ResidueButton` HBox with ChiSurf's
  `SequenceDock` — QListWidget with proper scrolling, residue numbers, and
  the same wiring pattern `MolViewPluginWindow` uses.

## What remains in QuEst (application-specific)

The donor/acceptor toggle and the `_apply_selection` logic are QuEst-specific:
ChiSurf's `PositionPanel` tracks arbitrary labeling sites, while QuEst has
exactly one donor and one optional acceptor. That adapter lives in
`quest/gui/chimol_widget.py` and calls into the reused widgets.

## Fallback when ChiSurf is absent

Each chisurf import is behind `try/except`. Where it fails, the widget shows a
placeholder label and the rest of the GUI continues. This is the same pattern
as every other optional dependency (`IMP`, `LabelLib`).
