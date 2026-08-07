# ChiSurf AutoForm Dependency & Integration Architecture

This specification details the ChiSurf AutoForm components available to QuEst as an allowed dependency, explaining how QuEst consumes them both as a host-integrated ChiSurf plugin and as a standalone desktop application.

---

## 1. Overview & Dependency Scope

ChiSurf provides a complete, production-grade declarative AutoForm engine under `chisurf.gui.autoform` and `chisurf.core.dataspec`. QuEst imports these components whenever ChiSurf is reachable on `sys.path` (via `quest.hosts.enable_chisurf()`), and provides a lightweight in-tree fallback (`InTreePyQtAutoForm`) when running standalone without ChiSurf.

---

## 2. Primary ChiSurf AutoForm Classes

### `chisurf.gui.autoform.auto_form.AutoForm`
The master `QWidget` AutoForm container that walks a model's declarative `view_spec()` / `DataSet` and builds the control panel by composition.

- **`AutoForm(model)`**: Renders a bound control panel for any model object (such as `ProjectFormModel`) from its `view_spec()`.
- **`AutoForm.from_rpc_method(method, parent=None, **kwargs)`**: Renders a parameter form for any declared plugin RPC method in `quest/manifest.json` (e.g. `quest.simulate`).
- **`AutoForm.from_parameter_group(group, parent=None, **kwargs)`**: Renders a `FittingParameterGroup` parameter grid without authored JSON.
- **`form.sync_fields()`**: Re-reads model attributes into controls without rebuilding the Qt widget layout.
- **`form.refresh_plots()`**: Triggers live redraw across all embedded plot widgets.
- **`form.state()` / `form.apply_state(values)`**: Serializes and restores complete form state as JSON mappings.

---

## 3. Available AutoForm Sections & Widgets

ChiSurf registers 27 section/plot factories in `chisurf.gui.autoform.sections`:

| Section Key / Class | ChiSurf Module | Description & QuEst Usage |
| :--- | :--- | :--- |
| **`panel`** (`PanelSection`) | `chisurf.gui.widgets.collapsible_box` | Collapsible accordion group boxes with auto-fold timers (`CollapsibleBox`). |
| **`dock_area`** (`DockAreaSection`) | `chisurf.gui.widgets.dock_area` | Rearrangeable, floatable dock tab containers authored in `.view.json`. |
| **`value`** (`ValueSection`) | `chisurf.gui.autoform.sections.builtin` | Scalar value inputs (spin boxes, line edits) with min/max, step, and unit badges. |
| **`choice`** (`ChoiceSection`) | `chisurf.gui.autoform.sections.builtin` | Dropdown selection combo boxes bound to model attributes. |
| **`toggle`** (`ToggleSection`) | `chisurf.gui.autoform.sections.builtin` | Checkbox toggles. |
| **`toggle_row`** (`ToggleRowSection`) | `chisurf.gui.autoform.sections.builtin` | Horizontal row of checkable toggles. |
| **`button_row`** (`ButtonRowSection`) | `chisurf.gui.autoform.sections.builtin` | Action button rows. |
| **`plot`** (`PlotSection`) | `chisurf.gui.autoform.sections.builtin` | Live inline pyqtgraph plots for fluorescence decays, dye trajectories, and ACFs. |
| **`table`** (`TableSection`) | `chisurf.gui.autoform.sections.builtin` | List-of-dict table section editor (e.g. `quencher_rows`). |
| **`chimol`** (`chimol_section`) | `chisurf.gui.autoform.sections.chimol_section` | Embedded 3D Molecular structure viewer canvas. |
| **`help`** (`help_section`) | `chisurf.gui.autoform.sections.help_section` | Inline markdown/HTML help documentation panels. |
| **`embed`** (`embed_section`) | `chisurf.gui.autoform.sections.embed_section` | Adopt existing custom Qt widgets into the AutoForm layout. |
| **`progress`** (`progress_section`) | `chisurf.gui.autoform.sections.progress_section` | Simulation progress bar and cancellation controls. |
| **`background_run`** (`background_run_section`) | `chisurf.gui.autoform.sections.background_run_section` | Asynchronous background worker execution wrapper. |

---

## 4. QuEst Dual-Mode Architecture

```
                               ┌────────────────────────────────┐
                               │       quest.view.json          │
                               │   (AutoForm View Schema)       │
                               └───────────────┬────────────────┘
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       │                                               │
            ChiSurf Host Present                             Standalone / Fallback
                       │                                               │
       ┌───────────────▼───────────────┐               ┌───────────────▼───────────────┐
       │ chisurf.gui.autoform.AutoForm │               │      InTreePyQtAutoForm       │
       │    (Full ChiSurf Engine)      │               │   (QuEst In-Tree Fallback)    │
       └───────────────────────────────┘               └───────────────────────────────┘
```

1. **Host-Integrated Mode**: `dye_widget.py` and `autoform_panel.py` import `chisurf.gui.autoform.auto_form.AutoForm` when ChiSurf is installed/linked.
2. **Standalone Fallback Mode**: When ChiSurf is absent, `InTreePyQtAutoForm` parses `quest.view.json` and renders native PyQt widgets (`InTreeCollapsiblePanel`, `QuEstPlotCanvas`, `QuencherTableWidget`), preserving identical GUI definitions across desktop and web endpoints.
