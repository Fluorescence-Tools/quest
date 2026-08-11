---
type: Specification
title: Declarative form scheme — ChiSurf AutoForm and QuEst as the blueprint
description: The view-spec JSON contract for declaring editors without GUI code, and how QuEst already implements it — the blueprint for porting a Python app to a web app.
resource: webui/frontend/src/components, quest/settings/parameter_catalog.json, quest/gui/generate_view_spec.py
tags: [autoform, scheme, forms, web, chisurf, blueprint, porting]
status: draft
generated: { by: human:tpeulen, at: 2026-08-07T00:00:00Z }
stale_after: 2026-11-07
sources:
  - id: chisurf-dataspec
    resource: ../../chisurf/chisurf/core/dataspec/__init__.py
    title: chisurf/core/dataspec (section vocabulary + parser)
    author: human:tpeulen
    last_modified: 2026-08-07
  - id: chisurf-autoform
    resource: ../../chisurf/chisurf/gui/autoform/auto_form.py
    title: chisurf/gui/autoform (renderer)
    author: human:tpeulen
    last_modified: 2026-08-07
  - id: quest-view-spec
    resource: quest/gui/generate_view_spec.py
    title: quest/gui/generate_view_spec.py (generator)
    author: human:tpeulen
    last_modified: 2026-08-07
  - id: quest-catalog
    resource: quest/settings/parameter_catalog.json
    title: quest/settings/parameter_catalog.json
    author: human:tpeulen
    last_modified: 2026-08-07
---

# Why this exists

A Python desktop app and its web twin should not each hand-write a form. QuEst
already proves the alternative: **one declarative JSON scheme, generated from a
single source of truth, rendered by two unrelated toolkits.** This concept
documents that scheme so a new app being ported from Python to the web can reuse
it instead of inventing a second way to describe an editor.

The scheme predates QuEst: it is ChiSurf's AutoForm view-spec. What QuEst adds,
and what the port should copy, is the *derivation* — one catalog driving the
desktop spec, the docs and the web UI so none of them can silently diverge.

# The scheme (ChiSurf AutoForm view spec)

An editor is a plain, toolkit-free JSON tree of typed **sections**. The parser
lives in `chisurf/core/dataspec/__init__.py`; the renderer in
`chisurf/gui/autoform/auto_form.py`.

```
{ "sections": [ ... ], "plots": [ ... ] }
```

Section primitives (`"type"` field):

| type | role |
|---|---|
| `panel` | foldable group box holding an ordered list of child sections |
| `value` | one scalar field (`kind`: int / float / str / text / expression / file / directory / date) |
| `choice` | one-of-N selector (combo/radio), options inline or via `options_source` |
| `toggle` / `toggle_row` | a boolean / a run of booleans |
| `button_row` | a row of action buttons |
| `table` | record table with declared columns, `source`, optional `update_call` |
| `parameter_group` / `parameter_group_table` | a `FittingParameterGroup`'s params as a grid or table |
| `dynamic_group` | variable-length list of rows with add/del |
| `curve_input` | a data-curve picker (e.g. an IRF) |
| `custom` | bespoke widget referenced by string `key` + registry + `options` |
| `plot` | an inline plot whose data comes from a named `source` method |
| `dock_area` | rearrangeable/floatable panels with `split`/`sizes`/`persist` |
| `info` | read-only HTML/Markdown block |
| `wizard` / `wizard_step` | directed two-column multi-step form |

Three properties make it work:

- **Binding by name, not reference.** A section resolves `target`/`attr`
  against the model via `getattr`/`setattr`, or dispatches a `set_action`.
  `custom` sections reference a GUI widget by `key` + a GUI-side registry. The
  spec never imports a toolkit.
- **Localisation is a single parse seam.** On load, the text-carrying fields
  (`title`, `label`, `description`, `add_label`, `column`… — the lists
  `_TEXT_KEYS` / `_ITEM_TEXT_KEYS` in the parser) run through the i18n layer
  `tr()`, one seam every spec passes through.
- **A plain dict is a form for free.** `SettingsView(dict)` derives a
  `ModelView` from the dict's structure (nest → `panel`; scalar → typed field),
  and `ParameterGroupView(group)` derives one from a param group — zero authored
  JSON for the common case.

## QuEst as the blueprint

QuEst stands on a source of truth and *generates* every surface's schema:

- `quest/settings/parameter_catalog.json` — one entry per parameter: `type`,
  `unit`, `label`, `description`, `options`, `category`.
- `quest/settings/locales/{en,de,fr}.json` — the catalog + UI strings in each
  language; `quest/gui/generate_view_spec.py` emits **one AutoForm view spec per
  language** (`quest.view.json`, `quest.view.<code>.json`) that the desktop's
  `quest/gui/autoform_panel.py` hands straight to ChiSurf's renderer.
- The web UI serves the **same** spec: `/api/autoform-spec` returns
  `build_view_spec(locale)` and a small React autoform
  (`webui/frontend/src/components/autoform/`) renders it by section `type` —
  the port's = `Field` per spec type. `doc/generate_parameter_docs.py` writes
  the docs.

Since 2026-08-09 the web form is **AutoForm-only**: the hand-built React form
that used `Field.tsx` (from the catalog) was deleted, so both the desktop and
the web render the same authored spec. `lib/autoformPaths.ts` maps each spec
`attr` (the flat `ProjectFormModel` name) to its nested project path.

So the invariant the port wants: **change the catalog and the desktop form, the
web form, the docs and the translated labels all follow — asserted by
`test_view_spec.py` which regenerates in memory and fails on drift.**

The reusable pieces the port should copy:

1. A single parameter catalog (authored in one place).
2. AutoForm view-spec emission from that catalog (the shared contract).
3. A small schema→field renderer per toolkit (e.g. one React `Field` per spec
   `type` — what `AutoSection.tsx` plus the `.view.json` together are).
4. All user-facing text at the parse/localisation seam.

# Decision aids for a port

| python app | port via |
|---|---|
| parameters on model/group | `SettingsView`/`ParameterGroupView` derivation + catalog |
| output fields / states | `value` read-only / `info` |
| tables of records | `table` with `source` + `update_call` |
| bespoke panel | `custom` `key` + registry |
| multi-page setup | `wizard` |
| plot windows | `plot` `key` + options |

# Open questions

- Whether the port's web renderer can consume the view-spec JSON *directly* (a
  React autoform by spec `type`) or whether the spec should back to a second,
  tabular catalog first. QuEst's web UI today uses the catalog, the desktop the
  spec: unifying on the spec both renderers is the cleaner target.
- Where the reusable tour + i18n ratchet settle (they are siblings of this).

# Related

- The limits of "flat" parameters vs the spec's `custom`/table primitives.
- [project-schema.md](/architecture/project-schema.md) — the *project JSON*
  contract (data), distinct from the *view spec* (the form over it).