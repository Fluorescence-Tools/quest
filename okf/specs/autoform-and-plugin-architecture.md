---
type: Specification
title: AutoForm Web Support & ChiSurf Multi-Interface Plugin Architecture for QuEst
description: Comprehensive PRDs, architecture specifications, and implementation checklist for bringing dynamic AutoForm support to the Web UI and standardizing QuEst on the ChiSurf 5-way interface plugin contract (API, CLI, RPC JSON, ZMQ, Web AutoForm).
resource: quest/rpc/zmq_server.py, quest/rpc/services.py, quest/gui/generate_view_spec.py, webui/frontend/src/components/autoform
tags: [autoform, chisurf, plugins, zmq, rpc, cli, api, webui, prd, architecture]
status: draft
generated: { by: human:tpeulen, at: 2026-08-07T00:00:00Z }
stale_after: 2026-11-07
---

# Architecture Overview: ChiSurf Plugin Compatibility & Web AutoForm

This specification defines how **QuEst** implements the **ChiSurf Plugin Standard** across 5 unified execution interfaces, bringing desktop-grade AutoForm declarative rendering to modern web applications.

```mermaid
flowchart TD
    subgraph Single Core Science Engine ["QuEst Python Core Science Engine"]
        API["1. Python API\n(quest.api / quest.project)"]
        Catalog["Parameter Catalog\n(parameter_catalog.json)"]
    end

    subgraph Interface Seams ["ChiSurf Plugin Interfaces"]
        CLI["2. CLI\n(quest.cli:cli)"]
        RPC["3. RPC JSON (HTTP)\n(quest.rpc.services)"]
        ZMQ["4. ZMQ Server\n(quest.rpc.zmq_server)"]
        ViewGen["5. AutoForm Spec Generator\n(generate_view_spec.py)"]
    end

    subgraph Web UI Engine ["Web UI Client"]
        WebAutoForm["Web AutoForm Engine\n(React / TS)"]
        RestAPI["FastAPI / REST Client"]
    end

    API --> CLI
    API --> RPC
    API --> ZMQ
    Catalog --> ViewGen
    ViewGen -->|quest.view.json| WebAutoForm
    WebAutoForm -->|State & RPC Callbacks| RestAPI
    RestAPI --> RPC
```

---

# PRD 1: Web AutoForm Engine (React / WebUI)

### Objective
Provide a native, schema-driven React **Web AutoForm** component tree that dynamically renders complex hierarchical configuration forms directly from ChiSurf AutoForm view-spec JSON (`quest.view.json` or any plugin `.view.json`).

### Requirements
1. **Dynamic Section Dispatching**:
   - `panel`: Accordion / collapsible container holding an ordered list of child sections.
   - `value`: Scalar input (`kind`: int, float, str, text) with unit labels, help badges, step sizes, and dirty state tracking.
   - `choice`: Dropdown/radio selector supporting inline options or dynamic catalog parameter options.
   - `toggle` / `toggle_row`: Checkbox toggle inputs.
   - `button_row`: Action trigger buttons.
   - `table` / `custom`: Tabular record displays and bespoke extension slot renderers.
2. **State & Event Binding**:
   - Accepts `spec` (JSON tree), `data` (current project JSON object), `onChange(path, value)`, `onSave(path)`, and `catalog`.
   - Resolves target paths (e.g. `attachment.chain`, `fret.R0_matrix`) using dot-notation getters/setters.
3. **Localisation Seam**:
   - Integrates with `t(key, fallback)` or uses translated text directly from localized view specs (`quest.view.de.json`, `quest.view.fr.json`).

---

# PRD 2: ChiSurf 5-Way Plugin Interface Compatibility

### Objective
Ensure QuEst serves as the gold-standard blueprint for ChiSurf plugins by exposing all 5 standard plugin execution surfaces:

| Surface | Interface Module | Description |
|---|---|---|
| **1. API** | `quest.api` / `quest.project` | Direct Python function imports for scripts and notebooks. |
| **2. CLI** | `quest.cli:cli` | Command-line interface for headless batch runs and pipeline scripting. |
| **3. RPC JSON** | `quest.rpc.services` / `dispatcher.py` | Standalone JSON-RPC 2.0 service dispatcher over HTTP (FastAPI `/api/rpc`). |
| **4. ZMQ** | `quest.rpc.zmq_server` | ZeroMQ REQ/REP server allowing out-of-process RPC execution from any language (C++, Julia, Rust, Python). |
| **5. Web AutoForm** | `quest.view.json` + `webui/frontend` | Dynamic schema-driven web UI form generation replacing PyQt-specific widgets. |

---

# Implementation Checklist

- [x] **Phase 1: Architecture Specification & PRDs**
  - [x] Document ChiSurf 5-way interface standard and Web AutoForm PRDs (`okf/specs/autoform-and-plugin-architecture.md`).
- [ ] **Phase 2: Python Backend ZMQ & RPC Extensions**
  - [ ] Implement `quest.rpc.zmq_server` (ZeroMQ RPC server for QuEst RPC methods).
  - [ ] Extend `quest.rpc.services` and FastAPI backend to expose `/api/autoform-spec` and ZMQ endpoints.
- [ ] **Phase 3: Web AutoForm React Engine (`webui/frontend`)**
  - [ ] Build `src/components/autoform/AutoForm.tsx`.
  - [ ] Build `src/components/autoform/AutoSection.tsx` & `AutoPanel.tsx`.
  - [ ] Build `src/components/autoform/AutoValue.tsx`, `AutoChoice.tsx`, `AutoToggle.tsx`, `AutoButtonRow.tsx`.
  - [ ] Add AutoForm mode toggle / component view in Web UI sidebar.
- [ ] **Phase 4: Tests & Verification**
  - [ ] Python unit tests for ZMQ server (`tests/test_zmq_rpc.py`).
  - [ ] Backend tests for `/api/autoform-spec` & RPC services.
  - [ ] E2E browser tests for Web AutoForm rendering and interactions.
