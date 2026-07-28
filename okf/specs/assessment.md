---
type: Specification
title: Assessment
description: The gap backlog — where the current tree diverges from the target architecture, itemised with ids and status.
resource: okf/specs/plugin-integration.md
tags: [backlog, assessment, layering, technical-debt]
timestamp: '2026-07-26T00:00:00Z'
---

Gaps between the tree and
[the target split](plugin-integration.md). Each row is a discrete, landable
unit. Update the row and the totals in the same change that closes it, and say
so in [okf/log.md](../log.md).

**Status: 9 of 11 closed** (opened 2026-07-26; LAY-02, LAY-03, LAY-05, LAY-08 and
LAY-09 closed 2026-07-27; LAY-06, LAY-10 and LAY-11 closed 2026-07-27 by the
AutoForm migration). Open: LAY-04 (`quest.api` is still a re-export shim) and
LAY-07 (three unrelated long-running mechanisms, none cancelable).

| Id | Severity | Status | Gap |
|---|---|---|---|
| LAY-01 | high | **closed 2026-07-26** | ~~The web backend imports **private CLI helpers**.~~ Project template and validation now live in [`quest/project.py`](/subsystems/core-api.md) (`template_project`, `validate_project`, `ProjectValidationError`), exported from `quest.api`. The CLI keeps `_validate_project` as a thin translator to `ClickException`; the web backend and `quest/lib/fps_json.py` — which imported the CLI *from inside the library* — now call the domain. Guardrail: `tests/test_service_layer.py::TestLayering`. |
| LAY-02 | high | **closed 2026-07-27** | ~~The GUI owns a domain object.~~ `TransientDecayGenerator` no longer inherits `DonorDecay`. It **builds** one per run from the form (`build_model()`, driven by `MODEL_ARGUMENT_SOURCES`) and reads results back through delegating properties. The model is a plain object with no Qt in it; a guardrail test asserts `DonorDecay not in TransientDecayGenerator.__mro__`. Behaviour is unchanged, pinned by 11 characterisation tests written before the change. |
| LAY-03 | high | **closed 2026-07-27** | ~~No RPC registry.~~ `quest/backend/` provides a `ServiceDispatcher`, an `InProcessClient` and a duck-typed `register_services(dispatcher)` registering **all eleven** methods behind one `{"ok", "result"\|"error"}` envelope. The last three (`quest.structure.metadata`, `quest.jobs.list`, `quest.jobs.get`) landed with LAY-09. |
| LAY-04 | medium | open | **`quest.api` is a re-export shim, not a facade.** It re-exports names from `quest.core`; it is not verb-oriented and has no local/remote resolution. Callers reach past it: the web backend imports `quest.lib.tools.dye_diffusion.dye_diffusion` and `quest.core` directly. |
| LAY-10 | medium | **closed** (2026-07-27) | **The GUI edited a legacy quenching shape.** `DonorDecay.quencher` (`{residue: {atoms, kQ}}`) carried no per-residue contact radius and no per-residue stickiness, so the Qt table could not express what `amino_acid_interactions` does, and the widget ran `stickiness_mode="global"` while every other surface ran `"amino_acid"`. Closed by the AutoForm migration, not by a fix to the old table: the form's state *is* the project, so its quencher table edits `amino_acid_interactions` directly — kQ, contact radius, quenching atoms and slow factor per residue. There is no second shape left to disagree. |
| LAY-09 | medium | **closed 2026-07-27** | ~~Domain logic living in the web backend.~~ The PDB/CIF chain–residue parser moved to `quest/structure_info.py::structure_metadata`, and job listing/reading/archiving to `quest/jobs.py`. The routes are now adapters; the capabilities are reachable from the CLI, notebooks and a host, and are covered by 14 domain tests instead of only through HTTP. |
| LAY-05 | medium | **closed 2026-07-27** | ~~QuEst declares nothing.~~ `quest/manifest.json` carries the identity, the `gui`/`cli`/`services` entry points and all eleven RPC methods with params/result schemas and `long_running` flags; five tests fail if it and the registry disagree. **Still open elsewhere:** the ChiSurf-side plugin folder has no manifest of its own — `chisurf/plugins/quenching_estimator/` contains only `__init__.py`, `README.md` and `test/`, so it loads through the host's legacy AST-based discovery and declares no entry points, no RPC methods, and no state schema. This is exactly the "one first-class plugin has no manifest at all" case ChiSurf's own plugin spec calls out. |
| LAY-11 | medium | **closed** (2026-07-27) | **The Qt form drew its own plots.** It built pyqtgraph widgets by hand for the decay, the trajectory and the autocorrelation. Closed with LAY-06, as predicted: the three plots are now `plot` sections in `quest.view.json`, rendered by AutoForm through ChiSurf's `chiplot`, and QuEst supplies only the data (`decay_series`, `trajectory_series`, `autocorrelation_series` on the form model). QuEst owns no plotting code at all. |
| LAY-06 | medium | **closed** (2026-07-27) | ~~No declarative view spec.~~ The Qt form is `quest/gui/quest.view.json`, **generated** from `quest/settings/parameter_catalog.json` by `quest/gui/generate_view_spec.py` and rendered by ChiSurf's `AutoForm`. Every `.ui` file is gone — `dye_diffusion2.ui`, `pdb_widget.ui`, `ui/*.ui` — and with them the 1,899-line widget, `widgets.py` and the 98 kB generated Qt resource module; `quest/gui/dye_widget.py` is now ~130 lines that put an AutoForm in a QWidget. The form binds to `quest.gui.form_model.ProjectFormModel`, whose state **is** a project dict, so the hand-written `to_project_dict`/`load_project_dict` pair — where the GUI drifted from the CLI before — no longer exists. `quest/gui/autoform_panel.py` remains for rendering a single RPC method's parameters from the manifest. Covered by `tests/test_view_spec.py`, `tests/test_form_model.py`, `tests/test_gui_widget.py` and `tests/test_gui_simulation.py`. |
| LAY-07 | medium | open | **Long-running work has three mechanisms.** A thread pool in the web backend, a blocking call in the CLI, a `QProgressDialog` in the GUI. None is cancelable from outside; a scan cannot be observed or stopped through any programmatic surface. |
| LAY-08 | low | **closed 2026-07-27** | ~~`sys.path` injection.~~ Every internal import is spelled `quest.lib.…` and `quest/__init__.py` no longer touches `sys.path`. This was not merely cosmetic: `lib.structure.Structure` and `quest.lib.structure.Structure` were **two different classes**, so an `isinstance` check across the boundary failed and a `Structure` got wrapped twice. |

# Deduplication against imp-tricks

Target: [imp-tricks-dedup.md](imp-tricks-dedup.md). General fluorescence
machinery belongs in `IMP.bff`; QuEst carries a second copy of much of it.

> **Numbering warning, 2026-07-28.** The `DUP-*` labels in this table are the
> authoritative ones, and [handover.md](../handover.md) used a *different* set:
> it called the `density2points` adoption "DUP-02" (here **DUP-05**) and the
> distance metrics "DUP-03" (here **DUP-04**). The commit messages
> `78dc4c8` and `9b1192e` follow the handover's numbering and are therefore
> mislabelled against this table. The work is right; the labels in those two
> commits are not. Cite this table, not the handover, for a row number.

| Id | Severity | Status | Gap |
|---|---|---|---|
| DUP-01 | high | **DONE** (2026-07-28) | **The AV backend dispatch exists twice.** `AV.__init__` + `quest/lib/imp_av.py` reimplement `IMP.bff.av.compute_av`, down to the same backend ordering — and QuEst's copy **does not work**: in `arm64`, where a compiled `IMP.bff` exists, neither backend of `compute_av` runs: `av/compute.py:162` calls `Particle.get_particle_index()` (this IMP spells it `get_index()`), and `av/compute.py:231` rejects LabelLib because it looks for `LabelLib.AV` while this LabelLib exposes `dyeDensityAV1`. Both are one-line fixes **in imp-tricks**, and they unblock the largest deletion QuEst has left. Selection now defaults to LabelLib explicitly (`resolve_av_backend`, `QUEST_AV_BACKEND`, `AV.av_backend`), which contains the damage but does not remove the duplicate. |
| DUP-02 | high | **mostly closed** (2026-07-28) | **The capability check asks the wrong question.** `HAS_IMP_BFF = hasattr(IMP.bff, "AV")` looks for a compiled attribute; the imp-tricks package exposes its AV at `IMP.bff.av.compute_av`. Partially addressed: the backend is now chosen explicitly rather than by this flag alone. `quest.imp_tricks.enable_imp_tricks()` makes a checkout importable (a compiled `IMP.bff` shadows it otherwise), but imp-tricks is still **not installed** in any environment. |
| DUP-03 | medium | open | **`fret_rate_trace` is general code in the wrong repository.** Donor trajectory + acceptor cloud + R0 → per-frame rate is fluorescence machinery. It belongs with `IMP.bff.cgdye.analysis.fret`, which additionally handles κ² — the thing QuEst's version lacks. |
| DUP-04 | medium | **DONE** (2026-07-28) | **Distance metrics are duplicated.** ⟨R_DA⟩ and ⟨R_DA⟩_E: adopt `random_distances` + `av_pair_statistics` — verified to agree with `RDAMean`/`RDAMeanE` within sampling noise, and nothing blocks it. **`dRmp` goes the other way:** upstream's `r_mp` returns ⟨R_DA⟩ (its own docstring says "same as `r_da_mean` here"), which is a different quantity — 47.65 vs 51.53 Å on 148l E15/E90 — and it cannot be computed from a distance sample at all. QuEst's implementation is the correct one and should be contributed upstream. |
| DUP-05 | low | **partly done** (2026-07-28) | **Grid and geometry kernels are duplicated**: `density2points`, `subav`/contact-volume splitting, `asa`/`spherePoints` all have homes in `IMP.bff.av._kernels` / `IMP.cgmol.sterics`. |
| DUP-07 | medium | **partly closed 2026-07-27** | ~~RCSB fetching duplicated with ChiSurf.~~ Moved into `chisurf/core/fio/structure/fetch.py` (merging QuEst's `.cif` fallback into it), re-exported by the `fps_json_editor` plugin, and QuEst delegates to it when a checkout is reachable. **Remaining:** fps.json payload handling — ChiSurf's `payload.py` drags its `FpsJsonModel` along, so moving it to core is the next step. **Not a duplicate after all:** MRC writing — ChiSurf voxelises point clouds via `IMP.em`, QuEst writes precomputed density grids. |
| DUP-06 | low | open | **QuEst carries its own PDB/CIF parser and `Structure`.** Predates the IMP dependency; long-term this is IMP's job. |

# Severity

- **high** — blocks the integration outright, or is an active layering
  violation that new code will copy.
- **medium** — real duplication or a missing contract; costs correctness over
  time rather than today.
- **low** — a hazard that has not bitten yet.

# Notes

The cheap half is done: project logic is in the domain (LAY-01), the method
surface is complete and declared (LAY-03, LAY-05), and the capabilities that
only the browser could reach are now domain functions (LAY-09).

**LAY-02 is closed**, so the model is a plain object on every surface and a host
can drive it without a form — the structural prerequisite for
[OBJ-02](objectives.md). LAY-04 (a real facade) is the natural next step, and
LAY-10 is now cheap: the widget builds its model through one place
(`build_model`), so switching the quencher table to `amino_acid_interactions`
touches that method and the table, not the whole widget.

Closing LAY-01 also removed a **triplicated** implementation of "simulate one
labeling site and report it": the CLI, the web backend and
`quest/lib/fps_json.py` each had their own copy, and they had drifted — the
library's copy omitted the quencher attribution the other two reported, and the
CLI's copy dropped the fps.json AV overrides into the project without also
applying them to `av_parameter`. All three now call `quest.scan.simulate_site`.

None of these are correctness bugs in the simulation. For those see
[references/known-issues.md](/references/known-issues.md); for where QuEst is
headed scientifically see [objectives.md](objectives.md).
