---
type: Specification
title: Assessment
description: The gap backlog — where the current tree diverges from the target architecture, itemised with ids and status.
resource: okf/specs/plugin-integration.md
tags: [backlog, assessment, layering, technical-debt]
status: stable
generated: { by: process:okf-v0.1-import, at: 2026-07-26T00:00:00Z }
stale_after: 2026-10-26
---

Gaps between the tree and
[the target split](plugin-integration.md). Each row is a discrete, landable
unit. Update the row and the totals in the same change that closes it, and say
so in [okf/log.md](../log.md).

**Status: 11 of 11 closed** (opened 2026-07-26; LAY-02, LAY-03, LAY-05, LAY-08
and LAY-09 closed 2026-07-27; LAY-06, LAY-10 and LAY-11 closed 2026-07-27 by the
AutoForm migration; **LAY-04 and LAY-07 closed 2026-07-28**).

**Still open elsewhere:** LAY-05's ChiSurf half — the `quenching_estimator`
plugin folder still has no manifest of its own and loads through the host's
legacy AST discovery. That is phase 10 of
[programme-2026-07.md](programme-2026-07.md).

| Id | Severity | Status | Gap |
|---|---|---|---|
| LAY-01 | high | **closed 2026-07-26** | ~~The web backend imports **private CLI helpers**.~~ Project template and validation now live in [`quest/project.py`](/subsystems/core-api.md) (`template_project`, `validate_project`, `ProjectValidationError`), exported from `quest.api`. The CLI keeps `_validate_project` as a thin translator to `ClickException`; the web backend and `quest/lib/fps_json.py` — which imported the CLI *from inside the library* — now call the domain. Guardrail: `tests/test_service_layer.py::TestLayering`. |
| LAY-02 | high | **closed 2026-07-27** | ~~The GUI owns a domain object.~~ `TransientDecayGenerator` no longer inherits `DonorDecay`. It **builds** one per run from the form (`build_model()`, driven by `MODEL_ARGUMENT_SOURCES`) and reads results back through delegating properties. The model is a plain object with no Qt in it; a guardrail test asserts `DonorDecay not in TransientDecayGenerator.__mro__`. Behaviour is unchanged, pinned by 11 characterisation tests written before the change. |
| LAY-03 | high | **closed 2026-07-27** | ~~No RPC registry.~~ `quest/rpc/` provides a `ServiceDispatcher`, an `InProcessClient` and a duck-typed `register_services(dispatcher)` registering **all eleven** methods behind one `{"ok", "result"\|"error"}` envelope. The last three (`quest.structure.metadata`, `quest.jobs.list`, `quest.jobs.get`) landed with LAY-09. |
| LAY-04 | medium | **closed 2026-07-28** | ~~`quest.api` is a re-export shim.~~ It is verb-oriented (`template`, `validate`, `simulate`, `scan`, `describe`, …), every verb goes through a **client**, and `use_client` swaps in a remote one — the local/remote resolution the row asked for. The web backend calls `api.simulate` instead of reaching into `quest.core`, and maps `error_code` onto HTTP status. **Serialization moved to the transport**: handlers return native results and only `to_payload` at a boundary converts them, because `OBJ-02`'s host optimiser calls `simulate` thousands of times and was paying a numpy→list conversion of every decay channel per iteration. The domain re-exports stay — a notebook that wants `build_donor_from_project` should have it — but the verbs are the supported way in. Contract, error codes and `quest.contract.describe` landed with it; guardrail `tests/test_contract.py` (30 tests). |
| LAY-10 | medium | **closed** (2026-07-27) | **The GUI edited a legacy quenching shape.** `DonorDecay.quencher` (`{residue: {atoms, kQ}}`) carried no per-residue contact radius and no per-residue stickiness, so the Qt table could not express what `amino_acid_interactions` does, and the widget ran `stickiness_mode="global"` while every other surface ran `"amino_acid"`. Closed by the AutoForm migration, not by a fix to the old table: the form's state *is* the project, so its quencher table edits `amino_acid_interactions` directly — kQ, contact radius, quenching atoms and slow factor per residue. There is no second shape left to disagree. |
| LAY-09 | medium | **closed 2026-07-27** | ~~Domain logic living in the web backend.~~ The PDB/CIF chain–residue parser moved to `quest/core/structure/metadata.py::structure_metadata`, and job listing/reading/archiving to `quest/jobs.py`. The routes are now adapters; the capabilities are reachable from the CLI, notebooks and a host, and are covered by 14 domain tests instead of only through HTTP. |
| LAY-05 | medium | **closed 2026-07-27** | ~~QuEst declares nothing.~~ `quest/manifest.json` carries the identity, the `gui`/`cli`/`services` entry points and all eleven RPC methods with params/result schemas and `long_running` flags; five tests fail if it and the registry disagree. **The ChiSurf half closed 2026-07-28** (chisurf `669fce499`). `chisurf/plugins/quenching_estimator/` now follows the `modelling/fps_json_editor` layout — `manifest.json`, `api/`, `rpc/`, `cli/`, `gui/` — with **no `core/`**, because QuEst is the core and nothing there reimplements it: `rpc/services.py` hands the host's dispatcher to `quest.rpc.services.register_services`, `api/contract.py` re-exports `quest.rpc.contract`, and `rpc_methods` is copied verbatim from `quest/manifest.json` with a test that fails if the two diverge. The GUI is a `ChisurfDockTool` instead of a `QMainWindow` used as a central widget, and **nothing imports `quest` at module scope** — the old single file did, so ChiSurf startup pulled in QuEst, IMP and numba whether or not anyone opened the tool. |
| LAY-11 | medium | **closed** (2026-07-27) | **The Qt form drew its own plots.** It built pyqtgraph widgets by hand for the decay, the trajectory and the autocorrelation. Closed with LAY-06, as predicted: the three plots are now `plot` sections in `quest.view.json`, rendered by AutoForm through ChiSurf's `chiplot`, and QuEst supplies only the data (`decay_series`, `trajectory_series`, `autocorrelation_series` on the form model). QuEst owns no plotting code at all. |
| LAY-06 | medium | **closed** (2026-07-27) | ~~No declarative view spec.~~ The Qt form is `quest/gui/quest.view.json`, **generated** from `quest/settings/parameter_catalog.json` by `quest/gui/generate_view_spec.py` and rendered by ChiSurf's `AutoForm`. Every `.ui` file is gone — `dye_diffusion2.ui`, `pdb_widget.ui`, `ui/*.ui` — and with them the 1,899-line widget, `widgets.py` and the 98 kB generated Qt resource module; `quest/gui/dye_widget.py` is now ~130 lines that put an AutoForm in a QWidget. The form binds to `quest.gui.form_model.ProjectFormModel`, whose state **is** a project dict, so the hand-written `to_project_dict`/`load_project_dict` pair — where the GUI drifted from the CLI before — no longer exists. `quest/gui/autoform_panel.py` remains for rendering a single RPC method's parameters from the manifest. Covered by `tests/test_view_spec.py`, `tests/test_form_model.py`, `tests/test_gui_widget.py` and `tests/test_gui_simulation.py`. |
| LAY-07 | medium | **closed 2026-07-28** | ~~Long-running work has three mechanisms.~~ `quest/tasks.py` owns one: statuses, `progress`, cooperative `should_cancel`, and a `to_dict()` shape all matching `chisurf/server/jobs.py` so a host recognises them — implemented in QuEst, not imported, because a plugin that needs its host to run standalone is not a plugin. `scan_positions` takes a `should_cancel` predicate and a `report_progress` callback (plain callables, so `quest.core` knows nothing about job bookkeeping), `quest.scan` registers a job and returns its `job_id`, and `quest.jobs.list/get/cancel` drive it. **The word was also split:** run artefacts are `quest/runs.py` and `quest.runs.*`; `jobs` now means a running operation, as it does to the host — `quest.jobs.list` would otherwise have handed a ChiSurf caller directories. **Stated limitation:** cancellation is cooperative and a scan's only checkpoint is between sites, so a single `simulate` is effectively atomic — `_simulate_traj` computes a whole trajectory in one `njit` call and a jitted kernel cannot poll a Python flag. `CANCELLING` is a distinct state for exactly that reason. |
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
| DUP-02 | high | **DONE** (2026-07-28) | ~~The capability check asks the wrong question.~~ `HAS_IMP_BFF` now asks both halves of the real question: the compiled `IMP.bff.AV` class **and** an importable `IMP.bff.av.compute_av` that drives it. `hasattr(IMP.bff, "AV")` was the compiled half only, so an environment with a compiled IMP.bff but no imp-tricks reported the backend available and failed on first use. `compute_av` is *imported* rather than probed with `hasattr`, because `IMP.bff.av` is a namespace subpackage spliced in at runtime and an attribute lookup does not answer that question. |
| DUP-03 | medium | **rejected, and re-specced** (2026-07-28) | ~~`fret_rate_trace` belongs with `IMP.bff.cgdye.analysis.fret`, which additionally handles κ².~~ **Both halves of that were wrong.** `calculate_fret_exact` takes an `(nd × na)` distance matrix *plus transition matrices for both dyes*, forms `np.kron(p_d, p_a)` and solves a Markov kinetic ensemble — a rotamer-library formulation, O((nd·na)²), against QuEst's ~10⁵-point clouds. It is not a slower drop-in; it is **a different model at a different scale**. Nor does it *handle* κ²: it accepts a κ² matrix as input. The real kernel, `IMP.bff.cgdye.rotamer.scoring.kappa2_from_vectors`, needs transition dipole vectors, which a structureless point in a volume does not have. **Direction reversed:** QuEst's trajectory kernel is the general one and should go *upstream* — same as `dRmp` under `DUP-04`. What moved instead is the assumption: `fret.kappa2` is now an explicit, validated, documented project field defaulting to the isotropic 2/3. |
| DUP-04 | medium | **DONE** (2026-07-28; `dRmp` contributed upstream 2026-07-28) | **Distance metrics are duplicated.** ⟨R_DA⟩ and ⟨R_DA⟩_E: adopt `random_distances` + `av_pair_statistics` — verified to agree with `RDAMean`/`RDAMeanE` within sampling noise, and nothing blocks it. **`dRmp` goes the other way:** upstream's `r_mp` returns ⟨R_DA⟩ (its own docstring says "same as `r_da_mean` here"), which is a different quantity — 47.65 vs 51.53 Å on 148l E15/E90 — and it cannot be computed from a distance sample at all. QuEst's implementation is the correct one and should be contributed upstream. |
| DUP-05 | low | **mostly done** (2026-07-28) | `density2points` and the contact-volume split are upstream's: `subav` is a thin adapter over `IMP.bff.av._kernels.split_av_acv` and the 93-line `_subav` kernel is deleted. **Neither adoption was a drop-in** — `split_av_acv` carried the float-corner *and* truncate-toward-zero registration defects QuEst had just been fixed for, and returned float64 masks (12.5 MB per site at ng=92 against 1.6 MB); all three were fixed upstream first (imp-tricks `3dbefc8`). Numbers verified unchanged. **Remaining:** `asa`/`spherePoints` → `IMP.cgmol.sterics.asa`, which is cold (once per simulation, for quencher SASA) and not yet adopted. |
| DUP-07 | medium | **partly closed 2026-07-27** | ~~RCSB fetching duplicated with ChiSurf.~~ Moved into `chisurf/core/fio/structure/fetch.py` (merging QuEst's `.cif` fallback into it), re-exported by the `fps_json_editor` plugin, and QuEst delegates to it when a checkout is reachable. **Remaining:** fps.json payload handling — ChiSurf's `payload.py` drags its `FpsJsonModel` along, so moving it to core is the next step. **Not a duplicate after all:** MRC writing — ChiSurf voxelises point clouds via `IMP.em`, QuEst writes precomputed density grids. |
| DUP-06 | ~~low~~ **high** | **DONE** (2026-07-28) | ~~QuEst carries its own PDB/CIF parser and `Structure`.~~ Severity was wrong: this was filed "low — a hazard that has not bitten yet", and it had bitten every multi-chain structure QuEst ever read. The parser half was already IMP's ([specs/structure-io.md](structure-io.md)); the **residue model** was not, and `Structure.residue_dict`/`residue_ids` keyed on `res_id` alone, which restarts per chain. On `mc4r_dimer.cif` all 534 numbers occur in both chains, so each entry held two residues' atoms and `move_center_of_mass` folded both side chains into one CB. Identity now comes from IMP's hierarchy at the reader — a `residue_index` per `(chain, res_id, insertion_code)`, plus the `insertion_code` column the reader had been discarding — and the domain still passes plain numpy arrays, so no IMP object crosses into it. Moved exactly one number: the dimer's CB centroid. Guarded by `tests/test_residue_identity.py`. |

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
