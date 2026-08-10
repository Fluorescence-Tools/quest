---
type: Subsystem
title: Core service layer
description: quest/core/ and the quest.api facade — project loading, simulate_project, DecaySimulationResult, residue-resolved analysis.
resource: quest/core/
tags: [api, service-layer, headless]
status: stable
generated: { by: process:okf-v0.1-import, at: 2026-07-26T00:00:00Z }
stale_after: 2027-02-06
sources:
  - id: quest-core
    resource: ../../quest/core/
    title: quest/core/
    author: human:tpeulen
    last_modified: 2026-07-28
---

# `quest.api` is a facade, not a re-export

Until 2026-07-28 (`LAY-04`) this module forwarded names from `quest.core` and
had no notion of *where* a call runs, so the web backend reached past it into
`quest.core` directly and a host had nothing single to drive. It is now the
supported way in, and every verb goes through a **client**:

```python
from quest import api

project = api.template(with_fret=False)
result  = api.simulate(project, pdb_path="tests/148l.pdb")
```

| Verb | Does |
|---|---|
| `describe()` | The contract: method names, versions, error codes |
| `template(*, with_fret=True)` | A valid project document to start from |
| `validate(project, *, pdb_path=None)` | Raises `ProjectValidationError` |
| `simulate(project, *, pdb_path, nbins, tac_range, mode, ...)` | One run; returns `DecaySimulationResult` |
| `scan(project, positions, *, pdb_path, ...)` | One run per labeling site |
| `parameter_catalog(locale=None)`, `dyes()` | The shipped data, translated |
| `structure_metadata(pdb_path)` | Chains, residues and atoms a dye could attach to |

`client()` returns the active client -- `InProcessClient` by default -- and
`use_client(other)` swaps in a remote one. That indirection is the point: the
CLI, the web backend, the GUI and a ChiSurf host all exercise the same registry,
the same validation and the same error codes, rather than four near-copies.

**Results stay native.** `simulate` hands back a `DecaySimulationResult` with
its numpy arrays intact; serialization happens at a *transport*
(`quest.rpc.contract.to_payload`), not in a handler, because a fitting loop is
now a primary consumer and converting every decay channel to a Python list per
iteration is a tax it would pay forever. `tests/test_contract.py` round-trips
each method's native result through `to_payload` and validates it against the
manifest schema, so payload and contract cannot drift quietly.

**The domain is still importable.** `load_project`, `build_donor_from_project`,
`simulate_project`, `DecaySimulationResult`, `top_quenchers` and the FPS-JSON
helpers remain re-exported from `quest.api`; a notebook that wants them should
have them. The difference is that the verbs are the supported entry point.

# What lives where

`quest/core/` is the domain: Qt-free, host-free, and it takes plain data plus
callables (`should_cancel` is a predicate, not a `Job` -- see [tasks.md](tasks.md)).
Its `__init__` carries the authoritative module table; in summary:

| Module | Owns |
|---|---|
| `quest/core/simulation.py` | The pipeline -- `simulate_project`, `load_project`, `build_donor_from_project`, `DecaySimulationResult`, the residue-resolved analysis, the MRC/grid writers |
| `quest/core/dye_diffusion.py` | `SimulateDiffusion`, `DonorDecay`, the PET chemistry |
| `quest/core/av.py` | Accessible volumes, grid kernels, FRET rate traces |
| `quest/core/photon.py` | Photon-trace and decay-curve kernels (numba) |
| `quest/core/structure/` | `Structure` (`model.py`), reading through IMP (`reader.py`), labeling-site metadata, PDB-ID fetch with an on-disk cache |
| `quest/core/fps_json.py`, `quest/core/io.py` | `fps.json` labeling files; point-cloud xyz output |

Neighbouring modules, reachable directly:

| Module | Owns |
|---|---|
| `quest/project.py` | The project document: `template_project`, `validate_project`, `ProjectValidationError`, `REQUIRED_TOP_LEVEL_KEYS` |
| `quest/scan.py` | One labeling site: `site_project`, `simulate_site` (never raises), `scan_positions` |
| `quest/runs.py` | Past runs: `list_runs`, `run_summary`, `read_run`, `archive_run`, `purge_runs`, `runs_directory`, `is_run_id`. Renamed from `quest/jobs.py` when "job" was reserved for *running* work ([tasks.md](tasks.md)) |
| `quest/tasks.py` | The running work itself: `Job`, `JobManager`, cooperative cancellation ([tasks.md](tasks.md)) |
| `quest/rpc/` | `contract.py` (names, envelopes, error codes), `dispatcher.py` (`ServiceDispatcher`, `InProcessClient`), `services.py` |
| `quest/hosts/` | Optional bridges: `chisurf.py`, and `imp_tricks.enable_imp_tricks()` splicing a checkout into the live interpreter |
| `quest/settings/` | The shipped data: `parameter_catalog()`, `dye_repository()`, `save_dye()`, `quenching_defaults()`, resolved package-relative |
| `quest/i18n.py` | Translations, from one place, for every surface |

None of it touches Qt, so it is safe in notebooks, servers and CI.

# The three calls that matter

| Call | Does |
|---|---|
| `load_project(path)` | JSON → mutable dict. Nothing more; there is no project class |
| `build_donor_from_project(project, pdb_path=None)` | dict → configured `DonorDecay`. This is where the top-level-vs-dye-entry precedence lives |
| `simulate_project(project, …)` | The whole pipeline; returns `DecaySimulationResult` |

`simulate_project` keyword arguments worth knowing: `nbins`, `tac_range`,
`mode`/`n_curves`/`dt_tac`/`n_tac` (see [photon-decay.md](photon-decay.md)),
`project_dir`, `save_outputs` and `render_data`
([architecture/job-artefacts.md](/architecture/job-artefacts.md)).

# `DecaySimulationResult`

A dataclass, not a dict:

- decay — `time`, `donor_counts`, `fret_counts`
- scalars — `quantum_yield_donor`, `quantum_yield_fret`, `fret_efficiency`,
  `collisions_fraction`, `lifetime_donor`, `av_volume`, `contact_volume`,
  `contact_to_free_ratio`
- trajectory — `trajectory_distance_to_mean`, `trajectory_time_ns`,
  `trajectory_autocorr`, `trajectory_autocorr_lag_ns` (thinned for display)
- clouds — `trajectory_points`, `av_points`, `acv_points` (thinned)
- provenance — `project`, `job_id`
- attribution — `aa_residence` ([quenching-model.md](quenching-model.md))

# Analysis helpers in `core/simulation.py`

- `calculate_aa_quenching`, `calculate_aa_residence`, `_residue_type_metrics`,
  `aa_residence_from_model` — the residue-resolved statistics.
- `autocorr_to_tau_window` — autocorrelates the full distance-to-mean series and
  truncates to a `tau0`-scaled lag window (correlate everything, thin only what
  is plotted).
- `sampled_density_grid`, `diffusion_coefficient_grid`, `write_mrc` — the
  voxel outputs.
- `_subsample_points` — the one place display thinning happens.

Verified against the tree on 2026-08-10, mechanically: every symbol named in
this concept was resolved with `hasattr` against its module, and
`DecaySimulationResult`'s dataclass fields were compared to the list above in
both directions — nothing documented is absent and nothing present is
undocumented. `simulate_project` still takes the keyword arguments listed
above. This concept was marked unverified on 2026-08-06 because the tree had
moved under it; it had not in fact rotted.

# FPS-JSON scanning

`quest/core/fps_json.py` reads an `fps.json` labeling file
(`load_fps_json`), normalises its entries to positions
(`normalize_to_fps_positions`) and runs one simulation per site
(`run_fps_scan`). This is what `quest scan` and the web `/api/scan` drive;
they run with `save_outputs=False`.
