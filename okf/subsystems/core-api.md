---
type: Subsystem
title: Core service layer
description: quest/core.py and the quest.api re-export — project loading, simulate_project, DecaySimulationResult, residue-resolved analysis.
resource: quest/core.py
tags: [api, service-layer, headless]
timestamp: '2026-07-26T00:00:00Z'
---

# The public surface

```python
from quest.api import (
    load_project, build_donor_from_project, simulate_project,
    DecaySimulationResult, top_quenchers,
    template_project, validate_project, ProjectValidationError,
    is_pdb_id, download_pdb_id, resolve_structure_path,
    load_fps_json, normalize_to_fps_positions, run_fps_scan,
)
```

Neighbouring domain modules, reachable directly:

| Module | Owns |
|---|---|
| `quest/project.py` | The project document: `template_project`, `validate_project`, `ProjectValidationError`, `REQUIRED_TOP_LEVEL_KEYS` |
| `quest/scan.py` | One labeling site: `site_project`, `simulate_site` (never raises), `scan_positions` |
| `quest/settings/` | The shipped data: `parameter_catalog()`, `dye_repository()`, `save_dye()`, `quenching_defaults()`, resolved package-relative |
| `quest/core/structure/metadata.py` | `structure_metadata(path)` — chains, residues and atom names a dye could attach to (PDB and mmCIF) |
| `quest/jobs.py` | Past runs: `list_jobs`, `job_summary`, `read_job`, `archive_job`, and `is_job_id` (a run is a UUID directory) |
| `quest/hosts/imp_tricks.py` | `enable_imp_tricks()` — splice an imp-tricks checkout into the live interpreter |
| `quest/rpc/` | The RPC surface: `ServiceDispatcher`, `InProcessClient`, `register_services`, `result_payload` |

`quest/api.py` is a re-export shim; the implementation is `quest/core.py`
(plus `core/structure/fetch.py` and `core/fps_json.py`). Import from `quest.api` in
new code — see [architecture/surfaces.md](/architecture/surfaces.md).

None of it touches Qt, so it is safe in notebooks, servers and CI.

# The three calls that matter

| Call | Does |
|---|---|
| `load_project(path)` | JSON → mutable dict. Nothing more; there is no project class |
| `build_donor_from_project(project, pdb_path=None)` | dict → configured `DonorDecay`. This is where the top-level-vs-dye-entry precedence lives |
| `simulate_project(project, …)` | The whole pipeline; returns `DecaySimulationResult` |

`simulate_project` keyword arguments worth knowing: `nbins`, `tac_range`,
`mode`/`n_curves`/`dt_tac`/`n_tac` (see [photon-decay.md](photon-decay.md)),
`project_dir`, and `save_outputs`
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

# Analysis helpers in `core.py`

- `calculate_aa_quenching`, `calculate_aa_residence`, `_residue_type_metrics`,
  `aa_residence_from_model` — the residue-resolved statistics.
- `autocorr_to_tau_window` — autocorrelates the full distance-to-mean series and
  truncates to a `tau0`-scaled lag window (correlate everything, thin only what
  is plotted).
- `sampled_density_grid`, `diffusion_coefficient_grid`, `write_mrc` — the
  voxel outputs.
- `_subsample_points` — the one place display thinning happens.

# FPS-JSON scanning

`quest/lib/fps_json.py` reads an `fps.json` labeling file
(`load_fps_json`), normalises its entries to positions
(`normalize_to_fps_positions`) and runs one simulation per site
(`run_fps_scan`). This is what `quest scan` and the web `/api/scan` drive;
they run with `save_outputs=False`.
