<!-- QuEst documentation index -->

# QuEst Documentation

QuEst (Quenching Estimation) simulates photo-induced electron transfer (PET) quenching of dyes tethered to proteins via flexible linkers. It provides:

- A GUI for interactive exploration.
- A Click-based CLI (`quest`) for batch/headless runs.
- A lightweight Python API (`quest.core` / `quest.api`) for notebooks and scripts.

This page is the entry point to the QuEst docs and links to detailed references.

## Installation

- **Conda** (recommended):
  ```bash
  conda install -c tpeulen quest
  ```
- **Editable/dev install** (inside this repository):
  ```bash
  pip install -e .
  ```

## Quickstart (headless Python)

```python
from pathlib import Path
from quest.core import load_project, simulate_project

project = load_project(Path("my_project.quest.json"))
result = simulate_project(project, tac_range=(0.0, 50.0))

print("Quantum yield:", result.quantum_yield_donor)
if result.fret_counts is not None:
    print("FRET efficiency:", result.fret_efficiency)
```

Outputs are numpy arrays in `result.time`, `result.donor_counts`, and (when enabled) `result.fret_counts`.

## Quickstart (CLI)

- Validate a project:
  ```bash
  quest validate -p project.quest.json
  ```
- Run a simulation with overrides:
  ```bash
  quest simulate -p project.quest.json --set tau0=3.8 --set attachment.residue=42 -o decay.csv
  ```
- Sweep multiple values (Cartesian grid):
  ```bash
  quest simulate -p project.quest.json \
    --grid attachment.residue=25,30,35 \
    --grid linker.length=10.0,12.5 \
    -o decay.csv
  ```
- Emit a starter template:
  ```bash
  quest template --with-fret > project.quest.json
  ```

See the full CLI reference in [cli.md](cli.md).

## Project file basics

Project files are JSON. Key blocks:

- `tau0`, `kQ`, `n_photons`, `dg`, `slow_fact`, `critical_distance`, `slow_radius`, `sticky_mode`, `save_avs`.
- `attachment`: `{chain, residue, atom}`.
- `quencher`: mapping of residue name to atom list and `kQ` per quencher.
- `fret` (optional): `enabled`, `dyes` (donor first, acceptor second), `R0_matrix`.
- `av_parameter`: `{linker_length, linker_width, radius1}`.
- `pdb`: path to the structure file.

The CLI accepts dotted overrides (e.g., `--set attachment.residue=42`).

## API surface

Core entry points (all re-exported from `quest.api`):

- `load_project(path)` → `dict`
- `build_donor_from_project(project, pdb_path=None)` → configured `DonorDecay`
- `simulate_project(project, pdb_path=None, nbins=None, tac_range=(0.0, 50.0))` → `DecaySimulationResult`

See [api.md](api.md) for detailed signatures and return values.

Notebook example:
- `notebooks/quest_core_headless_demo.ipynb` – headless API usage (load+simulate).

## Parameters reference

Detailed descriptions of GUI/CLI parameters: [parameters.md](parameters.md).

## Outputs

- **Histograms**: CSV/JSON depending on `-o` extension (CLI) or arrays in the API result.
- **Quantum yields**: `result.quantum_yield_donor`, `result.quantum_yield_fret` (when FRET).
- **Collision fraction**: `result.collisions_fraction`.

## Quenching workflow (from legacy manual)

Use this when simulating PET quenching with Brownian dynamics:

1. **Prepare the protein structure**
   - Remove crystallographic water.
   - Protonate/rename atoms with PDB2PQR (standard settings).
   - Open the resulting PQR in PyMOL and save as PDB.
   - Load the PDB in QuEST.
2. **Set dye/linker parameters**
   - Attachment: chain, residue, atom.
   - Linker geometry: `linker_length`, `linker_width`, `radius1`.
   - Accessible volume grid: `dg` (spacing).
   - Dye properties: `tau0` (unquenched lifetime), `D` (diffusion coefficient).
3. **Configure quenching**
   - Quenchers: list of amino-acid names (3-letter codes) and atoms; atoms in the exclude list are ignored.
   - `quench_radius`: distance threshold for quenching.
   - `kQ`: quenching rate constant.
   - `slow_fact`: scaling factor for diffusion near sticks; `critical_distance`/`slow_radius` define where it applies.
4. **Simulation controls**
   - Time: `sim_time`, timestep `dt`.
   - Photons/frames: `n_photons`, `frames`.
   - Histogram: `nbins`, `tac_range`, `skip` (frame stride), output `prefix`.
   - Outputs include simulated fluorescence quantum yield `QY(F)` and collision fraction.
5. **Run**
   - Click **Update** in the GUI or run `quest simulate ...` to generate BD trajectories and fluorescence decays.

See also: [parameters.md](parameters.md) for field-by-field definitions and CLI overrides.

Full legacy manual (with figures): [quenching_manual.md](quenching_manual.md).

## Troubleshooting

- Ensure `pdb` path in the project JSON is valid or pass `--pdb`/`pdb_path`.
- `dyes[0].D` must be set; otherwise `build_donor_from_project` raises a `KeyError`.
- For FRET, provide `fret.enabled: true`, two dyes, and an `R0_matrix` with donor→acceptor entry.
