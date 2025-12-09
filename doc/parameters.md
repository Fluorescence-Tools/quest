# QuEst parameters reference

This page summarizes the project JSON keys and their meaning. The same fields are surfaced in the GUI, CLI overrides (`--set key=value`, `--grid key=...`) and the Python API.

## Project-level fields

| Key | Description |
| --- | --- |
| `pdb` | Path to the structure file used for AV calculation. |
| `tau0` | Fluorescence lifetime in the absence of quenching (ns). |
| `kQ` | Default quenching rate constant (1/ns) applied to quencher residues. |
| `n_photons` | Number of simulated photons. |
| `dg` | Grid resolution for dye diffusion. |
| `slow_fact` | Factor by which the dye diffusion slows near the surface. |
| `critical_distance` | Distance threshold for counting collisions/quenching. |
| `slow_radius` | Interaction radius (Å) within which the dye experiences slowed diffusion. |
| `sticky_mode` | `"surface"` to slow near all surface atoms, `"quencher"` to slow only near quencher residues. |
| `save_avs` | Whether to persist accessible volume grids. |
| `output_file` | Base name for saved outputs. |
| `n_bins` | Histogram bin count for decay outputs (CLI/API default: 4096 if omitted). |

## Attachment block

```json
"attachment": {
  "chain": "A",
  "residue": 42,
  "atom": "CA"
}
```

Defines where the donor dye is tethered.

## Accessible volume

```json
"av_parameter": {
  "linker_length": 20.0,
  "linker_width": 0.5,
  "radius1": 5.0
}
```

- `linker_length`: Maximum distance from attachment point to dye center.
- `linker_width`: Width of the linker.
- `radius1`: Dye radius (single-sphere approximation).

## Dye block (FRET dyes list)

Located under `fret.dyes`. The first entry is the donor; the second (optional) is the acceptor.

| Key | Description |
| --- | --- |
| `D` | Diffusion coefficient of the dye when not interacting with the surface (Å²/ns). **Required for donor.** |
| `attachment` | Same structure as the top-level `attachment` (chain/residue/atom). |
| `av_parameter` | Optional AV parameters for acceptor (when FRET). |

## Quencher block

```json
"quencher": {
  "TRP": ["CB"],
  "TYR": ["CB"],
  "HIS": ["CB"]
}
```

- Keys are quenching residue names (3-letter codes).
- Values list atom names that act as quenchers.
- `kQ` at the project root sets the rate constant applied when within `critical_distance`.
- `all_quencher_atoms` (bool, default `true`): use all atoms of quencher residues if `true`, otherwise only the listed atoms.

## FRET block (optional)

```json
"fret": {
  "enabled": true,
  "dyes": [... donor, acceptor ...],
  "R0_matrix": [[null, 52.0], [52.0, null]]
}
```

- `enabled`: turn FRET on/off.
- `dyes`: list of dye entries (donor first). Provide acceptor when FRET is enabled.
- `R0_matrix`: Förster distances (nm); donor→acceptor entry is used (row 0, col 1).

## Simulation controls

| Key | Description |
| --- | --- |
| `t_max` | Total simulation time (µs). |
| `t_step` | Time step (ps) for Brownian dynamics. |
| `frames` | Derived: total number of frames (from `t_max` and `t_step`). |
| `n_photons` | Number of photons generated. |
| `show_av` | Show full AV in 3D illustration (GUI). |

## Results fields (outputs)

| Field | Meaning |
| --- | --- |
| `QY(F)` | Simulated fluorescence quantum yield of the bright species. |
| `collisions_fraction` | Fraction of frames where dye is within quench radius. |
| `nBins` / `n_bins` | Bin count used for decay histogram. |
| `range` | Time axis range for decay histogram (tac_range in API/CLI). |
| `skip` | GUI-only: subsampling factor for 3D illustration frames. |

## CLI override examples

- Set donor lifetime and attachment residue:
  ```bash
  quest simulate -p project.quest.json --set tau0=3.8 --set attachment.residue=55
  ```
- Sweep linker length and residue:
  ```bash
  quest simulate -p project.quest.json \
    --grid linker.length=10.0,12.5 \
    --grid attachment.residue=25,30,35
  ```

## Validation tips

- Ensure donor dye has `D` set; otherwise the API raises a `KeyError`.
- Provide a valid `pdb` path (or pass `--pdb`/`pdb_path`).
- When enabling FRET, include two dyes and an `R0_matrix` entry for donor→acceptor.
