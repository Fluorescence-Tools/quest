# QuEst CLI reference

The Click-based `quest` command exposes validation, simulation, and template generation. All commands accept `-p/--project` for the project JSON path unless noted.

## Common options

- `-p, --project PATH`: project JSON file.
- `--pdb PATH_OR_ID`: override structure path or load a bare four-character PDB ID from RCSB without editing the project file.
- `--set key=value`: override a project field (dotted paths supported).
- `--grid key=v1,v2,...`: sweep multiple values (Cartesian product across grids).
- `-o, --output PATH`: output file; extension selects format (`.csv` or `.json`).
- `--nbins INT`: override histogram bins (default 4096).
- `--tac-range MIN MAX`: override time axis range.
- `--quiet`: reduce logging.

### Examples

Validate a project:

```bash
quest validate -p project.quest.json
```

Simulate with overrides and CSV output:

```bash
quest simulate -p project.quest.json --set tau0=3.8 --set attachment.residue=42 -o decay.csv
```

Simulate directly from an RCSB PDB ID:

```bash
quest simulate -p project.quest.json --pdb 148l -o decay.csv
```

Sweep residues and linker lengths (tagged outputs):

```bash
quest simulate -p project.quest.json \
  --grid attachment.residue=25,30,35 \
  --grid linker.length=10.0,12.5 \
  -o decay.csv
```

Emit a starter template (with FRET fields):

```bash
quest template --with-fret > project.quest.json
```

Launch the legacy GUI (if available):

```bash
quest gui
```

## Command details

### `quest validate`
Checks that required fields exist (e.g., `tau0`, `kQ`, `n_photons`, `fret.dyes[0].D`, `pdb`) and that numeric values parse.

### `quest simulate`
Runs the quenching (and optional FRET) simulation, writing histogram data to CSV/JSON or printing to stdout. Overrides and grids apply before simulation.

### `quest template`
Prints a minimal project JSON; `--with-fret` includes acceptor and `R0_matrix` placeholders.

### `quest gui`
Starts the GUI if the Qt dependencies are installed.

## Output formats

- **CSV**: columns for time, donor counts, and (when applicable) FRET counts.
- **JSON**: time array, donor histogram, optional FRET histogram, and metadata.

## Troubleshooting

- Provide a valid `pdb` path/PDB ID or pass `--pdb`.
- Ensure donor dye has `D`; otherwise validation fails.
- For FRET runs, include two dyes and `fret.enabled: true`.
