# QuEst parameters reference

This page is generated from `parameter_catalog.json`. Update those first when changing parameter help text — the web UI, this page and the desktop form all read them.

## Catalog summary

| Key | Category | Type | Unit | Description |
| --- | --- | --- | --- | --- |
| `amino_acid_interactions` | Quenching | table | - | Per-residue-type interaction table. Each entry holds slow_factor (0-1, how much the residue slows the diffusing dye), kQ (quenching rate in 1/ns applied on contact), quench_radius (contact radius in Angstrom from the dye centre; null inherits critical_distance) and quench_atoms (atom names whose centroid defines the quenching centre). Only residues with a redox-active side chain quench by default: TRP, TYR, MET, HIS, CYS and PRO. |
| `amino_acid_interactions.kQ` | Quenching | number | 1/ns | Rate at which this residue type quenches the dye while the two are in contact. Contributions from residues whose contact spheres overlap add up. |
| `amino_acid_interactions.quench_atoms` | Quenching | list | - | Atom names whose centroid defines this residue's quenching centre. The defaults follow the redox-active moiety that mediates photo-induced electron transfer: the indole ring for TRP, the phenol ring for TYR, the imidazole ring for HIS, SD for MET and SG for CYS - not CB. |
| `amino_acid_interactions.quench_radius` | Quenching | number | Å | Distance from the dye centre to this residue's quenching centre within which quenching occurs. Leave empty to inherit the project-wide critical_distance. |
| `amino_acid_interactions.slow_factor` | Quenching | number | - | Factor between 0 and 1 scaling the dye's diffusion coefficient near this residue type, modelling unspecific stickiness. Overlapping residues multiply their factors. |
| `attachment.atom` | Labeling site | text | - | Atom name (e.g., CB, CA) used as the tethering point for the dye. |
| `attachment.chain` | Labeling site | text | - | Chain identifier of the amino acid residue where the dye is tethered. |
| `attachment.residue` | Labeling site | integer | - | Residue sequence number of the amino acid where the dye is tethered. |
| `av_parameter.linker_length` | Accessible volume | number | Å | Maximum distance from attachment point to the center of the dye. |
| `av_parameter.linker_width` | Accessible volume | number | Å | Width of the linker tethering the dye. |
| `av_parameter.radius1` | Accessible volume | number | Å | Exclusion radius representing the dye sphere (single-sphere approximation). |
| `coarse_grained` | Simulation | boolean | - | Reduce the structure to backbone plus CB before simulating, with CB moved to the side-chain centre of mass. Side-chain atoms are then absent, so every residue's quenching centre falls back to that CB pseudo-atom, and the accessible volume grows because side chains no longer block it. This is a modelling choice, not a speed optimisation: QuEst's cost is dominated by the AV grid rather than the atom count. |
| `critical_distance` | Quenching | number | Å | Contact radius used by residue types that do not define their own quench_radius. It is measured from the dye centre to the residue's quenching centre, so it must exceed the dye radius plus van der Waals contact (typically 7-9 A for a 3.5 A dye). |
| `dg` | Accessible volume | number | Å | Grid resolution used to compute the accessible volume (AV) and diffusion search space. |
| `fret.R0_matrix` | FRET Acceptor | number | Å | Förster radius R0 of the donor–acceptor pair, in Ångström like every other length in a project. Stored as the donor→acceptor cell of fret.R0_matrix; the transfer rate scales as (R0/r)^6, so entering nanometres makes FRET vanish. Typical xanthene pairs are 45–65 Å. |
| `fret.acceptor_dynamics` | FRET Acceptor | choice | - | How the acceptor is treated: 'trajectory' simulates its Brownian walk with its own diffusion coefficient and stickiness and uses the instantaneous donor-acceptor distance, 'averaged' averages over its static accessible volume (the fast-acceptor limit, valid only when it explores that volume quickly compared with the donor lifetime). The two disagree because a cloud is weighted uniformly while a trajectory is weighted by where the dye dwells. |
| `fret.dyes.0.D` | Simulation | number | Å²/ns | Diffusion coefficient of the donor dye in solution, when not interacting with the protein surface. |
| `fret.dyes.1.D` | FRET Acceptor | number | Å²/ns | Diffusion coefficient of the FRET acceptor dye when not interacting with the protein surface. |
| `fret.dyes.1.amino_acid_interactions` | FRET Acceptor | table | - | Per-residue-type interaction table for the acceptor, same shape as the donor's amino_acid_interactions. Only slow_factor is used, since QuEst reports the donor's decay and the acceptor's own PET quenching would change how bright the acceptor is, not the donor. Defaults to the donor's table. |
| `fret.dyes.1.attachment.atom` | FRET Acceptor | text | - | Atom name where the FRET acceptor dye is attached. |
| `fret.dyes.1.attachment.chain` | FRET Acceptor | text | - | Chain identifier of the residue where the FRET acceptor dye is attached. |
| `fret.dyes.1.attachment.residue` | FRET Acceptor | integer | - | Residue sequence number where the FRET acceptor dye is attached. |
| `fret.dyes.1.av_parameter.linker_length` | FRET Acceptor | number | Å | Maximum distance from attachment point to the center of the FRET acceptor dye. |
| `fret.dyes.1.av_parameter.linker_width` | FRET Acceptor | number | Å | Width of the linker tethering the FRET acceptor dye. |
| `fret.dyes.1.av_parameter.radius1` | FRET Acceptor | number | Å | Exclusion radius representing the FRET acceptor dye sphere. |
| `fret.dyes.1.slow_radius` | FRET Acceptor | number | Å | Radius around each amino-acid centre within which that residue slows the acceptor. Separate from the donor's: a cyanine and a rhodamine do not adhere to a protein surface in the same way. Defaults to the donor's value. |
| `fret.enabled` | FRET Acceptor | boolean | - | Whether a FRET acceptor dye is simulated to compute FRET efficiency and decay curves. |
| `fret.kappa2` | FRET Acceptor | number | - | Orientation factor κ² of the transfer, between 0 (perpendicular transition dipoles) and 4 (collinear). QuEst does not model it: the dye is a point diffusing in a volume and has no dipole to orient, so this is the averaging regime you are asserting. The default 2/3 is the isotropic dynamic average, which is what a published Förster radius already assumes — leaving it there reproduces the rate exactly. Raise it only with evidence that both dyes are orientationally restricted. |
| `n_bins` | Simulation | integer | - | Number of time channels/bins used to construct the output decay histogram. |
| `n_photons` | Simulation | integer | - | Number of excitation photons simulated to reconstruct the fluorescence decay histogram. |
| `output_file` | Advanced | text | - | Base filename prefix used when saving simulation outputs, trajectories, and decays. |
| `parallel_trajectories` | Advanced | integer | - | Number of independent Brownian dynamics trajectories run concurrently and combined before photon sampling. Use -1 to use all CPU cores, capped at 8. |
| `pdb` | Structure | file | - | Path to the structure file (PDB/CIF), or a bare four-character RCSB PDB ID used for AV calculation. |
| `random_seed` | Advanced | integer | - | Seed for the Monte-Carlo simulation. When set, both the diffusion trajectory and the photon trace are seeded, so repeated runs of the same project reproduce the same decay and quantum yield exactly. Leave unset for an independent random run; the photon trace then uses a faster multi-threaded kernel. |
| `save_avs` | Advanced | boolean | - | Whether to save the calculated accessible volume grids as XYZ files. |
| `skip_frame` | Advanced | integer | - | Subsampling factor for writing trajectory frames to coordinates files. |
| `slow_radius` | Advanced | number | Å | Radius around each whole-protein amino-acid center where that residue type contributes its slow_factor to the diffusion coefficient grid. |
| `t_max` | Simulation | number | ns | Total length of the simulated Brownian dynamics trajectory. |
| `t_step` | Simulation | number | ns | Time step size for the Brownian dynamics trajectory calculation. |
| `tau0` | Simulation | number | ns | Fluorescence lifetime of the donor dye in the absence of quenching. |

## Structure

### `pdb`

**Label:** Structure file  
**Type:** `file`

Path to the structure file (PDB/CIF), or a bare four-character RCSB PDB ID used for AV calculation.

## Labeling site

### `attachment.atom`

**Label:** Attachment atom  
**Type:** `text`

Atom name (e.g., CB, CA) used as the tethering point for the dye.

### `attachment.chain`

**Label:** Attachment chain  
**Type:** `text`

Chain identifier of the amino acid residue where the dye is tethered.

### `attachment.residue`

**Label:** Attachment residue  
**Type:** `integer`

Residue sequence number of the amino acid where the dye is tethered.

## Accessible volume

### `av_parameter.linker_length`

**Label:** Linker length  
**Type:** `number`
**Unit:** `Å`

Maximum distance from attachment point to the center of the dye.

### `av_parameter.linker_width`

**Label:** Linker width  
**Type:** `number`
**Unit:** `Å`

Width of the linker tethering the dye.

### `av_parameter.radius1`

**Label:** Dye radius  
**Type:** `number`
**Unit:** `Å`

Exclusion radius representing the dye sphere (single-sphere approximation).

### `dg`

**Label:** Grid resolution  
**Type:** `number`
**Unit:** `Å`

Grid resolution used to compute the accessible volume (AV) and diffusion search space.

## Simulation

### `coarse_grained`

**Label:** Coarse-grained structure  
**Type:** `boolean`

Reduce the structure to backbone plus CB before simulating, with CB moved to the side-chain centre of mass. Side-chain atoms are then absent, so every residue's quenching centre falls back to that CB pseudo-atom, and the accessible volume grows because side chains no longer block it. This is a modelling choice, not a speed optimisation: QuEst's cost is dominated by the AV grid rather than the atom count.

### `fret.dyes.0.D`

**Label:** Donor diffusion coefficient  
**Type:** `number`
**Unit:** `Å²/ns`

Diffusion coefficient of the donor dye in solution, when not interacting with the protein surface.

### `n_bins`

**Label:** Decay bins  
**Type:** `integer`

Number of time channels/bins used to construct the output decay histogram.

### `n_photons`

**Label:** Simulated photons  
**Type:** `integer`

Number of excitation photons simulated to reconstruct the fluorescence decay histogram.

### `t_max`

**Label:** Simulation time  
**Type:** `number`
**Unit:** `ns`

Total length of the simulated Brownian dynamics trajectory.

### `t_step`

**Label:** Time step  
**Type:** `number`
**Unit:** `ns`

Time step size for the Brownian dynamics trajectory calculation.

### `tau0`

**Label:** Unquenched lifetime  
**Type:** `number`
**Unit:** `ns`

Fluorescence lifetime of the donor dye in the absence of quenching.

## Advanced

### `output_file`

**Label:** Output file prefix  
**Type:** `text`

Base filename prefix used when saving simulation outputs, trajectories, and decays.

### `parallel_trajectories`

**Label:** Parallel trajectories  
**Type:** `integer`

Number of independent Brownian dynamics trajectories run concurrently and combined before photon sampling. Use -1 to use all CPU cores, capped at 8.

### `random_seed`

**Label:** Random seed  
**Type:** `integer`

Seed for the Monte-Carlo simulation. When set, both the diffusion trajectory and the photon trace are seeded, so repeated runs of the same project reproduce the same decay and quantum yield exactly. Leave unset for an independent random run; the photon trace then uses a faster multi-threaded kernel.

### `save_avs`

**Label:** Save AV files  
**Type:** `boolean`

Whether to save the calculated accessible volume grids as XYZ files.

### `skip_frame`

**Label:** Trajectory skip frames  
**Type:** `integer`

Subsampling factor for writing trajectory frames to coordinates files.

### `slow_radius`

**Label:** Slowing radius  
**Type:** `number`
**Unit:** `Å`

Radius around each whole-protein amino-acid center where that residue type contributes its slow_factor to the diffusion coefficient grid.

## FRET Acceptor

### `fret.R0_matrix`

**Label:** Förster radius  
**Type:** `number`
**Unit:** `Å`

Förster radius R0 of the donor–acceptor pair, in Ångström like every other length in a project. Stored as the donor→acceptor cell of fret.R0_matrix; the transfer rate scales as (R0/r)^6, so entering nanometres makes FRET vanish. Typical xanthene pairs are 45–65 Å.

### `fret.acceptor_dynamics`

**Label:** Acceptor dynamics  
**Type:** `choice`

How the acceptor is treated: 'trajectory' simulates its Brownian walk with its own diffusion coefficient and stickiness and uses the instantaneous donor-acceptor distance, 'averaged' averages over its static accessible volume (the fast-acceptor limit, valid only when it explores that volume quickly compared with the donor lifetime). The two disagree because a cloud is weighted uniformly while a trajectory is weighted by where the dye dwells.

### `fret.dyes.1.D`

**Label:** Acceptor diffusion coefficient  
**Type:** `number`
**Unit:** `Å²/ns`

Diffusion coefficient of the FRET acceptor dye when not interacting with the protein surface.

### `fret.dyes.1.amino_acid_interactions`

**Label:** Acceptor interactions  
**Type:** `table`

Per-residue-type interaction table for the acceptor, same shape as the donor's amino_acid_interactions. Only slow_factor is used, since QuEst reports the donor's decay and the acceptor's own PET quenching would change how bright the acceptor is, not the donor. Defaults to the donor's table.

### `fret.dyes.1.attachment.atom`

**Label:** Acceptor atom  
**Type:** `text`

Atom name where the FRET acceptor dye is attached.

### `fret.dyes.1.attachment.chain`

**Label:** Acceptor chain  
**Type:** `text`

Chain identifier of the residue where the FRET acceptor dye is attached.

### `fret.dyes.1.attachment.residue`

**Label:** Acceptor residue  
**Type:** `integer`

Residue sequence number where the FRET acceptor dye is attached.

### `fret.dyes.1.av_parameter.linker_length`

**Label:** Acceptor linker length  
**Type:** `number`
**Unit:** `Å`

Maximum distance from attachment point to the center of the FRET acceptor dye.

### `fret.dyes.1.av_parameter.linker_width`

**Label:** Acceptor linker width  
**Type:** `number`
**Unit:** `Å`

Width of the linker tethering the FRET acceptor dye.

### `fret.dyes.1.av_parameter.radius1`

**Label:** Acceptor dye radius  
**Type:** `number`
**Unit:** `Å`

Exclusion radius representing the FRET acceptor dye sphere.

### `fret.dyes.1.slow_radius`

**Label:** Acceptor slowing radius  
**Type:** `number`
**Unit:** `Å`

Radius around each amino-acid centre within which that residue slows the acceptor. Separate from the donor's: a cyanine and a rhodamine do not adhere to a protein surface in the same way. Defaults to the donor's value.

### `fret.enabled`

**Label:** FRET enabled  
**Type:** `boolean`

Whether a FRET acceptor dye is simulated to compute FRET efficiency and decay curves.

### `fret.kappa2`

**Label:** Orientation factor κ²  
**Type:** `number`

Orientation factor κ² of the transfer, between 0 (perpendicular transition dipoles) and 4 (collinear). QuEst does not model it: the dye is a point diffusing in a volume and has no dipole to orient, so this is the averaging regime you are asserting. The default 2/3 is the isotropic dynamic average, which is what a published Förster radius already assumes — leaving it there reproduces the rate exactly. Raise it only with evidence that both dyes are orientationally restricted.

## Quenching

### `amino_acid_interactions`

**Label:** Amino-acid interactions  
**Type:** `table`

Per-residue-type interaction table. Each entry holds slow_factor (0-1, how much the residue slows the diffusing dye), kQ (quenching rate in 1/ns applied on contact), quench_radius (contact radius in Angstrom from the dye centre; null inherits critical_distance) and quench_atoms (atom names whose centroid defines the quenching centre). Only residues with a redox-active side chain quench by default: TRP, TYR, MET, HIS, CYS and PRO.

### `amino_acid_interactions.kQ`

**Label:** Quenching rate kQ  
**Type:** `number`
**Unit:** `1/ns`

Rate at which this residue type quenches the dye while the two are in contact. Contributions from residues whose contact spheres overlap add up.

### `amino_acid_interactions.quench_atoms`

**Label:** Quenching centre atoms  
**Type:** `list`

Atom names whose centroid defines this residue's quenching centre. The defaults follow the redox-active moiety that mediates photo-induced electron transfer: the indole ring for TRP, the phenol ring for TYR, the imidazole ring for HIS, SD for MET and SG for CYS - not CB.

### `amino_acid_interactions.quench_radius`

**Label:** Contact radius  
**Type:** `number`
**Unit:** `Å`

Distance from the dye centre to this residue's quenching centre within which quenching occurs. Leave empty to inherit the project-wide critical_distance.

### `amino_acid_interactions.slow_factor`

**Label:** Slow factor  
**Type:** `number`

Factor between 0 and 1 scaling the dye's diffusion coefficient near this residue type, modelling unspecific stickiness. Overlapping residues multiply their factors.

### `critical_distance`

**Label:** Fallback contact radius  
**Type:** `number`
**Unit:** `Å`

Contact radius used by residue types that do not define their own quench_radius. It is measured from the dye centre to the residue's quenching centre, so it must exceed the dye radius plus van der Waals contact (typically 7-9 A for a 3.5 A dye).

## CLI override examples

- Set donor lifetime and attachment residue:
  ```bash
  quest simulate -p project.quest.json --set tau0=3.8 --set attachment.residue=55
  ```
- Sweep linker length and residue:
  ```bash
  quest simulate -p project.quest.json \
    --grid av_parameter.linker_length=10.0,12.5 \
    --grid attachment.residue=25,30,35
  ```

## Validation tips

- Ensure donor dye has `D` set; otherwise the API raises a `KeyError`.
- Provide a valid `pdb` path (or pass `--pdb`/`pdb_path`).
- When enabling FRET, include two dyes and an `R0_matrix` entry for donor→acceptor.
