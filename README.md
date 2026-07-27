[![Build Status](https://travis-ci.org/Fluorescence-Tools/quest.svg?branch=master)](https://travis-ci.org/Fluorescence-Tools/ques)

<p align="center">
  <img src="doc/logo.png" alt="QuEst Logo" width="220" />
</p>

# QuEst - Quenching Estimator

## General description

QUEST (QUenching ESTimation) simulates the dynamic quenching of xanthene 
dyes tethers to proteins by flexible linkers by simulating PET and the 
diffusion of dyes (see [Peulen et al., 2017](https://doi.org/10.1021/acs.jpcb.7b03441)).

The dynamic quenching of a fluorescent dye coupled to a protein is 
simulated in three steps:

1. The dye's accessible volume (AV) is calculated, the positions of the 
quenching amino acids are determined, to every quenching amino acid
a quenching rate constant is assigned. 
2. The diffusion of the dye within it's accessible volume is simulated
using Brownian dynamics (BD) simulations. In the BD simulations a
dye that is close to the vicinity of the protein diffuses slower due
to unspecific interactions.
3. The distance between the dye and the quenching amino acids is used
to calculate the dye's fluorescence decay.

![Simulation of dynamic quenching](https://github.com/Fluorescence-Tools/quest/blob/master/doc/img/readme_screenshot_0.png)


In QUEST the dyes are approximated by a sphere diffusing within their 
accessible volume (AV) (see [labellib](https://github.com/Fluorescence-Tools/LabelLib)). 

### Amino-acid specific quenching

PET-quenching is resolved per residue type. Each residue is described by four
quantities, editable per dye in the web UI (*Adv. Settings -> Amino-acid
specific quenching*) or under `amino_acid_interactions` in a project file:

| Field | Meaning |
| --- | --- |
| `kQ` | Quenching rate (1/ns) applied while the dye is in contact. |
| `quench_radius` | Contact radius (Å), measured from the **dye centre** to the residue's quenching centre. `null` inherits the project-wide `critical_distance`. |
| `quench_atoms` | Atom names whose centroid defines the **quenching centre**. |
| `slow_factor` | Diffusion scaling (0–1) modelling unspecific stickiness near the residue. |

The quenching centre sits on the residue's redox-active moiety rather than on
CB: the indole ring for TRP, the phenol ring for TYR, the imidazole ring for
HIS, SD for MET and SG for CYS. For tryptophan this moves the quencher ~3.3 Å
away from CB, which matters at contact distances of a few Ångström.

Quenching rates from residues whose contact spheres overlap **add up**;
stickiness factors **multiply**. Every simulation reports how much each residue
type contributed to the total quenching rate (`quench_rate_fraction` in
`result.json` and `aa_residence.csv`, and the *Quenching by residue type* panel
in the web UI).

The shipped dye presets start from reference PET parameters for a xanthene dye
(TRP > PRO ≈ TYR > MET > HIS > CYS), scaled to each dye's radius. These are
starting values intended to be calibrated against measured lifetimes, not fixed
constants.

The relevant simulation parameters can be adjusted either in a 
graphical user interface `quest_gui` or QuEst can be controlled
using a command line interface (see documentation below).

![Simulation of dynamic quenching](https://github.com/Fluorescence-Tools/quest/blob/master/doc/img/readme_screenshot_3.png)

Alternatively, QuEst can be used a library for potential integration
into other simulations and/or data analysis pipelines (see 
[Jupyter Notebook](https://github.com/Fluorescence-Tools/quest/blob/master/notebooks/quenching_and_fret.ipynb))

## Potential use-cases

* Design of labeling positions for FRET experiments
* Calibration of accessible contact volume 
([ACVs](https://doi.org/10.1016/j.sbi.2016.11.012)) using the 
fluorescence lifetime of the donor

## Installation

### Versions

There are two QuEST versions:
  1. GUI-QuEST a end-user software with graphical user interface for Windows 
  ([setup.exe](https://github.com/Fluorescence-Tools/quest/releases/download/170301/windows_setup_17.03.01.exe), 
  conda), Linux (conda), and macOS (conda). The conda installation is 
  described below. 
  2. Command-QuEST a command line version for Windows, Linux and MacOS

Both versions are documented in the Wiki of this repository 
[Wiki](https://github.com/Fluorescence-Tools/quest/wiki).

### GUI version

The windows GUI version can be installed using either a setup file 
([setup.exe](https://github.com/Fluorescence-Tools/quest/releases/download/170301/windows_setup_17.03.01.exe))
or conda. To install QuEst using conda use the conda repository ``tpeulen``

```bash
conda install -c tpeulen quest
```

Following the installation via conda, quest can be started from 
a command line interface

```bash
quest
```


## Usage

### GUI-QuEST

### Python API (scripts / notebooks)

You can use QuEst as a pure Python library without starting the GUI by
working with the high-level functions provided in the `quest.core` (or
`quest.api`, which re-exports the same objects) module. The typical workflow is:

1. Prepare or save a QuEst project JSON file (e.g. from the GUI).
2. Load the project dictionary via the API.
3. Run the simulation to obtain donor-only and, optionally, donor+FRET
   decay curves.
4. Analyze or plot the returned arrays in standard scientific Python
   environments (e.g. Jupyter notebooks).

Minimal headless example (works in notebooks or scripts):

```python
from pathlib import Path
from quest.core import load_project, simulate_project  # or `from quest.api ...`

project = load_project(Path("my_project.quest.json"))
result = simulate_project(project)

print(result.quantum_yield_donor)
if result.fret_counts is not None:
    print("FRET efficiency:", result.fret_efficiency)
```

The API is designed so that it does not depend on any GUI widgets and
can therefore be safely imported and used in headless environments.

Notebook examples:
- `modules/quest/notebooks/quest_core_headless_demo.ipynb` – minimal headless API usage.
- (Legacy) other notebooks for diffusion/quenching/FRET remain available in the same folder.

### Command-QuEST (Click CLI)

Run QuEst headlessly with the new Click-based CLI:

```bash
# Validate a project file
quest validate -p project.quest.json

# Run a single simulation, overriding parameters
quest simulate -p project.quest.json --set tau0=3.8 --set attachment.residue=42 -o decay.csv

# Sweep multiple values (Cartesian product) and write tagged outputs
quest simulate -p project.quest.json \
  --grid attachment.residue=25,30,35 \
  --grid linker.length=10.0,12.5 \
  -o decay.csv

# Emit a starter template
quest template --with-fret > project.quest.json
```

Notes:
- Use `--set key=value` to override any project field (dotted paths supported).
- Use `--grid key=val1,val2` to sweep multiple values; outputs are tagged per run.
- Output format chosen by extension (.csv or .json).

## Webapp and Electron

QuEst now includes a hostable web surface and a local Electron shell in `webui/`.

### Hostable web app (FastAPI + Next.js)

The frontend needs **Node.js 20.9 or newer** (Next.js 16). The repository ships
a [pixi](https://pixi.sh) manifest that provides a pinned Node toolchain in its
own `frontend` environment, so you do not need Node installed system-wide:

```bash
python -m pip install -e ".[web]"
python -m uvicorn webui.backend.app.main:app --host 127.0.0.1 --port 8000
# new terminal
pixi run -e frontend dev
```

`pixi.toml` and `pixi.lock` are committed, so `pixi install -e frontend`
reproduces the exact toolchain. Available tasks:

| Task | Does |
| --- | --- |
| `pixi run -e frontend install` | `npm install` in `webui/frontend` |
| `pixi run -e frontend dev` | Next.js dev server on :3000 |
| `pixi run -e frontend build` | Production build (+ standalone assets) |
| `pixi run -e frontend start` | Serve the production build |
| `pixi run -e frontend typecheck` | `tsc --noEmit` |
| `pixi run -e frontend node` | Print the environment's Node version |

If you prefer your own Node, the plain npm workflow still works — check
`node -v` first, since an older Node fails at startup with a version error:

```bash
npm --prefix webui/frontend install
npm --prefix webui/frontend run dev
```

For production-style frontend:

```bash
pixi run -e frontend build
pixi run -e frontend start
```

Use `QUEST_BACKEND_URL` to point frontend runtime to a hosted backend target.

### Electron app (local desktop shell)

```bash
npm --prefix webui/desktop install
npm --prefix webui/desktop run start
```

Optional Python override:

```bash
QUEST_PYTHON=/path/to/python npm --prefix webui/desktop run start
```

## Warnings
  1. QuEST determines precise values that are not necessary accurate.
  2. QuEST was the first software to implement the ACVs. ACVs were later described in more detail (see: [COSB2016](https://doi.org/10.1016/j.sbi.2016.11.012). Differencies in the ACV implementation, may produce slightly different results.
  3. QuEST operates on single static structures.
  4. A crude approximation of the dye is used by a sinlge sphere is used.
  5. Specific interactions e.g. binding pockets are not considered.

## Citation
If you have used QuEST in a scientific publication, we would appreciate citations to the following paper: 

[![DOI](https://img.shields.io/badge/DOI-10.1021%2Facs.jpcb.7b03441-blue.svg)](https://doi.org/10.1021/acs.jpcb.7b03441)
> Peulen, T.O., Opanasyuk, O., and Seidel, C.A., 2017. Combining Graphical and Analytical Methods with Molecular Simulations To Analyze Time-Resolved FRET Measurements of Labeled Macromolecules Accurately. The Journal of Physical Chemistry B  2017, 121, 35, 8211-8241 (Feature Article)


For more informations on accessible contact volumes (ACVs) see:

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.sbi.2016.11.012-blue.svg)](https://doi.org/10.1016/j.sbi.2016.11.012)
> Dimura, M., Peulen, T.O., Hanke, C.A., Prakash, A., Gohlke, H. and Seidel, C.A., 2016. Quantitative FRET studies and integrative modeling unravel the structure and dynamics of biomolecular systems. Current opinion in structural biology, 40, pp.163-185.


## Contribute

To improve our dye models we need a larger set of experimental data.
If you are interested in using, and improving experimental coarse-
grained dye models for integrative modelling. Independently if you 
are a developer of not, you can contribute by

* assembling more experimental data
* improve the documentation

If you are interested, sign up on GitHub, contact the developers, and
put a star on this project.

## Info

_Author(s)_: Thomas-Otavio Peulen

_Maintainer_: `tpeulen`

_License_: [MIT](https://mit-license.org/)
This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2 of the License, or (at your option) any later version.
