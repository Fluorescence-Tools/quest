# QUEST (QUenching ESTimation)

## Overview

The purpose of QUEST (QUenching ESTimation) is to simulate dynamic quenching of xanthene dyes tethered to proteins by flexible linkers by simulating PET and dye diffusion. In QUEST the dye is approximated by a sphere diffusing within its accessible volume (AV). PET quenching by MET, PRO, TYR, and TRP residues is approximated by a step function: if the dye is closer than a threshold distance, it is quenched with the provided rate constant.

Fig. 1 shows the QUEST interface: the input/output region (1, left) and the output region (2 and 3, right). Simulation parameters are set on the left; simulation results are displayed on the right.

![Fig. 1 — QUEST interface](img/quenching_manual/Docmuentaion_QUenching%20ESTimation_01.png)

## Simulation parameters (Tab. 1)

| Parameter | Description |
| --- | --- |
| **PDB** | Open PDB file (protonated; PARSE naming from PDB2PQR). |
| **Sticky** checkbox | If checked, surface (distance to Cβ atoms of quenchers) can be sticky; otherwise diffusion is uniform. |
| **Rs [Å]** | Radius below which AV points are “sticky”; their diffusion is scaled by **slow factor**. |
| **slow factor** | Scaling factor applied to diffusion for AV points within **Rs/slow radius** of stick objects. |
| **Label chain / residue / atom** | Attachment site of the dye. |
| **Length** | Linker length (Å). |
| **Width** | Linker width (Å). |
| **Radius** | Dye radius (Å). |
| **dg** | Grid spacing of the accessible volume. |
| **tau0** | Fluorescence lifetime without quenching. |
| **D** | Diffusion coefficient when not sticking. |
| **Quencher** | Quenching amino acids (3-letter codes, space-separated). |
| **Quench radius** | Distance threshold for quenching. |
| **kQ** | Quenching rate constant. |
| **Exclude atoms** | Atom names to ignore as quenching atoms. |
| **Sim time** | Brownian dynamics simulation length. |
| **dt** | BD time step. |
| **nPhotons** | Number of simulated photons. |
| **Frames** | Number of simulated frames. |
| **QY(F)** | Simulated fluorescence quantum yield. |
| **collided** | Fraction of dyes that collided with quenching amino acids. |
| **nBins** | Bins of the fluorescence decay histogram. |
| **range** | Time range (ns) of the decay histogram. |
| **skip** | Frames skipped when saving/visualizing BD simulations. |
| **Prefix** | Prefix for AV filenames when saved. |

## Example workflow

Before simulating fluorescence decays and fluorescence quenching, ensure the structure is in an appropriate format and crystal water is removed (it can otherwise block accessible surface). For example, PDB 172L of T4 lysozyme contains crystal water that should be removed to expose accessible quenching sites.

1. Remove crystallographic water from the PDB.  
2. Protonate and rename atoms using PDB2PQR (PARSE naming): <http://nbcr-222.ucsd.edu/pdb2pqr_2.0.0/>.  
3. Save the resulting PQR as PDB (e.g., via PyMOL).  
4. Open the PDB in QUEST, set parameters, then click **update** to calculate the BD trajectory and fluorescence decay.  

![Fig. 2 — T4 lysozyme with crystal water](img/quenching_manual/Docmuentaion_QUenching%20ESTimation_03.png)

![Fig. 3 — Steps to simulate decay (load PDB, then update)](img/quenching_manual/Docmuentaion_QUenching%20ESTimation_02.png)

Additional extracted assets:
- Vector original: `img/quenching_manual/Docmuentaion_QUenching ESTimation_02.emf`
