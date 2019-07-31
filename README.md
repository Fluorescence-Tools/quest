# QuEST

## General description

QUEST (QUenching ESTimation) simulates the dynamic quenching of xanthene dyes tethers to proteins by flexible linkers by simulating PET and the diffusion of dyes. 

In QUEST the dyes are approximated by a sphere diffusing within their accessible volume (AV) (see [labellib](https://github.com/Fluorescence-Tools/LabelLib)). 

PET-quenching of the dye by MET, PRO, TYR and TRP residues is approximated by a step function where the dye is quenched with a provided rate contestant if it is closer than a given threshold distance.

## Potential use-cases

* Design of labeling positions for FRET experiments
* Calibration of accessible contact volume ([ACVs](https://doi.org/10.1016/j.sbi.2016.11.012)) using the fluorescence lifetime of the donor

# Building and installation

There are two QuEST versions:
  1. GUI-QuEST a end-user software with graphical user interface for Windows 
  2. Command-QuEST a command line version for Windows, Linux and MacOS

Both versions are documented in the Wiki of this repository [Wiki](https://github.com/Fluorescence-Tools/quest/wiki).

### Windows GUI version

The windows GUI version can be installed using the setup files (see [releases](https://github.com/Fluorescence-Tools/quest/releases)).

## Warnings
  1. QuEST can be used to estimate dynamic quenching, e.g., to plan the labeling scheme for FRET measurements. QuEST provides precise estimates that are not necessary accurate.
  2. QuEST is the first software that implemented the accessible contact volume that was later described in more detail in the publication [COSB2016](https://doi.org/10.1016/j.sbi.2016.11.012). Due to differencies in the ACV implementation, QuEST may produce slightly different results (not quantified)
  3. QuEST operates on single structures and assumes that the structure is fixed while the dye samples its conformational space
  4. A crude approximation of the dye is used. Specific interactions e.g. binding pockets are not considered.

## Citation
If you have used QuEST in a scientific publication, we would appreciate citations to the following paper: 

[![DOI for citing QuEST](https://img.shields.io/badge/https://doi.org/10.1021/acs.jpcb.7b03441-blue.svg)](https://pubs.acs.org/doi/abs/10.1021/acs.jpcb.7b03441)
> Peulen, T.O., Opanasyuk, O., and Seidel, C.A., 2017. Combining Graphical and Analytical Methods with Molecular Simulations To Analyze Time-Resolved FRET Measurements of Labeled Macromolecules Accurately. The Journal of Physical Chemistry B  2017, 121, 35, 8211-8241 (Feature Article)


For more informations on ACVs see:

[![DOI for citing LabelLib](https://img.shields.io/badge/DOI-10.1016%2Fj.sbi.2016.11.012-blue.svg)](https://doi.org/10.1016/j.sbi.2016.11.012)
> Dimura, M., Peulen, T.O., Hanke, C.A., Prakash, A., Gohlke, H. and Seidel, C.A., 2016. Quantitative FRET studies and integrative modeling unravel the structure and dynamics of biomolecular systems. Current opinion in structural biology, 40, pp.163-185.
