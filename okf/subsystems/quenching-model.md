---
type: Subsystem
title: Quenching model
description: Residue-type-specific PET quenching — contact radii, quenching centres, additive rates, multiplicative stickiness, and per-type attribution.
resource: quest/lib/tools/dye_diffusion/dye_diffusion.py
tags: [pet, quenching, amino-acids, physics]
timestamp: '2026-07-26T00:00:00Z'
---

# Four numbers per residue type

Quenching is resolved per residue *type*, through the
`amino_acid_interactions` table in the project (editable per dye in the web UI
under *Adv. Settings -> Amino-acid specific quenching*):

| Field | Meaning |
| --- | --- |
| `kQ` | Quenching rate, 1/ns, applied while the dye is in contact |
| `quench_radius` | Contact radius, Å, from the **dye centre** to the residue's quenching centre; `null` inherits the project-wide `critical_distance` |
| `quench_atoms` | Atom names whose centroid defines the **quenching centre** |
| `slow_factor` | Diffusion scaling in [0, 1] — unspecific stickiness near the residue |

Defaults live in `dye_diffusion.py` as `DEFAULT_QUENCHER_ATOMS`,
`REFERENCE_PET_QUENCHING` and `default_amino_acid_interactions()`.

# The quenching centre is the redox-active moiety, not CB

`DEFAULT_QUENCHER_ATOMS` puts the quencher on the group that actually mediates
electron transfer: the indole ring for TRP, the phenol ring for TYR, the
imidazole ring for HIS, SD for MET, SG for CYS. For tryptophan this moves the
centre ~3.3 Å away from CB — at contact distances of a few Å that is not a
detail. Residues with no known PET-active group fall back to a side-chain stub
so a user-supplied rate still has a well-defined centre.

Stickiness and quenching therefore have **different centres**: stickiness stays
on the residue's slow centres, quenching sits on the moiety. `AV.calc_slow_av`
takes `quench_centers` separately from `slow_centers` for exactly this reason.

# Contact distances are stored relative to the dye *surface*

`REFERENCE_PET_QUENCHING` records `contact_distance` from the dye **surface**
(≈ van der Waals contact plus the offset from the moiety centroid to its outer
atoms). `default_amino_acid_interactions(dye_radius=…)` adds the dye radius to
get the centre-to-centre `quench_radius` the simulation uses. That is what makes
one reference table transferable across dyes of different size — a `kQ_scale`
handles how easily a given dye is reduced or oxidised.

Reference ordering for a xanthene (Alexa488-like) dye: TRP > PRO ≈ TYR > MET >
HIS > CYS.

# Composition rules

- **Quenching rates add.** Where contact spheres overlap, the voxel's
  `k_quench` is the sum of the contributing residues' rates.
- **Stickiness multiplies.** Overlapping slow centres multiply their factors, so
  a dye in a crowded pocket slows down more than near a single residue.

Both are baked into the grids in `AV.calc_slow_av` (`_additive_factor_grid` for
rates, `_slow_factor_grid` for stickiness), not recomputed per frame.

# Attribution: who did the quenching

`quest/core.py` walks the finished trajectory once and reports per residue type
(`aa_residence`, in `result.json` and `aa_residence.csv`):

- residence — `frames_near`, `fraction_near`, `total_time_ns`, event count,
  mean/max event duration, mean minimum distance;
- an **attributed** variant of the same statistics, where each frame is assigned
  to a single residue type (`_attribute_frames_to_residue_type`) so overlapping
  contacts are not double-counted;
- quenching — `mean_quench_rate`, `fraction_frames_quenched`, and
  `quench_rate_fraction`, the share of the total quenching rate contributed by
  that type.

`top_quenchers(aa_residence, limit=3)` ranks by `quench_rate_fraction`; the CLI
scan and the web UI both surface the dominant quencher from it.

Both the plain and attributed statistics exist on purpose. `frames_near` answers
"how much time did the dye spend near a TRP", counting a frame once per type;
the attributed variant answers "which residue type owned this frame", and the
attributed fractions sum to at most one.

# Where the kernels are

`_residue_type_contact_stats` and `_attribute_frames_to_residue_type` in
`quest/core.py` are numba kernels parallelised **over residue types**, so each
thread owns its accumulators and can still walk the trajectory sequentially —
which is what makes run-length (event) statistics possible in a parallel pass.

See also: [references/pet-quenching-theory.md](/references/pet-quenching-theory.md),
[architecture/simulation-pipeline.md](/architecture/simulation-pipeline.md).
