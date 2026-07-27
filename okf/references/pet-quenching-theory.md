---
type: Reference
title: PET quenching theory
description: The physics QuEst implements, the approximations it makes, and what its numbers can and cannot be trusted to mean.
resource: https://doi.org/10.1021/acs.jpcb.7b03441
tags: [pet, photophysics, fret, theory, approximations]
timestamp: '2026-07-26T00:00:00Z'
---

# The model

A dye tethered by a flexible linker samples an accessible volume around its
attachment point. While it is in van der Waals contact with an electron-rich
side chain, photo-induced electron transfer (PET) opens a non-radiative channel:
the excited-state decay rate becomes

```
k(t) = 1/tau0 + k_Q(r(t))
```

with `k_Q` a **step function** of position — the residue's `kQ` inside its
contact radius, zero outside — summed over all residues in contact. Averaging
over a Brownian trajectory in the accessible volume gives the observable decay,
which is generally multi-exponential even though the underlying photophysics has
one unquenched lifetime.

PET is a short-range, wavefunction-overlap process; its rate falls off roughly
exponentially with distance over ~1 Å. A step function at contact is therefore a
crude but defensible stand-in: the process is essentially all-or-nothing at the
distances the dye actually samples.

Residue reactivity for a xanthene dye follows the reference ordering
TRP > PRO ≈ TYR > MET > HIS > CYS (values in `REFERENCE_PET_QUENCHING`).

# Why the diffusion matters

If quenching were fast compared to the dye's exploration of its volume, the
decay would report on the *equilibrium* fraction of quenched configurations. It
is not: contact events last nanoseconds and the walk is diffusive, so the decay
depends on the **dynamics** — how often the dye reaches a quencher and how long
it stays. That is why QuEst simulates a trajectory rather than integrating over
the AV, and why the slow-factor grid (unspecific stickiness near the surface) is
not a cosmetic detail: it changes the residence times that set the decay.

The same trajectory yields the accessible **contact** volume (ACV) — the
fraction of the AV in which the dye is surface-associated — which is the
quantity a measured donor lifetime can calibrate.

# Approximations, stated plainly

From the README's warnings, restated as modelling assumptions:

1. **The dye is one sphere.** No orientation, no internal structure, no
   dye–protein chemistry beyond a contact radius.
2. **The protein is a single rigid structure.** No side-chain rearrangement, no
   backbone dynamics, no conformational ensemble.
3. **No specific interactions.** Binding pockets, stacking geometries and
   electrostatic steering are absent; the only "affinity" is the isotropic
   `slow_factor`.
4. **Parameters are transferable starting values, not constants.** The reference
   `kQ` and contact distances are for a xanthene dye and are meant to be
   calibrated against measured lifetimes.
5. **Precision is not accuracy.** The simulation returns tightly reproducible
   numbers; their agreement with experiment is a separate question.

The honest use of a QuEst number is comparative — this labeling site versus that
one, this ACV calibration versus that one — rather than absolute.

# FRET

With a second dye, the donor–acceptor distance is sampled from the two AV point
clouds and converted to a rate through `R0`. `RDAMeanE` (the Förster-weighted
average) is the quantity that corresponds to a FRET-efficiency measurement;
`RDAMean` and `dRmp` (mean-position distance) are different averages and are not
interchangeable with it.

# Citations

[1] Peulen, Opanasyuk & Seidel, "Combining Graphical and Analytical Methods with Molecular Simulations…", *J. Phys. Chem. B* **121**, 8211–8241 (2017), [doi:10.1021/acs.jpcb.7b03441](https://doi.org/10.1021/acs.jpcb.7b03441) — the QuEst method paper.
[2] Dimura et al., "Quantitative FRET studies and integrative modeling…", *Curr. Opin. Struct. Biol.* **40**, 163–185 (2016), [doi:10.1016/j.sbi.2016.11.012](https://doi.org/10.1016/j.sbi.2016.11.012) — accessible contact volumes.
