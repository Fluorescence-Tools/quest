---
type: Specification
title: One seam for structure input
description: Every structure enters QuEst through a single loader that yields an atom array; nothing downstream ever sees a filename or a format.
resource: quest/core/pdb.py
tags: [io, structure, pdb, cif, radii, architecture]
timestamp: '2026-07-27T00:00:00Z'
---

# The rule

**One module turns a structure *source* into an atom array. Everything else
takes the array.** No other code opens a structure file, knows a format, or
decides a radius.

A "source" is any of: a path to PDB/mmCIF, a four-character RCSB ID, bytes, or
an already-parsed structure. A caller says *what* it wants labelled; it never
says *how to read it*.

# Why — three bugs from one cause, all found 2026-07-27

Every structure-input defect this project has hit is the same shape: something
downstream re-derived the structure instead of being handed it.

1. **`imp_av.py` re-reads the file.** `build_imp_accessible_volume` calls
   `IMP.atom.read_pdb(path)` rather than using the structure it was given, so a
   coarse-grained project silently gets the **all-atom** volume — coarse and
   all-atom measured byte-identical at 13883.805 Å³.
2. **The same re-read is `read_pdb` only**, so `.cif` projects fail outright
   under that backend ("No molecule read from file … mc4r_dimer.cif") while
   working perfectly through `quest/core/pdb.py`, which handles mmCIF. One
   format, two answers, depending on which backend you happened to pick.
3. **Two radius tables.** LabelLib was fed QuEst's Bondi-like radii (C 1.76,
   N 1.65, O 1.40) and IMP.bff its own (C 2.10, N 1.85, O 1.71) — 19 % larger
   carbon, **72 % more excluded volume**, and hence AVs differing by ~20 %
   (14 500 vs 17 500 Å³ on 148l E15). That was misread for a whole session as an
   *algorithmic* difference between the backends. It was an input difference.
   The backends agree on where the dye is (centroids within 0.6 Å); they were
   never asked the same question.

The same argument applies to `fps.json`: the format is somebody else's standard
and it will change. One reader, one writer, one place to fix.

# The seam

    load_structure(source, *, coarse=False, radii=..., selector=...) -> Atoms

`Atoms` is the structured array QuEst already simulates on — `coord`, `radius`,
`atom_name`, `res_name`, `res_id`, `chain`, `mass`. Downstream consumers take
that array and nothing else:

| consumer | takes today | must take |
|---|---|---|
| `AV` / LabelLib backend | a `Structure` | the array |
| `imp_av.py` | **a filename** | the array |
| `IMP.bff.av.compute_av` | already arrays ✔ | — |
| quenching / diffusion grids | a `Structure` | the array |

`compute_av`'s signature (`atoms_xyz`, `atoms_vdw`, `source_xyz`) is the model
to copy: a function that takes coordinates *cannot* re-read a file, miss a
coarse-graining, or disagree about a format.

# Reading is IMP's job; radii are QuEst's decision

Two separable things that got conflated:

- **Parsing** — use IMP. It handles mmCIF, altlocs, insertion codes and hetero
  records, and it is maintained. QuEst's own parser should go.
- **Radii** — inherit IMP's. **Decided 2026-07-27 and implemented**: the reader
  takes `IMP.core.XYZR(p).get_radius()` off the particle IMP decorated, and
  `quest/core/elements.py` is gone. (An earlier draft of this spec argued the
  opposite — keep a QuEst table because IMP's are ~19 % larger than the
  Bondi-like values the FPS convention assumes. That was overruled: one source
  of radii matters more than matching the older convention, and having two was
  what made the backends look like they disagreed.)

  The cost is real and measured: IMP's are CHARMM-style (C 2.10 Å against
  1.76), so accessible volumes shrink and **only 43 of 151 CB sites on 148l
  chain E remain labelable**. Numbers are no longer comparable to published FPS
  work computed at Bondi radii. Both backends are fed the same array, so the
  backend choice is availability only — IMP.bff preferred, LabelLib where
  IMP.bff is not built (Windows).

**Decided 2026-07-27: HETATM and water are not stripped.** A bound ligand
occludes the dye, so the loader keeps IMP's full atom set rather than QuEst's
ATOM-only selector — 1363 atoms on 148l against 1322, the extra 63 being `FGA`,
`API`, `DAL`, `MUB`, `NAG` and `BME`. Not yet implemented; see
[handover](../handover.md).

One thing to settle while implementing it: IMP's *default* read already omits
water and hydrogens (`NonWaterNonHydrogenPDBSelector` returns the same 1363), so
passing no selector satisfies "keep the ligands". Keeping crystallographic
waters as steric obstacles as well is a stronger and more unusual choice —
mobile solvent does not normally occlude a dye — and needs
`NonHydrogenPDBSelector` (1503 atoms). Measure both.

# Done, and what remains

Done 2026-07-27:

- `quest/core/elements.py` **deleted** (271 lines) — radii and masses come off
  the IMP particle, and the `assignCharge` path it also served had no caller.
- `quest/core/structure.py` 623 → ~345 lines: RMSD, clustering and the torsion
  subsystem had no readers.
- `imp_av.py` takes the atom array; its file handling is gone, which fixed both
  the discarded coarse-graining and the mmCIF failure above.
- The acceptance test passed: with identical radii the two backends converge to
  a **0.90 ratio** on 148l E36, centroids within 0.5 Å. Before, with a table
  each, it was 0.78 — and that gap was misread as algorithmic.

Remaining:

1. The loader still has no single `load_structure(...)` entry point; `AV` and
   the quenching grids still take a `Structure` rather than an array.
2. Same treatment for `fps.json`: one reader, one writer.
3. The decisions in [handover](../handover.md) — none implemented.
