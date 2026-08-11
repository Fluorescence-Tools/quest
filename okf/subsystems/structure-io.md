---
type: Subsystem
title: Structure I/O
description: PDB/CIF/PQR parsing, the Structure object, coarse-graining, and PDB-ID fetching.
resource: quest/core/structure/
tags: [pdb, cif, structure, io, rcsb]
status: stable
generated: { by: process:okf-v0.1-import, at: 2026-07-26T00:00:00Z }
stale_after: 2027-02-06
sources:
  - id: quest-core-structure
    resource: ../../quest/core/structure/
    title: quest/core/structure/
    author: human:tpeulen
    last_modified: 2026-07-28
---

# Four concerns, one package

`quest/core/structure/` states its own layout, and the order is the direction of
travel: a caller says *what* it wants labelled, never *how to read it*.

| Module | Owns |
|---|---|
| `fetch.py` | Where a structure comes from: a path, a PDB ID, bytes |
| `reader.py` | Turning that into QuEst's atom array -- **IMP does the parsing** |
| `model.py` | The `Structure` object and coarse-graining |
| `metadata.py` | What is in it: chains, residues, atoms, for a site picker |

# Parsing: IMP is the parser

`read(filename)` returns a NumPy structured array with the fields in
`reader.keys`. **QuEst no longer carries its own PDB or mmCIF parser** --
`IMP.atom.read_pdb` / `read_mmcif` do it, and `read` raises `ImportError`
without IMP, which the accessible-volume stage needs anyway. The public surface
is small: `read`, `keys`, `formats`, plus `write` and `sequence`.

The old hand-rolled entry points (`parse_string_pdb`, `parse_string_pqr`,
`parse_string_cif`, `assign_element_to_atom_name`) are **gone**; so is the
elements table they fed.

**What counts as an obstacle changed with them, and it was measured rather than
assumed.** No selector is passed, so IMP's default applies: everything except
waters and hydrogens. Ligands, cofactors and modified residues are steric
obstacles because they physically are -- a dye cannot diffuse through a bound
sugar. QuEst used to strip every HETATM, which on 148L discarded 41 atoms
(1322 against 1363). Waters are deliberately *not* obstacles: adding them costs
**17 % of every accessible volume** on the 148l chain-E sites, and
crystallographic solvent is mobile on the timescale a tethered dye explores.
Full measurement: [specs/structure-io.md](/specs/structure-io.md).

As a defensive guarantee added 2026-08-07, `_strip_waters` runs after the IMP
parse and removes every atom whose residue name identifies it as water (`HOH`,
`WAT`, `TIP3`, `SOL`, … — see `_WATER_RESIDUE_NAMES`). IMP's selector already
drops waters, but the explicit filter keeps QuEst correct regardless of which
selector a caller or a future parser passes. `tests/test_water_stripping.py`
covers it.

**One residue table, three points of use**, because a water reaching *any*
surface is the same defect:

| Where | What it does |
|---|---|
| `_strip_waters(atoms)` | filters the array after the IMP parse — the simulation path |
| `strip_water_pdb_text(text)` | filters raw PDB text by columns 18–20 — for anything that consumes text |
| `water_stripped_pdb_path(src)` | writes a cleaned copy to a temp file, reused per source — for a viewer that insists on a *file* |

`structure_metadata` runs PDB text through `strip_water_pdb_text` and skips
water rows while parsing CIF, so the `pdb_text` it returns to the browser
viewer carries none either, and no site picker can offer a water as a labeling
site. The ChiMol 3D viewer loads with `keep_water=True`, which is why stripping
at the text level — rather than trusting the viewer's own selector — is the
thing that actually holds. Baselines moved with it: 148L `atom_count`
1525 → 1385, chain E residues 291 → 164.

Residue names coming out of the structured array are byte-string valued and need
normalising before comparison -- hence `_normalize_residue_name` in both
`quest/core/simulation.py` and `quest/core/dye_diffusion.py` (it also unwraps
the `b'ALA'` repr that leaks through some paths).

# `Structure`

`model.py`'s `Structure` is now a **thin accessor over the atom array**, not the
geometry toolkit this concept used to describe. It exposes `xyz`, `vdw`,
`b_factors`, `atom_types`, `residue_names`, `residue_ids`, `residue_indices`,
`residue_dict`, `n_atoms`, `n_residues`, `radius_gyration`, and `to_coarse`;
module-level helpers add `move_center_of_mass`, `make_residue_lookup_table` and
`get_residue_sequence`.

**The alignment and clustering methods are gone** -- `rmsd`, `super_impose`,
`average`, `find_best`, `find_representative`, `cluster`, `onRMSF`,
`calc_internal_coordinates`, `_internal_to_cartesian`. QuEst does not align or
cluster structures; that machinery belonged to the general-purpose library this
package replaced, and keeping a second copy of it is exactly what
[specs/imp-tricks-dedup.md](/specs/imp-tricks-dedup.md) argues against.

`Structure(path, make_coarse=True)` still builds the coarse-grained
representation behind the project's `coarse_grained` flag
([dye-diffusion.md](dye-diffusion.md)); `tests/test_coarse_grained.py` covers it.

# Structure sources: path or PDB ID

`quest/core/structure/fetch.py` lets every surface accept a bare four-character RCSB
identifier wherever a file path is accepted:

- `is_pdb_id(value)` — a 4-character alphanumeric string;
- `download_pdb_id(id, cache_dir=None)` — tries `.pdb` then `.cif` from
  `files.rcsb.org`, caching under `QUEST_PDB_CACHE_DIR` (default: a
  `quest-pdb-cache` folder in the system temp dir), and returns the cached path
  if it already exists;
- `resolve_structure_path(source)` — the one call every surface uses.

A 404 on `.pdb` falls through to `.cif` (large/modern entries have no PDB file);
other HTTP or network errors raise. `tests/test_pdb_id_support.py` covers this.

Anything that takes a structure should call `resolve_structure_path`, not
`Path(...)` — that is how PDB-ID support stays uniform across CLI, API and web.

# A path quirk that is gone — do not reintroduce it

This section used to describe `quest/__init__.py` calling
`utils.set_search_paths(<package dir>)`, which inserted `quest/` itself onto
`sys.path` so that `import lib.fps` resolved. **Both are gone**: the injection
with `LAY-08` (2026-07-27), and `quest/utils.py` itself on 2026-07-28 — nothing
imported it, and the rest of it was Python-2.6 `unittest` monkey-patching plus a
`sitecustomize` coverage hook that wrote `.coverage.*` files into the working
directory.

The rule it leaves behind outlived the package it was written about: every
internal import is spelled `quest.…`, absolutely. A bare `import lib…` was not a
legacy spelling to tolerate but a bug -- `lib.structure.Structure` and
`quest.lib.structure.Structure` were two different classes, so an `isinstance`
check across the boundary failed and a `Structure` got wrapped twice. `quest/lib`
itself is gone; the rule is why the replacement is a package under
`quest/core/`.

Re-read against the tree on 2026-08-10, mechanically: every name above was
resolved with `hasattr`, and every name this concept claims is *gone* was
checked to be actually absent — a removal claim rots the same way a presence
claim does. `tests/test_coarse_grained.py` and `tests/test_pdb_id_support.py`
both exist. The only sentence that needed changing was the water paragraph,
which described one filter where there are now three.
