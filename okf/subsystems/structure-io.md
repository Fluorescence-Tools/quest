---
type: Subsystem
title: Structure I/O
description: PDB/CIF/PQR parsing, the Structure object, coarse-graining, and PDB-ID fetching.
resource: quest/core/structure/
tags: [pdb, cif, structure, io, rcsb]
status: stable
generated: { by: process:okf-v0.1-import, at: 2026-07-26T00:00:00Z }
stale_after: 2026-08-06
sources:
  - id: quest-core-structure
    resource: ../../quest/core/structure/
    title: quest/core/structure/
    author: human:tpeulen
    last_modified: 2026-07-28
---

> **Marked stale on 2026-08-06.** The file paths in this concept were
> corrected against the tree, but its *descriptions* were not re-read --
> its subject moved into `quest/core/structure/` and split into model/reader/metadata/fetch. Treat the mechanism it describes as unverified until someone
> reads it against the code. See
> [references/known-issues.md](../references/known-issues.md).
# Parsing

`quest/core/structure/reader.py` parses PDB, PQR and mmCIF into a single NumPy structured
array (fields include `chain`, `res_id`, `res_name`, `atom_name`, `xyz`,
`radius`, `charge`). Entry points: `read(filename)`, `parse_string_pdb`,
`parse_string_pqr`, `parse_string_cif`, plus `write` and `sequence`.
`assign_element_to_atom_name` recovers elements from ambiguous atom names, which
is what the van der Waals radii lookup depends on.

Because the array is structured and byte-string valued, residue names coming out
of it need normalising before comparison — hence `_normalize_residue_name` in
both `core.py` and `dye_diffusion.py` (it also unwraps the `b'ALA'` repr that
leaks through some paths).

# `Structure`

`quest/core/structure/model.py` wraps the atom array with the geometry the
simulation and analysis need: `rmsd`, `super_impose`, `average`,
`find_best`/`find_representative`, `cluster` (hierarchical), `onRMSF`,
internal-coordinate conversion (`calc_internal_coordinates`,
`_internal_to_cartesian`), `move_center_of_mass`, and residue lookup tables.

`Structure(path, make_coarse=True)` builds the coarse-grained representation
used by the project's `coarse_grained` flag
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

The rule it leaves behind: every internal import is spelled `quest.lib…`. A bare
`import lib…` is not a legacy spelling to tolerate, it is a bug — `lib.structure.
Structure` and `quest.lib.structure.Structure` were two different classes, so an
`isinstance` check across the boundary failed and a `Structure` got wrapped
twice.
