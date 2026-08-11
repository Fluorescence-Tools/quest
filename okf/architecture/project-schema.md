---
type: Specification
title: Project schema
description: The project-JSON contract shared by every QuEst surface, and the parameter catalog that documents it.
resource: quest/cli.py
tags: [schema, project, json, parameters]
status: stable
generated: { by: process:okf-v0.1-import, at: 2026-07-26T00:00:00Z }
stale_after: 2027-01-26
sources:
  - id: quest-cli-py
    resource: ../../quest/cli.py
    title: quest/cli.py
    author: human:tpeulen
    last_modified: 2026-07-28
---

# One document, every surface

A QuEst **project** is a plain JSON object. The CLI, the Python API, the web UI
and the Qt GUI all read and write the same document — there is no second format
and no per-surface variant. `quest template [--with-fret]` emits a valid
starter; `quest validate -p project.json` checks one. Both live in
`quest/cli.py` (`_template_project`, `_validate_project`).

# The document

```jsonc
{
  "tau0": 4.2,                    // unquenched donor lifetime, ns
  "n_photons": 500000,            // photons to collect (photon mode)
  "t_max": 16000.0,               // diffusion duration, ns
  "t_step": 0.032,                // diffusion time step, ns
  "dg": 0.5,                      // AV grid spacing, Angstrom
  "critical_distance": 8.5,       // fallback dye-centre -> quencher radius, A
  "slow_radius": 8.5,             // stickiness radius around a residue, A
  "av_parameter": {"linker_length": 21.5, "linker_width": 0.5, "radius1": 3.5},
  "amino_acid_interactions": { "TRP": {"kQ": …, "quench_radius": …,
                                       "quench_atoms": [...], "slow_factor": …}, … },
  "pdb": "structure.pdb",         // path, or a bare 4-character RCSB PDB ID
  "attachment": {"chain": "A", "residue": 1, "atom": "CB"},
  "n_bins": 4096,                 // TAC histogram bins
  "parallel_trajectories": -1,    // -1 = all cores
  "coarse_grained": false,
  "random_seed": null,            // set for reproducible runs
  "decay_mode": "photon",         // or "curve" (+ n_curves / dt_tac / n_tac)
  "save_avs": false,
  "output_file": "out",
  "fret": {
    "enabled": true,
    "dyes": [ {"name": "donor", "D": 7.5, "attachment": {...},
               "av_parameter": {...}, "slow_radius": …,
               "amino_acid_interactions": {...}},
              {"name": "acceptor", "D": 7.5, "attachment": {...}, "av_parameter": {...}} ],
    "R0_matrix": [[null, 5.5], [5.5, null]]
  }
}
```

**Required top-level keys** (enforced by `validate`): `tau0`, `n_photons`,
`t_max`, `t_step`, `dg`, `critical_distance`, `av_parameter`,
`amino_acid_interactions`, plus a structure (`pdb` or an override) and at least
one entry in `fret.dyes` carrying a diffusion coefficient `D`.

# Units: Ångström and nanoseconds, everywhere

Every length in a project is in **Ångström** — linker geometry, dye radii,
`critical_distance`, `slow_radius`, the grid spacing `dg`, and the Förster radii
in `fret.R0_matrix`. Every time is in **nanoseconds** (`tau0`, `t_max`,
`t_step`), and diffusion coefficients are Å²/ns.

`R0_matrix` is the one that has bitten: the transfer rate goes as `(R0/r)^6`, so
a Förster radius entered in nanometres is ten times too small and FRET
disappears without an error. `quest/project.py` exposes the template value as
`DEFAULT_FORSTER_RADIUS = 52.0` with the unit stated, and
`tests/test_fret_units.py` pins it.

# Two rules that keep it coherent

- **The donor is `fret.dyes[0]`, always** — even when `fret.enabled` is false.
  That is where `D`, and optionally a per-dye `slow_radius`,
  `amino_acid_interactions`, `av_parameter`, `random_seed` and `coarse_grained`
  live. Top-level values are the fallback; the dye entry wins where both exist
  (see `build_donor_from_project`).
- **Dotted paths address everything.** The CLI's `--set a.b.c=value` and
  `--grid a.b.c=v1,v2` walk the same dotted path into the document, which is why
  nesting stays shallow and predictable. New keys should keep that property.

# The parameter catalog is the single source of truth for help text

`quest/settings/parameter_catalog.json` maps each dotted parameter id to its
label, category, type, unit and description. It feeds:

- the web UI's field labels and help badges,
- `doc/parameters.md`, which is **generated** by `doc/generate_parameter_docs.py`
  and must never be hand-edited, and
- the `/api/parameter-catalog` endpoint.

Changing a parameter's meaning means editing the catalog and regenerating the
doc **in the same change** — see
[workflows/change-tracking.md](/workflows/change-tracking.md).

# Dye presets

`quest/settings/dye_repository.json` holds per-dye starting values (`tau0`, `D`,
`av_parameter`, `slow_radius`, and a full `amino_acid_interactions` table) for
Alexa488, Alexa594, Cy3, Cy5 and Atto488. The web UI serves them at `/api/dyes`
and can append user dyes there. They are **starting values to be calibrated**,
not constants — see [references/pet-quenching-theory.md](/references/pet-quenching-theory.md).
