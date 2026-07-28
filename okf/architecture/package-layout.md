---
type: Architecture
title: Package layout
description: Where each kind of code lives in `quest/`, the rule that decides it, and the two naming habits that produced the mess this replaced.
resource: quest/
tags: [layout, architecture, packaging, naming]
timestamp: '2026-07-28T20:00:00Z'
---

# The rule

**A module's directory says what kind of thing it is; its name says which
thing.** If you find yourself putting the kind in the *name* — `structure_fetch`,
`structure_info` — the directory is missing.

```text
quest/
  api.py              the facade: verbs, over a client (LAY-04)
  cli.py              Click commands -> the facade
  manifest.json       identity, entry points, RPC methods, state schema

  project.py          the project schema: template, validate
  scan.py             simulating a set of labelling sites
  runs.py             run *artefacts* on disk
  tasks.py            *running* work: jobs, progress, cancellation
  i18n.py             locales

  core/               the domain. Qt-free, host-free, no job bookkeeping
    simulation.py     the pipeline, `simulate_project`
    dye_diffusion.py  the walk and the quenching model
    av.py             accessible volumes and the grid kernels
    photon.py         the photon trace
    io.py             xyz/MRC writers
    fps_json.py       the fps.json format
    structure/        everything about a structure
      fetch.py        where one comes from: path, PDB ID, bytes
      reader.py       -> QuEst's atom array (IMP parses)
      model.py        the `Structure` object and coarse-graining
      metadata.py     what is in it: chains, residues, atoms

  rpc/                the service layer
    contract.py       identity, error codes, the serialization boundary
    dispatcher.py     the envelope and the in-process client
    services.py       the method table and its handlers

  gui/                the Qt surface, rendered from the parameter catalog
  hosts/              reaching sibling checkouts, neither installed
    imp_tricks.py     splices `IMP.bff` in from source
    chisurf.py        splices ChiSurf in from source
  settings/           parameter catalog, dye presets, locales
```

# What this replaced, and why it was worse

Until 2026-07-28 the top level was flat and the *names* carried the structure:

| was | is |
|---|---|
| `quest/structure_fetch.py` | `quest/core/structure/fetch.py` |
| `quest/structure_info.py` | `quest/core/structure/metadata.py` |
| `quest/core/structure.py` | `quest/core/structure/model.py` |
| `quest/core/pdb.py` | `quest/core/structure/reader.py` |
| `quest/imp_tricks.py` | `quest/hosts/imp_tricks.py` |
| `quest/chisurf_host.py` | `quest/hosts/chisurf.py` |
| `quest/backend/` | `quest/rpc/` |

Two habits produced it, and both are worth naming so they are not repeated.

**A prefix is a directory that was never created.** `structure_fetch`,
`structure_info`, `core/structure` and `core/pdb` were four aspects of one
subject, split across two packages, with a naming convention that *implied* the
relationship the layout denied. "Where does a structure come from" had no single
answer, and `core/pdb.py` did not even carry the prefix — it read mmCIF too.

**A suffix that means "the other side of a bridge" hides a pair.**
`imp_tricks.py` and `chisurf_host.py` do the same job — splice a sibling
checkout onto the import path — and looked unrelated. They are `hosts/` now, and
a third bridge has an obvious home.

`backend/` → `rpc/` is the smaller change and is about matching what already
exists elsewhere: ChiSurf plugins use `rpc/services.py`, QuEst's own ChiSurf
plugin uses `rpc/services.py`, and `manifest.json` calls the entry point
`services`. "Backend" also meant three other things in this project — the *AV*
backend, the *web* backend, the *transport*.

# The seams this preserves

Renaming did not move any boundary, and must not:

- **`core/` has no Qt, no host, and no job bookkeeping.** It takes plain data
  and callables — `should_cancel` is a predicate, not a `Job`
  ([subsystems/tasks.md](/subsystems/tasks.md)) — so the domain never learns
  what is driving it.
- **`hosts/` is optional in every direction.** Each bridge returns empty or
  `False` with no checkout, and QuEst runs standalone.
- **`rpc/` is where serialization *stops*.** Handlers return native objects; a
  transport converts ([specs/programme-2026-07.md](/specs/programme-2026-07.md),
  decision 7).
- **One structure seam.** A caller says *what* to label, never *how to read it*
  ([specs/structure-io.md](/specs/structure-io.md)).

# Where new code goes

- Something the science does → `core/`, and if it is about structures,
  `core/structure/`.
- A new RPC method → `rpc/services.py`, plus `manifest.json`, plus the ChiSurf
  plugin's copy of the method table (a test enforces all three).
- A new **parameter** → `settings/parameter_catalog.json`, then regenerate the
  docs and the Qt view spec — **and edit the web UI by hand**, which is the one
  surface the catalog does not reach. See the trap in
  [handover.md](/handover.md).
- Reaching another checkout → `hosts/`.
- Anything Qt → `gui/`, never `core/`.
