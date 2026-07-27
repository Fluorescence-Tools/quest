---
type: Specification
title: Deduplication against imp-tricks
description: General fluorescence and structural machinery belongs in imp-tricks under IMP.bff; QuEst is the application on top of it.
resource: https://github.com/fluorescence-tools/imp-tricks
tags: [target, dedup, imp, imp-bff, layering, accessible-volume]
timestamp: '2026-07-27T00:00:00Z'
---

> Target. Companion to [plugin-integration.md](plugin-integration.md): that one
> is about *how QuEst is called*, this one is about *what QuEst should own*.

# The rule

**General fluorescence machinery lives in imp-tricks, under `IMP.bff`.**
QuEst depends on it rather than carrying its own copy. Anything that is not
specific to "estimate PET quenching for this project document" is a candidate to
move: accessible volumes, dye distributions, FRET distance metrics and rates,
labelling geometry, structure handling.

What stays in QuEst:

- the [project schema](/architecture/project-schema.md) and its validation,
- the PET-quenching *workflow* — grids, trajectory, photon generation, residue
  attribution — insofar as it is not already in `IMP.bff`,
- the [surfaces](/architecture/surfaces.md): CLI, web UI, Qt GUI, RPC,
- scanning, [job artefacts](/architecture/job-artefacts.md), the parameter
  catalog and dye library.

Nothing in imp-tricks may depend on QuEst. The arrow points one way.

# What imp-tricks already has

Checked against `~/dev/imp-tricks/src/IMP/bff/` on 2026-07-27:

| Module | Provides |
|---|---|
| `IMP.bff.av.compute_av` | AV from atoms + attachment site, **with the same IMP.bff → LabelLib backend dispatch QuEst reimplements** (`_av_imp_bff`, `_av_labellib`), and an explicit `backend=` override QuEst lacks |
| `IMP.bff.av.BasicAV` / `ACV` | AV and accessible **contact** volume objects: point clouds, density grids, mean position, slow centres, trapped fraction |
| `IMP.bff.av._kernels` | `density2points`, `random_distances`, `average_distance`, `mean_fret_distance`, `weighted_mean`, `split_av_acv` |
| `IMP.bff.distance_metrics` | `fret_efficiency`, `distance_from_fret_efficiency`, `av_pair_statistics` (⟨R_DA⟩, R_mp, R_E, ⟨E⟩ in one pass), `chi2_score` |
| `IMP.bff.cgdye` | Coarse-grained dye simulation: labelling, rotamers, sampling, and `analysis/fret.py` with **κ²-aware** rates and an exact efficiency from a rate matrix |
| `IMP.bff.polymer`, `.distributions`, `.label`, `.restraints` | Chain models, distance distributions, labelling, restraints |

# Historical: what `quest/lib` was, module by module

`quest/lib` no longer exists — it became `quest/core/` on 2026-07-27. The audit
below is kept because it records *why* each piece survived or died; the current
verdict is the re-measured one further down.

A minimal in-tree library is legitimate — QuEst must run standalone — but it has
to be *minimal*, and today's is neither that nor compatible with the two
libraries it shadows. Audited 2026-07-27:

| Module | Lines | Verdict |
|---|---|---|
| `lib/tools/dye_diffusion/` | ~1700 | **Keep.** This is QuEst: the PET workflow, `SimulateDiffusion`, `DonorDecay`, the photon kernels. |
| `lib/fps/` | ~1190 | **Mostly duplicate** of `IMP.bff.av` + `IMP.bff.distance_metrics` (`DUP-01`, `DUP-03`–`DUP-05`). Blocked on imp-tricks. `AvPotential` was dead and is gone. |
| `lib/structure/Structure.py` | ~640 | **Keep the reader, drop the analysis.** RMSD, superposition, clustering, internal coordinates and `Universe` were unreachable and are gone — that is IMP/ChiSurf work. What remains is the atom array QuEst simulates on. |
| `lib/io/PDB.py` | ~400 | **Duplicate** of IMP's readers *and* ChiSurf's `core.fio.structure.coordinates` (`DUP-06`) — a third copy. Blocked on a dependency decision, not on code: IMP is missing from the base environment (where the web layer runs), and ChiSurf is missing from `arm64` and would pull tttrlib/mdtraj/pandas into every simulation. Recommended home: **IMP.atom**. |
| `lib/common.py` | 271 | Chemical reference data (atomic weights, VDW radii, charges) for the parser. Moves out with `DUP-06`. |
| ~~`lib/math/`~~ | — | **Gone.** The twelve lines of FFT autocorrelation are inlined in `quest/core.py`; ChiSurf's `core.math.signal.autocorr` is the same thing, but a headless run must not import a host application for a primitive. |
| `lib/imp_av.py` | ~200 | The broken IMP integration (`DUP-01`/`DUP-02`). Delete when `compute_av` works. |
| `lib/fps_json.py` | 112 | fps.json labeling files. QuEst-specific enough to keep. |
| ~~`lib/plots/`, `lib/widgets.py`, `lib/ui/`, `lib/genealogy.py`~~ | ~2400 | **Moved to `quest/gui/`** — they were Qt, used only by the form. |

`quest/lib` went from **7673 to 4511 lines** on 2026-07-27: ~2400 moved to
`quest/gui/`, and **668 lines of unreachable code deleted** (found by
reachability analysis from the real entry points, not by eye).

# Duplication against ChiSurf

The presentation arrow points at ChiSurf, so its code is fair game too. Three
QuEst modules are shadowed by ChiSurf's **`fps_json_editor` plugin**
(`chisurf/plugins/modelling/fps_json_editor/core/`):

| QuEst | ChiSurf equivalent |
|---|---|
| `quest/core/fps_json.py` — load and normalise fps.json positions | `core/payload.py` — `normalize_payload`, `validate_payload`, `summarize_payload` |
| `quest/structure_fetch.py` — RCSB download and cache | `core/pdb.py` — `download_pdb_file`, `default_cache_dir`, `pdb_source_url` |
| `write_mrc` in `quest/core/simulation.py` | `core/mrc.py` — `save_av_mrc` |

**It cannot be consumed as it stands.** All three live inside a ChiSurf
*plugin*, and a plugin is not a library: QuEst's domain importing
`chisurf.plugins.modelling.fps_json_editor.core.payload` would couple two
plugins through the host's plugin tree, and would drag the whole ChiSurf import
(tttrlib, mdtraj, pandas) into a headless simulation.

The fix belongs on the ChiSurf side: fps.json handling, RCSB fetching and MRC
writing are **library** concerns, so they belong in `chisurf.core.*` (or, for
the labelling parts, imp-tricks). Once they are there, QuEst deletes its copies.
Recorded as `DUP-07`.

# Verdict, module by module (re-measured 2026-07-27)

Everything below was **run**, not read: imports attempted, kernels compared
numerically, the AV built both ways. The evidence is in `okf/log.md` for the
same date.

## Stays in QuEst — it *is* the application

| module | lines | why it stays |
|---|---|---|
| `dye_diffusion.py` | 1551 | The PET quenching model, sticky diffusion grids, `DonorDecay`. This is the science QuEst exists for; nothing upstream has it. |
| `simulation.py` | 1323 | Project → result orchestration, `aa_residence`, job artefacts. Application logic. |
| `photon.py` | 212 | Monte-Carlo photon trace against a per-frame quenching rate. Coupled to the quenching trajectory model. |
| `pdb.py` | 228 | Already delegates to `IMP.atom.read_pdb`; what remains is QuEst's radii, the parse cache and the SWIG lock. |
| `fps_json.py` | 112 | Scan glue. The payload half already moved to `chisurf.core`. |

## Belongs in imp-tricks — and is adoptable **today**

| what | upstream | evidence |
|---|---|---|
| `density2points` **and** `density2points_ll` (two copies!) | `IMP.bff.av._kernels.density2points` | Numerically identical on a test grid. Theirs is a strict superset: arbitrary grid dimensions and it keeps the per-voxel weight, which QuEst throws away. QuEst carries a flat/centre-origin variant *and* a 3-D/corner-origin one; upstream needs neither. |
| `RDAMean`, `RDAMeanE`, `_ran_dist` | `random_distances` + `av_pair_statistics` | Agree within sampling noise on 148l E15/E90: ⟨R_DA⟩ 51.56 vs 51.53 Å, ⟨R_DA⟩_E 51.36 vs 51.53 Å. |

Neither is blocked by anything. `IMP.bff.av._kernels` and
`IMP.bff.distance_metrics` import and run in `arm64` today.

## Unblocked 2026-07-27 — fixed upstream

The three faults below were fixed **in imp-tricks**, and
`IMP.bff.av.compute.compute_av` now reproduces QuEst's accessible volume
**bitwise** on 148l E15 (167 196 points, identical after sorting). Pinned by
`tests/test_imp_tricks_bridge.py::TestTheUpstreamAvMatchesQuEsts`, which skips
where LabelLib or the checkout is missing.

1. `av/compute.py` — `Particle.get_particle_index()` did not exist on this IMP,
   which spells it `get_index()`. Now goes through a `_particle_index()` shim
   that accepts either, because the two spellings are both in the wild.
2. `av/compute.py` — the LabelLib probe required `LabelLib.AV`. Current LabelLib
   exposes only the `dyeDensityAV1` kernel, so a perfectly working install was
   reported as missing and the backend was unreachable. The probe now accepts
   either, and `_av_labellib` drives `dyeDensityAV1` when the high-level entry
   point is absent, converting its centre-origin Fortran grid to the
   corner-origin convention `density2points` expects.
3. `av/compute.py` — **the reason it returned an empty volume**: the attachment
   atom's own van-der-Waals sphere was left in the obstacle list. The linker
   starts *at* that atom, so the source sits inside an obstacle and LabelLib
   returns nothing — no error, just an empty cloud. QuEst had always zeroed it;
   upstream had not.

Two more upstream fixes went with them:

- `src/sitecustomize.py` extended `IMP.__path__` but not the `__path__` of
  subpackages that are **also compiled and installed**. `IMP.bff` is a regular
  package whose `__path__` points at site-packages, so Python never consulted
  `IMP.__path__` for its children and `import IMP.bff.av` failed however
  `sys.path` was arranged — imp-tricks' own test suite could not import its own
  AV code (28 collection errors). QuEst had solved this independently in
  `quest/imp_tricks.py`; the same handling is now upstream. Its `bff` suite goes
  from *uncollectable* to **191 passed, 5 failed, 24 skipped**.
- Two tests asserted `ImportError` "without a backend" and passed only *because*
  of fault 2 — a usable LabelLib was being reported as missing. They now
  monkeypatch the availability flags to create the condition they describe.

The 5 remaining failures (`cgdye/rotamer` R0-vs-FRETpredict, `dye_io`
PDB→mmCIF, `polymer` Gaussian-chain PDF) touch none of this — zero references to
`av.compute`, LabelLib or `compute_av` — and had never run before, since the
suite could not be collected. Newly *visible*, not newly broken.

## What still stops the deletion: packaging, not capability

`IMP.bff.av` resolves in **both** environments once the checkout is on the path —
including the base interpreter, where there is no compiled `IMP.bff` to shadow
it. So the barrier is no longer "it does not work". It is that imp-tricks is a
development path shim (`quest/imp_tricks.py`), not a declared dependency:
deleting QuEst's `AV` today would break any install that does not happen to have
a checkout. Declaring it in `pyproject.toml` and the rattler recipe is the next
step, and it is a decision about what QuEst depends on, not a code change.

## Superseded — the original diagnosis, for reference

`AV` (in `av.py`) and `imp_av.py` should both become
`IMP.bff.av.compute.compute_av`, whose signature is already the right shape —
plain arrays (`atoms_xyz`, `atoms_vdw`, `source_xyz`), no IMP particles. It
cannot compute an AV in `arm64` by **either** backend:

- `compute.py:162` — `ca_p.get_particle_index()`. The compiled IMP here spells
  it `get_index()`. `AttributeError`.
- `compute.py:231` — the LabelLib backend refuses with "LabelLib is required".
  LabelLib *is* installed; it exposes `dyeDensityAV1` / `dyeDensityAV1_arr`,
  and imp-tricks looks for `LabelLib.AV`.

Both are one-liners **in imp-tricks**. Until they are fixed, QuEst keeps its own
AV and the duplication stands — this is the single change that unblocks the
largest deletion (`DUP-01`, `DUP-02`, and `imp_av.py` entirely).

## The arrow points the other way — QuEst is right, upstream is wrong

`av_pair_statistics` returns four numbers, and its second, `r_mp`, is
documented as *"Distance between mean positions (same as `r_da_mean` here)"* —
it returns `r_da_mean` again. **R_mp is not ⟨R_DA⟩.** R_mp is
|⟨r_D⟩ − ⟨r_A⟩|, the distance between the two cloud centroids, and it cannot be
recovered from a sample of pair distances at all: the sampling destroys the
information. Measured on 148l E15/E90, QuEst's `dRmp` gives **47.65 Å** and
upstream's `r_mp` gives **51.53 Å** — 8 % apart, and the smaller number is the
correct one.

So `dRmp` does not get deleted. It gets **contributed upstream**, and
`av_pair_statistics` should either take the two clouds or stop claiming to
return R_mp. Anyone who trusted that field has a systematically wrong R_mp.

## Deleted 2026-07-27 — dead weight, now gone

`structure.py` went **623 → 345 lines**. What went: `rmsd`, `cluster`,
`calc_internal_coordinates`, `move_center_of_mass`'s geometry primitives
(`_norm`, `_angle`, `_dihedral`, `r2i`), and the whole torsion subsystem —
`coord_i`, `internal_coordinates`, and the `phi`/`psi`/`omega`/`chi` accessors,
which nothing read. What stayed: the constructor, `to_coarse` (reached through
`make_coarse=`), `xyz`, `vdw`, `atoms`, `residue_names`, `residue_ids`, `write`,
and the three helpers the class itself calls.

**A caution learned doing it.** "No callers outside this module" is the wrong
question for a module-level helper: `make_residue_lookup_table`,
`get_residue_sequence` and `move_center_of_mass` had none, and are called from
`Structure.__init__`. Deleting them broke construction immediately. A second
trap: cutting a function span up to the next `def`/`class` swallows the
module-level constants that follow it — `a2id`, `res2id`, `internal_keys` went
with the first cut. And a third: the deletion left an orphaned `@njit` that
silently decorated the next function added below it. Each was caught by running
the code, none by reading the diff.

The original survey, for reference:

- module level, zero callers: `rmsd`, `cluster`, `calc_internal_coordinates`,
  `move_center_of_mass`, `r2i`, `make_residue_lookup_table`,
  `get_residue_sequence`;
- `Structure` methods, zero callers: `internal_coordinates`,
  `radius_gyration`, `update_dist`, `update_coordinates`, `b_factors`,
  `atom_types`, `residue_dict`, `n_residues`, and the `phi`/`psi`/`omega`/`chi`
  torsion accessors.

What is live: the constructor, `to_coarse` (through `make_coarse=`), `xyz`,
`vdw`, `atoms`, `residue_names`, `residue_ids`, `write`. This is a QuEst-side
cleanup with no upstream dependency — it does not need imp-tricks fixed first.

## No counterpart yet

`fret_rate_trace` and `fret_rate_pair_trace` have none.
`IMP.bff.cgdye.analysis.fret` computes *efficiencies* (and κ², which QuEst
lacks — `OBJ-01` item 6), not a per-frame rate along a trajectory. The right
home eventually, but that is a new function upstream, not a move.

## Order of work, cheapest first

1. Delete the dead half of `structure.py`. No dependency on anything.
2. Adopt `_kernels.density2points` and drop both QuEst copies.
3. Adopt `random_distances` + `av_pair_statistics` for ⟨R_DA⟩ and ⟨R_DA⟩_E,
   keeping `dRmp` local until upstream is fixed.
4. Fix the two imp-tricks lines; then delete `imp_av.py` and QuEst's `AV`.
5. Move the FRET rate kernel up once κ² is wanted.

# Do not install it — work from `src`

imp-tricks is under active development, so the **source tree is what has to
work**; an installed copy goes stale the moment the checkout changes. A real
`pip install` was tried and reverted for exactly that reason.

# Reaching a checkout from the live interpreter

`quest.imp_tricks.enable_imp_tricks()` splices a checkout in at runtime
(`~/dev/imp-tricks/src`, or `$IMP_TRICKS_SRC`), returning the subpackages it
made importable. **It runs automatically** when `quest.lib.imp_av` is imported —
which every QuEst entry point does — so using QuEst is enough to reach
`IMP.bff.av`, `IMP.bff.distance_metrics` and `IMP.cgmol.*` from source, in both
environments. It is never fatal: no checkout means an empty list.

This mirrors what imp-tricks does for itself: it ships a `sitecustomize.py`
that extends `IMP.__path__` when `src` is on `PYTHONPATH`, which is how ChiSurf
consumes it. That mechanism is enough for subpackages IMP does not ship
(`IMP.cgmol`, `IMP.finite`, …) but **not** for `IMP.bff`, because a compiled
regular package always beats a namespace directory of the same name no matter
which comes first on the path. Reaching `IMP.bff.av` needs the *subpackage's*
`__path__` extended too, which is what the helper adds.

**`PYTHONPATH` alone is not enough.** imp-tricks ships `IMP/<name>/` as
*namespace* directories. In the base environment, where nothing else provides
`IMP.bff`, adding `src` to the path works. In the canonical `arm64`
environment, `IMP` and `IMP.bff` are **compiled regular packages** with their
own `__init__.py`, and a regular package shadows a namespace directory of the
same name completely — `import IMP.bff.av` fails no matter what is on the path.

The helper therefore appends to the *already-imported* package's `__path__`,
which is where Python looks for submodules. The checkout's `IMP.bff.av` and
`IMP.bff.distance_metrics` then resolve **alongside** the compiled
`IMP.bff.AV`, not instead of it. Verified in both environments; idempotent; a
missing checkout returns `[]` rather than raising.

# The capability check is looking for the wrong thing

`quest/lib/imp_av.py` decides whether the preferred backend exists with:

```python
HAS_IMP_BFF = hasattr(IMP.bff, "AV")
```

`IMP.bff` here is the **namespace package from imp-tricks**, not a compiled
module — with `PYTHONPATH=~/dev/imp-tricks/src` it imports fine, exposes
`IMP.bff.av.compute_av` and `IMP.bff.distance_metrics`, and has **no top-level
`AV` attribute**. So QuEst concludes "IMP.bff is unavailable" and falls back to
LabelLib even when imp-tricks is present and could serve the request.

Two separate things to fix, in this order:

1. **Install imp-tricks into the environment** (it is not installed here; only
   the path trick works), and
2. **ask for the capability, not the attribute** — call
   `IMP.bff.av.compute_av`, which already performs the backend dispatch itself.

Until then the LabelLib path is what actually runs — now **by explicit default**
rather than by accident, since QuEst's own IMP integration turned out to return
an empty accessible volume when it finally got selected in `arm64`. That is the
strongest argument for this whole document: the duplicated integration was
broken and nobody could tell, because the environment that would have exercised
it was never the one being tested. See
[references/known-issues.md](/references/known-issues.md).

# Rules for the migration

1. **Move, don't copy.** A symbol lands in `IMP.bff` and QuEst imports it; two
   maintained copies is the state being fixed, not a milestone toward it.
2. **Keep the LabelLib fallback reachable.** QuEst must still run where the
   compiled IMP is absent — that is the current development environment.
3. **Migrate behind QuEst's own names first** where a signature differs, so the
   swap is one edit and one test run rather than a rewrite of call sites.
4. **The tests move with the code.** `tests/test_grid_kernels.py` and
   `tests/test_fret_physics.py` pin behaviour that will live in imp-tricks;
   their assertions belong beside the implementation, with QuEst keeping only
   an integration-level check.
5. **Nothing moves without a green run on both sides** — QuEst's suites and
   imp-tricks' (`PYTHONPATH=src` from its root).

# Where the FRET rate trace should land

`fret_rate_trace` computes `k_FRET(t) = (1/tau0)·⟨(R0/|r_D(t) − r_A|)^6⟩_A`
for a donor trajectory against an acceptor cloud. `IMP.bff.distance_metrics`
already owns the static counterparts, and `IMP.bff.cgdye.analysis.fret` owns
κ²-aware rates and exact efficiencies from rate matrices — **that module is the
natural home**, and merging into it would bring κ² with it, which QuEst's
implementation lacks ([objectives.md](objectives.md) OBJ-01 item 6).

Tracked as `DUP-03` in [assessment.md](assessment.md).
