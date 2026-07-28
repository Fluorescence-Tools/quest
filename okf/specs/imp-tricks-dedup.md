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

| what | upstream | evidence | state |
|---|---|---|---|
| `density2points` | `IMP.bff.av._kernels.density2points` | Selects exactly the same voxels on 5×7×9, 8×8×8 and 11×11×11 grids. Theirs is a strict superset: it keeps the per-voxel weight, which QuEst throws away. | **`DUP-02` done 2026-07-28** |
| `RDAMean`, `RDAMeanE`, `_ran_dist` | `random_distances` + `av_pair_statistics` | Agree within sampling noise on 148l E15/E90: ⟨R_DA⟩ 55.164 ± 0.037 vs 55.171 ± 0.032 Å, ⟨R_DA⟩_E 54.467 ± 0.022 vs 54.473 ± 0.029 Å. | **`DUP-03` done 2026-07-28** |

Neither is blocked by anything. `IMP.bff.av._kernels` and
`IMP.bff.distance_metrics` import and run in `arm64` today.

### What `DUP-02` actually cost

The stale claim above — "two copies" of `density2points` — was already wrong
when it was written: an earlier change had merged the flat/centre-origin and
3-D/corner-origin variants into one kernel plus
`_points_from_centred_grid`. Only one copy was deleted.

Two things had to be fixed before the swap was a swap rather than a regression:

- **Upstream's kernels were bare `@nb.njit`.** QuEst's carried
  `cache=True, nogil=True`. `nogil` is what keeps a threaded site scan — the web
  backend — from serialising on the GIL in the loop that visits every voxel;
  `cache` is what stops a short CLI run paying a fresh compile. Contributed
  upstream (`a49e17c`) rather than worked around: the annotations belong with the
  kernel, not with one of its callers.
- **The enumeration order differs**, so the point cloud comes back permuted. It
  is safe here only because nothing reads `AV.points` positionally — checked
  rather than assumed. Rule for the rest of this migration: **an upstream kernel
  that returns a collection may not return it in the same order**, and that has
  to be verified against every consumer before the swap, not after.

### What `DUP-03` cost: a 5× slowdown, taken knowingly

`RDAMean`/`RDAMeanE` now aggregate `random_distances` + `av_pair_statistics`,
and `_ran_dist` is gone. The numbers agree inside sampling noise, but upstream
costs **3.4 ms a call against 0.68 ms**, for two structural reasons that will
recur elsewhere in this migration:

- QuEst's kernel was `parallel=True`; upstream's is **sequential because it
  seeds**, and numba's per-thread RNG cannot be seeded reproducibly. Determinism
  and parallelism are in direct conflict in this kernel, and upstream chose
  determinism — correctly, for a *reported* distance. QuEst gains reproducibility
  it did not have.
- Upstream's signature takes `(N, 4)` weighted points, so a binary QuEst cloud
  has to be widened first (0.87 ms of the 3.4). The weights are then all ones and
  the weighted mean reduces to the plain one.

Accepted because this runs **once per simulation** (~0.4 s across a 151-site
scan). **The general rule this establishes: check where the upstream symbol sits
in the call graph before adopting it.** The same trade on a per-frame path — the
FRET rate trace, the trajectory kernels — would be unacceptable, and those are
exactly what this document proposes moving next.

A second, smaller lesson: `TestTheAdoptedDistanceKernelsBehave` first used
E15/E90, the pair quoted throughout this document, and errored in the base
environment. Any test that must pass in both has to pick sites labelable under
**both** backends, and E15 is not among LabelLib's 43 of 151. It uses E55/E124.

And one that surfaced only by running it: importing `IMP.bff.av._kernels` at
module scope broke `test_importing_the_domain_does_not_import_qt` and
`test_the_model_needs_no_qt`, via the dylib conflict in
[references/known-issues.md](/references/known-issues.md). **Every adoption in
this document has to be imported lazily**, or the domain layer stops being
importable without IMP — which is the property that lets the web backend run in
the base environment at all.

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

## Declared 2026-07-28 — imp-tricks is a dependency, and it does not resolve yet

`IMP.bff.av` resolves in **both** environments once the checkout is on the path —
including the base interpreter, where there is no compiled `IMP.bff` to shadow
it. So the barrier was never "it does not work"; it was that imp-tricks was a
development path shim (`quest/imp_tricks.py`) rather than a declared dependency.

**It is now declared**, on the owner's instruction that installability is not a
constraint ("nobody installs quest; imp-tricks goes public later"). Declared in
`rattler-recipe/recipe.yaml` (run) and `pixi.toml` (`feature.py.dependencies`),
**not** in `pyproject.toml` — which is the convention `imp` itself already
follows here, because neither is pip-installable. A `pip install quest` has
never produced a working science stack and still does not.

**Consequence, stated plainly:** `imp-tricks` is on a private GitLab and is on
no public channel, so every pixi environment in the `py` solve-group now fails
to solve, and with it **all four CI jobs** — library, GUI, frontend-adjacent and
packaging — not merely the packaging one. This is accepted, not overlooked. The
fix is publication, and then nothing else: the version constraint is `*`, so a
package appearing on conda-forge under the name `imp-tricks` is enough.

Day-to-day work is unaffected because it does not go through pixi: the canonical
`arm64` conda environment reaches the checkout through `quest/imp_tricks.py`
([workflows/environment.md](/workflows/environment.md)). That shim stays until
publication — it is what makes the source tree, rather than a stale install, the
thing that runs.

## Superseded — the original diagnosis, for reference

`AV` (in `av.py`) and `imp_av.py` should both become
`IMP.bff.av.compute.compute_av`, whose signature is already the right shape —
plain arrays (`atoms_xyz`, `atoms_vdw`, `source_xyz`), no IMP particles.

**Unblocked 2026-07-27.** Both backends now work and both are verified against
QuEst's, so the precondition for `DUP-01`/`DUP-02` is met. What it took, in
imp-tricks (commits `936a17c` and `b95c000`):

*LabelLib branch* — the availability probe required `LabelLib.AV`, which current
builds do not ship (they expose `dyeDensityAV1`); the attachment atom kept its
own exclusion sphere, so the linker started inside an obstacle and the backend
returned an empty volume with no error; and `Particle.get_particle_index()` does
not exist in IMP 2.24, which spells it `get_index()`.

*IMP.bff branch* — **had never run at all.** It raised
`UsageException: Can't get attribute that is not there: mass` on every input,
because it read its density by handing the source particle to an
`IMP.em.SampledDensityMap` that needs an `IMP.atom.Mass` it never set — and with
a mass, that samples a Gaussian blob around the attachment point rather than the
accessible volume. It also never passed `allowed_sphere_radius`, decorated the
AV onto the source particle itself and resampled with `shift_xyz=False` (leaving
the map at the coordinate origin), and reshaped the flat tile values in the
wrong axis order.

Acceptance: upstream and QuEst return point-for-point identical clouds on 148l
E118, E36 and E90 under **both** backends, pinned by
`TestTheUpstreamAvMatchesQuEsts`, which is now parametrised over both.

**Done 2026-07-28.** `quest/core/imp_av.py` (329 lines) and `calculate1R` (69)
are deleted; `AV.__init__` makes one `compute_av` call with an explicit
`backend=`. AV numbers verified unchanged on 148l E36/E55/E118 under both
backends — identical grid shape, identical binary density voxel-for-voxel,
identical point counts. What QuEst keeps is the application: site selection, the
cubic/binarisation adapter, and `AV`'s derived grids.

One blocker was found only by looking: `compute_av` had **no thread safety**, so
adopting it would have re-introduced the SWIG race QuEst had just fixed.
Contributed upstream (`9cd5c13`) with the same construction/resample split, so
the lock moved rather than vanished.

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
