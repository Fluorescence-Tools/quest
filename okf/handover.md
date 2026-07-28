---
type: Handover
title: Session handover
description: Where the tree stands, what is uncommitted, which decisions are open, and the traps that cost time — written to be the first thing a fresh session reads after the index.
resource: okf/log.md
tags: [handover, state, open-questions]
timestamp: '2026-07-28T00:00:00Z'
---

# Read this, then [`log.md`](log.md)

**Both repositories are committed and clean, and nothing has been pushed.**
QuEst: decision 4 and the deduplication landed as six commits
(`52db006`, `78dc4c8`, `9b1192e`, `80d39fe`, `da2774d`, plus this one) on top of
the thirteen already recorded. imp-tricks: three more (`a49e17c`, `9cd5c13`,
after `936a17c`/`b95c000`), still unpushed.

**All four decisions are now closed.** Decision 4 was unblocked by the owner
rather than by publication — installability is explicitly not a constraint
("nobody installs quest, will make imp-tricks public later") — and the
deduplication it gated is done: `DUP-01`, `DUP-04` and half of `DUP-05` in
[specs/assessment.md](specs/assessment.md).

**CI is red, deliberately.** Declaring `imp-tricks` makes the `py` solve-group
unsolvable until it reaches a public channel, and that takes *all four* jobs
with it, not just packaging. The constraint is `*`, so publication alone fixes
it with no further change here.

# State

| | tests | note |
|---|---|---|
| `arm64` (Qt + compiled IMP.bff) | **358 passed, 4 skipped** | canonical interpreter |
| base + web backend | **319 passed, 27 skipped** | no Qt, no compiled `IMP.bff` |
| Qt GUI | **16 passed** | `QT_QPA_PLATFORM=offscreen`, `arm64` |
| notebooks | **9 passed** | `tests/test_notebooks.py` alone — see below |
| examples | **7 passed** | `tests/test_examples.py` alone — see below |
| imp-tricks `tests/bff` | **198 passed, 5 failed** | the 5 predate all of this |
| browser e2e | 26 passed | *not re-run since the AV changes* |
| ChiSurf `quenching_estimator` | 1 passed | *not re-run since the AV changes* |

The five QuEst numbers above were re-measured after every commit in this
session. `DUP-01` additionally verified the **volumes themselves** rather than
trusting the suite: 148l E36/E55/E118 under both backends, identical grid shape,
identical binary density voxel-for-voxel, identical point counts.

The notebooks and examples pass **one file per process**. Run as a single
`pytest tests -m notebooks` they give 6 failed / 10 passed, from a dynamic
loader conflict in the child process — pre-existing, verified at `1807e72`,
and written up in [references/known-issues.md](references/known-issues.md).

The two italicised rows are still the previous session's numbers. They have now
gone two sessions without being re-run, across changes that touched every
accessible volume — treat them as unverified.

All runs above were on a **cold numba cache**. That matters; see Traps.

## Committed in `quest` (138 paths, nine commits)

Five threads landed this session, in order — see [`log.md`](log.md) for which
commit carries which:

1. **AutoForm migration.** Every `.ui` file is gone. The Qt form is
   `quest/gui/quest.view.json`, *generated* from the parameter catalog by
   `quest/gui/generate_view_spec.py`, rendered by ChiSurf's AutoForm, bound to
   `quest/gui/form_model.py::ProjectFormModel` — whose state *is* a project
   dict, so the GUI can no longer drift from the CLI.
   [subsystems/gui.md](subsystems/gui.md).
2. **i18n — English, German, French.** `quest/i18n.py` plus
   `quest/settings/locales/*.json`. Because the catalog is the single
   description, translating it reaches the docs, the Qt form and the web UI at
   once. [subsystems/i18n.md](subsystems/i18n.md).
3. **FRET: the acceptor diffuses and sticks**, with its own `D`, `slow_radius`
   and interaction table; `k_FRET(t)` uses the instantaneous pair separation.
   The old fast-acceptor limit remains as `fret.acceptor_dynamics: "averaged"`.
4. **`render_data` split from `save_outputs`** — they were one flag doing two
   jobs, which cost the Qt form its plots.
5. **Structure I/O reduced to one reader**, taking IMP's radii.
   [specs/structure-io.md](specs/structure-io.md). This is the one that changes
   numbers; see Decisions.

## Committed in `~/dev/imp-tricks` (two commits, unpushed)

Contributed upstream while unblocking QuEst:

- `src/IMP/bff/av/compute.py` — three fixes: `get_particle_index()` →
  a shim accepting `get_index()`; the LabelLib probe accepting `dyeDensityAV1`
  (current builds have no `LabelLib.AV`, so a working install was reported
  missing); and the attachment atom's own vdW sphere left in the obstacle list,
  which made the LabelLib backend return an **empty volume with no error**.
- `src/sitecustomize.py` — extends the `__path__` of subpackages that are
  themselves compiled. Without it `import IMP.bff.av` fails however `sys.path`
  is arranged, and imp-tricks' own suite could not collect (28 errors).
- `tests/bff/test_label.py`, `tests/bff/test_restraints.py` — two tests
  asserted `ImportError` "without a backend" and passed only *because* of the
  LabelLib probe bug. They now monkeypatch the flags.

The 5 remaining imp-tricks failures (`cgdye/rotamer` R0-vs-FRETpredict,
`dye_io` PDB→mmCIF, `polymer` Gaussian-chain PDF) reference none of the changed
code and had never been collected before. Newly *visible*, not newly broken.

# Decisions taken 2026-07-27 — **all four closed**

Decisions 1-3 landed 2026-07-28 with a re-baseline between each; decision 4
followed once the owner ruled installability out of scope.

1. **`IMP.bff` is the primary AV backend.** Done (`1b997ec`). Promoting it
   surfaced three defects that had been invisible for as long as nothing entered
   the branch — a **transposed** density grid (x↔z, which still overlapped the
   truth by 83 % of its voxels, so counts and bounding boxes looked right while
   quenching silently vanished), a grid origin one voxel out on even edge
   lengths, and no thread safety at all. Details in
   [subsystems/accessible-volume.md](subsystems/accessible-volume.md).

2. **HETATM is no longer stripped; water still is.** Done (`5b8fc2a`). Both
   candidates were measured over all 151 CB sites rather than argued: keeping
   waters as obstacles costs **17 %** of every accessible volume, which is a
   stronger claim than mobile solvent supports. Admitting ligands broke two
   things that had only ever seen amino acids — `to_coarse` and hetero atom
   names. [specs/structure-io.md](specs/structure-io.md).

3. **`allowed_sphere_radius` defaults to 2.1 Å.** Done (`4f3dad2`). It also
   **corrected a claim made in decision 1**: the labelability gap between the
   backends is not mostly this parameter. Raising the default left LabelLib at
   exactly 43/151, because `calculate1R`/`calculate3R` accept `linkersphere` and
   never pass it on — LabelLib has no such knob and frees only the attachment
   atom instead. Only the IMP.bff backend honours it.

4. **imp-tricks is a declared dependency — done, and CI is red for it.**
   Declared in `rattler-recipe/recipe.yaml` and `pixi.toml`, deliberately not in
   `pyproject.toml`: that is the convention `imp` already follows, since neither
   is pip-installable and a pip install of QuEst has never yielded a working
   science stack. It was blocked on publication; the owner cleared that, so the
   declaration lands now and publication catches up later. The path shim in
   `quest/imp_tricks.py` stays — it is what makes the *source tree* the thing
   that runs, which matters while imp-tricks is changing.

# What the deduplication actually cost

All three adoptions were verified numerically, and none of them was a drop-in.
The recurring lesson: **an upstream symbol is not equivalent just because its
numbers match** — check its concurrency annotations, its return order, and where
it sits in the call graph first.

- **`DUP-01`** (`imp_av.py` + `calculate1R` → `compute_av`, `da2774d`).
  `compute_av` had **no thread safety**, so adopting it would have re-introduced
  the SWIG race QuEst had just fixed. Lock contributed upstream (`9cd5c13`).
- **`DUP-04`** (⟨R_DA⟩, `9b1192e`, filed as "DUP-03" in the commit — see the
  numbering warning below). Upstream is **5× slower**: sequential because it
  seeds, where QuEst's was `parallel=True`. Taken knowingly — it runs once per
  simulation — and it buys reproducibility QuEst did not have.
- **`DUP-05`, in part** (`density2points`, `78dc4c8`, filed as "DUP-02").
  Upstream's kernels were bare `@nb.njit`; adopting them as-is would have
  re-serialised threaded scans on the GIL. `cache=True, nogil=True` contributed
  upstream (`a49e17c`). Its enumeration order also differs, so the cloud comes
  back permuted — safe only because nothing reads `AV.points` positionally,
  which was checked against every consumer.

**Numbering warning.** [`specs/assessment.md`](specs/assessment.md) is the
authoritative `DUP-*` table and **this handover previously used different
numbers**. Commits `78dc4c8` and `9b1192e` follow the old ones and are
mislabelled: they are `DUP-05` and `DUP-04`. Cite the assessment table.

# Next steps, in order

1. **Publish imp-tricks.** It is no longer a blocker for code — everything it
   gated is done — but it is what turns CI green again, and five commits are
   sitting unpushed on a private GitLab. Nothing else here is waiting on it.
2. **Re-run the two unverified rows**: the browser e2e suite (start both
   servers; restart uvicorn first, it does not reload) and the ChiSurf
   `quenching_estimator` plugin. They have now gone two sessions without a run,
   across changes that moved every accessible volume.
3. **The half-voxel offset on the LabelLib path**
   ([references/known-issues.md](references/known-issues.md)): `AV.points` sits
   0.25 Å per axis off its own density grid for even `ng`. Needs a decision on
   *which* convention is right — the float corner is geometrically true, the
   integer one is what every grid kernel indexes with — and then its own
   re-baseline, because it moves LabelLib results.
4. **The coarse-grain chain bug**, same file: `move_center_of_mass` matches
   residues by `res_id` alone, so in any multi-chain structure one chain's side
   chains fold into another's CB. Also moves numbers; also wants its own change.
5. Remaining dedup rows: `DUP-02` residue (the capability check), `DUP-03`
   (`fret_rate_trace` → `IMP.bff.cgdye.analysis.fret`, which would bring κ² with
   it), the rest of `DUP-05` (`subav`, `asa`), `DUP-06`, `DUP-07`.
   **Do not** adopt upstream's `r_mp`: it returns ⟨R_DA⟩ under that name.
   QuEst's `dRmp` is correct and should go *upstream*.
6. Backlog: `LAY-04` (`quest.api` is still a re-export shim), `LAY-07` (three
   unrelated long-running mechanisms, none cancelable), `OBJ-01` item 6 (κ²
   unmodelled), `OBJ-02` (fitting — not started, by instruction).
7. Web UI chrome is still partly English — plot card titles, metric row names,
   empty states, toolbar. `t()` is wired; it is one key at a time.

# Traps that cost time here

- **A warm numba cache hides compile-time failures.** `@njit(cache=True)`
  kernels only re-read configuration when they actually compile, so a green
  suite proves nothing after touching a jitted module. Clear it:
  `find quest -name '__pycache__' -type d -exec sh -c 'rm -f "$1"/*.nbi "$1"/*.nbc' _ {} \;`
  `tests/conftest.py` now pins `NUMBA_NUM_THREADS` and `NUMBA_THREADING_LAYER`
  for exactly this reason.
- **`Structure(path)` defaults to `make_coarse=True`.** A scratch script
  comparing coordinates against a PDB file will look 1.5 Å wrong. Pass
  `make_coarse=False`.
- **uvicorn does not reload.** Restart it before any browser run. Locale files
  are `lru_cache`d per process, so editing one has no effect until restart.
- **The frontend needs Node ≥ 20**; `/opt/homebrew/bin/node` is 26, the one on
  `PATH` is 18.15 and Next refuses to start.
- **`-p no:pytest-qt` only where there is no Qt binding.** The plugin aborts
  collection in the base environment, so the flag is required there. In `arm64`
  pytest-qt *is* installed and the flag is harmful: it removes the `qtbot`
  fixture, which is what made ChiSurf's own chimol widget tests error. Passing
  it reflexively to another project's suite will look like that project is
  broken.
- **`import IMP.bff` succeeding does not mean the compiled AV exists.** In the
  base environment the *source* `IMP.bff` imports (nothing shadows it) while the
  C++ `AV` class is absent. Guard on `quest.core.imp_av.HAS_IMP_BFF`.
- **A concurrency or grid test that has never been *seen* to fail is not
  evidence.** Three times now: the upstream grid test derived its point cloud
  from the density it was checking; the first thread-safety test used 165
  atoms over 4 workers and passed with the lock replaced by a no-op (it takes
  ~2000 atoms over 8); and `TestDensityToPoints` pinned an enumeration order
  nothing depended on. Defeat the mechanism and watch the test go red before
  believing it.
- **Verify that an edit actually applied.** Three `str.replace` calls in a row
  silently no-oped here because the anchor text had already changed, and it only
  surfaced as a test failing on a stale value.
- **A backend with no coverage will ship broken code.** `imp_av.py` carried a
  `Model.get_particles()` call that does not exist; nothing noticed because
  `AV_BACKENDS` starts with LabelLib and no test entered that branch.
