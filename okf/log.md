# Update Log

## 2026-07-28

* **`DUP-03` done: the RDA sampling is upstream's, and it is 5× slower.**
  `RDAMean`/`RDAMeanE` now aggregate `IMP.bff.av._kernels.random_distances` +
  `av_pair_statistics`; `_ran_dist` is deleted. Numbers agree inside sampling
  noise (⟨R_DA⟩ 55.164 ± 0.037 vs 55.171 ± 0.032 Å on 148l E15/E90), but the
  call costs **3.4 ms against 0.68 ms**: QuEst's kernel was `parallel=True`
  while upstream's is sequential *because it seeds*, and numba's per-thread RNG
  cannot be seeded reproducibly — determinism and parallelism genuinely conflict
  here. Taken knowingly: this runs once per simulation, ~0.4 s over a 151-site
  scan, and QuEst **gains reproducibility it did not have** (the old kernel gave
  a different answer on every call). The rule it establishes: check where an
  upstream symbol sits in the call graph before adopting it — the same trade on
  the per-frame FRET rate path would be unacceptable, and that is what the spec
  proposes moving next.
* **`dRmp` stays, and index 1 of `av_pair_statistics` is now documented as
  unreadable.** Upstream returns `r_mp` as a copy of `r_da_mean`; R_mp cannot be
  recovered from a sample of pair distances at all. QuEst reads indices 0 and 2.
* **The new tests had to change site to exist in both environments.**
  `TestTheAdoptedDistanceKernelsBehave` first used E15/E90 — the pair this
  project's own spec quotes everywhere — and errored under LabelLib, which
  labels 43 of 148l chain E's 151 CB sites and not that one. Now E55/E124, the
  widest-separated pair labelable under **both** backends (64.0 Å). Same class
  of mistake as the `QUENCHED_SITES` re-pick earlier: a number measured under
  one backend is not a fixture for a suite that runs under two.
* **`DUP-02` done: `density2points` is upstream's.** QuEst's `_density2points`
  is deleted; `quest.core.av.density2points` is now an adapter over
  `IMP.bff.av._kernels.density2points`. Same voxels on 5×7×9, 8×8×8 and
  11×11×11 grids. Two preconditions had to be created first, neither of which
  the plan mentioned: upstream's kernels were bare `@nb.njit`, so adopting them
  as-is would have **re-serialised threaded site scans on the GIL** in the loop
  that visits every voxel (fixed upstream, `a49e17c`, contributing QuEst's
  `cache=True, nogil=True`); and upstream enumerates `ix` outermost where QuEst
  enumerated `iz`, so the cloud comes back **permuted** — safe only because
  nothing reads `AV.points` positionally, which was checked against every
  consumer rather than assumed. General rule for the rest of the migration: an
  upstream kernel returning a collection may not return it in the same order.
* **A lazy import is now a load-bearing constraint on the domain layer.**
  Importing `IMP.bff.av._kernels` at the top of `quest/core/av.py` failed
  `test_importing_the_domain_does_not_import_qt` and `test_the_model_needs_no_qt`
  — not with an `ImportError` but with the *documented* dylib conflict, where a
  subprocess of a process that has loaded IMP resolves `libimp_kernel` against
  the base environment's `libboost_filesystem`. Both tests probe the domain layer
  in a subprocess, so a module-scope import promoted a latent environment fault
  into a hard failure of the service layer. Every future adoption from imp-tricks
  has to be imported on first use, or the web backend stops being able to run in
  the base environment. Noted in
  [references/known-issues.md](references/known-issues.md) that this is a
  route-around, not a fix.
* **Decision 4 taken: imp-tricks is a declared dependency.** Unblocked by the
  owner rather than by publication — installability is explicitly not a
  constraint here. Declared in `rattler-recipe/recipe.yaml` and `pixi.toml`,
  deliberately **not** in `pyproject.toml`, which is the convention `imp` itself
  already follows: neither is pip-installable, so a pip install of QuEst has
  never yielded a working science stack. **This turns every pixi environment in
  the `py` solve-group unsolvable, and therefore all four CI jobs red**, until
  imp-tricks reaches a public channel — the constraint is `*`, so publication
  alone fixes it. Recorded rather than worked around; day-to-day work goes
  through the `arm64` conda environment and the checkout shim, not pixi.
  [specs/imp-tricks-dedup.md](specs/imp-tricks-dedup.md).
* **Decisions 1–3 implemented, one at a time, with a re-baseline between each**
  — `1b997ec` (IMP.bff primary), `5b8fc2a` (keep HETATM, strip water),
  `4f3dad2` (`allowed_sphere_radius` 2.1 Å), `ea74ca9` (pin against upstream).
  Decision 4 is **blocked on packaging, not code**; see
  [handover.md](handover.md).
* **Promoting a backend nobody had run found three silent defects.** The worst
  was a **transposed** density grid: IMP numbers voxels x-fastest, so the
  C-order reshape exchanged x and z. An accessible volume is globular enough
  that the transpose still overlapped the truth by **83 %** of its voxels — the
  point count, the bounding box and the total volume were all correct, and only
  the *shape* was mirrored, so the contact volume landed on the wrong side of
  the protein and donors came out unquenched. The lesson for any grid: compare
  the density and the point cloud **against each other**, and only when they
  come from independent sources. The first version of the upstream test derived
  one from the other and passed with the bug deliberately reintroduced.
  The other two: a grid origin one voxel out whenever the edge length is even
  (86 on the reference site), and no thread safety — a pooled scan, which the
  web backend runs, died in SWIG exactly as `pdb.py` already locks against.
* **A claim from decision 1 was wrong, and decision 3 disproved it.** Decision 1
  recorded that most of the 43-vs-147 labelability gap between the backends was
  `allowed_sphere_radius` rather than the algorithm. Raising the default from
  0.5 to 2.1 Å left LabelLib at **exactly 43/151** — because
  `calculate1R`/`calculate3R` accept a `linkersphere` argument and never pass it
  on. LabelLib has no such parameter; it frees the attachment atom's own radius
  and nothing else, so a CB hemmed in by CHARMM-sized neighbours has no route
  out. Corrected in the constant, in
  [subsystems/accessible-volume.md](subsystems/accessible-volume.md) and in the
  commit message.
* **Both imp-tricks AV backends now work** (`936a17c`, `b95c000`, unpushed).
  The `imp_bff` branch had **never produced a volume in its life**: it read its
  density by handing the source particle to an `IMP.em.SampledDensityMap`
  needing a `Mass` it never set, so every call raised — and with a mass it would
  have sampled a blob around the attachment atom rather than the path map. Plus
  no `allowed_sphere_radius`, the AV decorated onto the source particle itself
  with `shift_xyz=False` (map left at the coordinate origin), and the same
  transpose. `TestComputeAvBackends` is new upstream; there had been no test
  touching `compute_av` at all, which is why a backend that could not work
  looked fine. Upstream and QuEst now agree point-for-point on 148l E118, E36
  and E90 under both backends.
* **Decisions have consequences at the edges, and they surfaced as errors.**
  Admitting HETATM broke `to_coarse` (`KeyError: 'FGA'` — untemplated residues
  are now kept whole, since a bound sugar has no backbone-plus-CB reduction) and
  revealed that every hetero atom was named `"HET:"`, IMP's prefix truncated by
  the `|U5` field. Both are in
  [specs/structure-io.md](specs/structure-io.md).
* **Two defects found and deliberately *not* fixed**, both recorded in
  [references/known-issues.md](references/known-issues.md) because fixing either
  changes numbers and deserves its own re-baseline: coarse-graining matches
  residues by `res_id` alone, so in any multi-chain structure one chain's side
  chains are folded into another's CB; and `pytest tests -m notebooks` fails as
  a single selection (6 failed / 10 passed) while the two files pass separately
  (9 + 7), from a dynamic-loader conflict in the child process. The second is
  **pre-existing** — verified in a worktree at `1807e72` — and means the
  previous handover's "16 passed" never came from that one command.

## 2026-07-27

* **The work is committed, in nine pieces.** Everything that had accumulated in
  the working tree — 138 paths, five threads — is now history rather than
  uncommitted state. Nothing is pushed. The split, oldest first:
  `5962ce3` .gitignore; `64aaebe` the OKF bundle and `CLAUDE.md`; `92ce43b`
  `quest.lib` → `quest.core`; `d4d46d2` the service layer; `fde6731` the
  parameter catalog and i18n; `790dc1d` the AutoForm GUI; `f34e9a2` the web UI;
  `7f51688` notebooks and examples; `1fa4226` pixi and rattler-build.
  Deliberately **not** bisectable: the tree was restructured wholesale, so the
  middle commits do not stand alone. They are split to be *read*, and each
  message records why the change was made rather than what the diff shows.
  Two things were found while staging and fixed rather than committed:
  `jobs/` had reached **2292 run directories and 32 GB** (now ignored, along
  with the `out*.xyz` / `test.pdb` / `test.cif` / `sim_test.log` /
  `webui-viewer-testpdb.png` leftovers), and `tests/data/148l.pdb` was a
  byte-identical duplicate of `tests/148l.pdb` used only by `test_fps_json.py`,
  which now reads the top-level copy. The index also carried a stale
  `quest/core_new/` snapshot from an abandoned rename; it was superseded by
  `quest/core/` and was cleared before staging.
* **Four architectural decisions taken, none implemented yet** — written up in
  [handover.md](handover.md) as the next session's work list: `IMP.bff` becomes
  the primary AV backend with LabelLib the Windows fallback; HETATM and water
  are no longer stripped; `allowed_sphere_radius` defaults to 2.1 Å (just above
  the ~2 Å cliff below which IMP's path search returns an empty volume without
  complaining); and imp-tricks becomes a declared dependency, which unblocks
  `DUP-01`–`DUP-03`. Decisions 1–3 each change the accessible volume, so they
  must land one at a time with a re-baseline between them, or the number
  changes cannot be attributed. Decision 4 depends on the upstream fixes in
  `~/dev/imp-tricks` being committed and released first.
* **Radii now come from IMP, and the structure layer is one thin reader**
  (`quest/core/pdb.py`, per [specs/structure-io.md](/specs/structure-io.md)).
  `read_pdb` already decorates every particle with a radius, so the reader takes
  `IMP.core.XYZR(p).get_radius()` and `IMP.atom.Mass(p).get_mass()` directly —
  no helper, no fallback table, no `add_radii` call (redundant: the parser has
  already done it). `quest/core/elements.py` is **deleted** (271 lines), and
  with it the last table that made the two AV backends disagree. The dead
  `assignCharge` path went too — no caller ever set it.
  **This changes the science.** IMP's radii are CHARMM-style
  (`get_all_atom_CHARMM_parameters`): C 2.10, N 1.85, O 1.71 against the
  Bondi-ish 1.76 / 1.65 / 1.40 QuEst carried. ~19 % fatter atoms, ~72 % more
  excluded volume, and accessible volumes shrink accordingly. On 148l chain E
  **only 43 of 151 CB sites remain labelable** — pockets that were open at
  Bondi radii are closed at CHARMM ones. Burial is also resolution-dependent: a
  site can be labelable at `dg = 1.0` and buried at `0.5`, because the coarser
  grid cannot resolve the atoms that close it (mc4r A1).
* **The two AV backends now converge.** Fed the same atom array with the same
  radii, IMP.bff and LabelLib give 136 707 vs 151 869 points on 148l E36 —
  ratio **0.90**, centroids within 0.5 Å. Before, with QuEst's own radii on one
  side and IMP's on the other, the ratio was 0.78 and it was misread as an
  algorithmic difference. The residual 10 % is the real algorithmic gap.
* **`imp_av.py` takes the atom array, not a filename.** It builds the IMP model
  from `structure.atoms`, so coarse-graining can no longer be discarded and
  mmCIF works wherever the reader works. It also shipped a
  `Model.get_particles()` call that does not exist — **an entire backend with
  no test coverage**, because `AV_BACKENDS` starts with LabelLib and nothing
  ever entered that branch. `TestTheImpAvBackendRuns` now exercises it, and
  guards on `HAS_IMP_BFF` rather than on `import IMP.bff` succeeding: in the
  base environment the *source* `IMP.bff` imports (nothing compiled to shadow
  it) while the C++ `AV` class this backend needs is absent.
* **Test baseline regenerated for the new radii.** Sites that no longer have an
  accessible volume were replaced everywhere, and the choice is not arbitrary —
  each test needs a site that satisfies *its own* assertion:
  `test_api` asserts the quenching grid reaches 2.0 /ns with **only TYR**
  quenching, so it needs a tyrosine within `critical_distance` (36 or 82 are
  the only labelable sites that qualify); `test_simulation_performance` needs
  `0 < QY < 1` under its own chemistry (18, 96, 36 — note 118 quenches under
  the template project but not under that one); `test_fret_physics` needs a
  pair with *intermediate* transfer so the two acceptor regimes can differ
  (55→82: E 0.73 trajectory vs 0.77 averaged; the previous pair saturated at
  0.95 and the regimes agreed to within the tolerance).
* **A process note.** Three `str.replace` edits in a row silently no-oped
  because the anchor text had already changed, and I only noticed when the test
  kept failing on the old value. Assert the replacement changed the file, or
  read the line back.
* **Tried to make IMP.bff the primary AV backend. It is not ready, and now we
  know exactly why.** Flipping `AV_BACKENDS` and running the suite in `arm64`
  produced four failures that are all one root cause:
  `quest/core/imp_av.py::build_imp_accessible_volume` **re-reads the structure
  file from disk** (`IMP.atom.read_pdb`) instead of using the structure it was
  handed. Consequences, both measured:
  - a coarse-grained project silently gets the **all-atom** volume — coarse and
    all-atom came out byte-identical at 13883.805 Å³, which is what
    `test_coarse_grained` caught;
  - `.cif` projects fail outright, because that re-read is `read_pdb` only
    (`test_mc4r`: "No molecule read from file … mc4r_dimer.cif").
  Patching `imp_av.py` is the wrong fix. `IMP.bff.av.compute.compute_av` takes
  plain arrays (`atoms_xyz`, `atoms_vdw`, `source_xyz`) and therefore *cannot*
  re-read a file or miss an in-memory transformation. Routing through it is the
  correct end state; it needs its own `imp_bff` backend fixed upstream first,
  which currently resamples a density of the *attachment particle* rather than
  the AV, and dies on a missing `mass` attribute before it gets there.
  Order reverted to LabelLib-first; four tests in
  `test_imp_tricks_bridge.py::TestTheImpBackendIsNotReadyToBePrimary` pin the
  two defects so the ordering cannot stay wrong by inertia.
* **The empty IMP volume is solved: `allowed_sphere_radius`.** OKF had carried
  this as "a third fault unresolved" since 2026-07-27. It is the radius around
  the attachment atom inside which obstacles are ignored so the linker can leave
  the atom it is tied to; at QuEst's default of 0.5 Å the attachment atom's own
  neighbours block every starting voxel and the path search returns **an empty
  volume with no error**. 2.0 Å gives 108 771 points. `MIN_IMP_ALLOWED_SPHERE_RADIUS`
  now floors it. Same physical issue as the LabelLib attachment-radius bug fixed
  upstream earlier today — LabelLib has no such parameter and zeroes the
  attachment atom's radius instead.
* **The two backends disagree by ~20 % on volume.** 148l E15 all-atom, linker
  21.5/0.5, dye 3.5 Å, dg 0.5: IMP.bff ~14 500 Å³, LabelLib ~17 500 Å³, with
  centroids within 0.6 Å — the same place, less of it. Widening
  `allowed_sphere_radius` to 6 Å moves IMP only to 14 512 Å³, so it is an
  algorithmic difference, not a parameter one. It propagates to the contact
  fraction and hence to the quenching, so promoting IMP.bff changes results and
  must not be done quietly.
* **A false alarm worth recording.** `Structure.__init__` defaults to
  `make_coarse=True`, so `Structure(path)` in a scratch script silently returns
  a coarse-grained structure — which made QuEst's CB for 148l E15 look 1.5 Å
  off the PDB. It is not: with `make_coarse=False` it matches the file exactly
  (3.097, 47.598, 53.650), and the simulation path always passes the project's
  flag explicitly. The AV comparison had to be redone apples-to-apples.
* **Acted on the imp-tricks evaluation.** Three fixes landed *upstream* in
  `~/dev/imp-tricks`, and `compute_av` now reproduces QuEst's accessible volume
  **bitwise** (167 196 points on 148l E15, identical after sorting) in both
  environments:
  `get_particle_index()` → a shim accepting `get_index()` too; the LabelLib
  probe accepting `dyeDensityAV1` (current builds have no `LabelLib.AV`, so a
  working install was reported missing); and — the reason it returned an empty
  volume — the attachment atom's own vdW sphere left in the obstacle list, so
  the linker source sat inside an obstacle and LabelLib returned nothing
  without complaining. QuEst had always zeroed that radius.
  Also upstream: `src/sitecustomize.py` now extends the `__path__` of
  subpackages that are themselves compiled, which is why imp-tricks' own test
  suite could not import `IMP.bff.av` (28 collection errors → **191 passed, 5
  failed, 24 skipped**); QuEst had solved the same problem privately in
  `quest/imp_tricks.py`. Two of its tests asserted `ImportError` "without a
  backend" and passed only *because* of the LabelLib probe bug; they now
  monkeypatch the flags instead. The 5 residual failures (rotamer R0, dye_io,
  polymer PDF) reference none of this and had never been collected before.
  `tests/test_imp_tricks_bridge.py::TestTheUpstreamAvMatchesQuEsts` pins the
  equivalence — the precondition for deleting QuEst's `AV`.
* **What now blocks `DUP-01` is packaging, not capability.** `IMP.bff.av`
  resolves in the base interpreter too (nothing compiled there to shadow it),
  so the only thing stopping the deletion is that imp-tricks is a development
  path shim rather than a declared dependency. That is a decision about what
  QuEst depends on.
* **`quest/core/structure.py`: 623 → 345 lines.** RMSD, clustering, the torsion
  subsystem (`coord_i`, `internal_coordinates`, `phi`/`psi`/`omega`/`chi`) and
  the geometry primitives behind them had no readers. Three traps, all caught by
  *running* rather than reading: "no callers outside this module" is the wrong
  question for helpers the class itself calls; cutting a function span up to the
  next `def` swallows the module constants that follow it (`a2id`, `res2id`,
  `internal_keys`); and the cut left an orphaned `@njit` that silently decorated
  the next function written below it, which then failed to type-infer.
* **One `density2points`, not two.** QuEst carried a flat/centre-origin variant
  for its own AV and a 3-D/corner-origin one for LabelLib's grid, differing only
  by a shift. Now one kernel plus `_points_from_centred_grid`, with the argument
  order matching `IMP.bff.av._kernels.density2points` so the eventual swap is a
  thin adapter.
* **Re-measured the imp-tricks split, by running it rather than reading it**
  ([specs/imp-tricks-dedup.md](/specs/imp-tricks-dedup.md)). Four findings that
  change the plan:
  1. **The blockers are two lines, and they are upstream.** `compute_av` fails
     in `arm64` on *both* backends: `av/compute.py:162` calls
     `Particle.get_particle_index()` where this IMP has `get_index()`, and
     `av/compute.py:231` refuses LabelLib because it looks for `LabelLib.AV`
     while the installed LabelLib exposes `dyeDensityAV1`. Its signature is
     already the right shape (plain arrays, no IMP particles), so fixing those
     two lines deletes QuEst's `imp_av.py` and its `AV` class outright.
  2. **Two duplicates are adoptable today**, nothing blocking:
     `IMP.bff.av._kernels.density2points` (numerically identical to QuEst's,
     and a superset — arbitrary dimensions, keeps voxel weights, while QuEst
     carries *two* lesser copies) and `random_distances` +
     `av_pair_statistics` for ⟨R_DA⟩ / ⟨R_DA⟩_E (agree within sampling noise).
  3. **One duplicate points the other way.** `av_pair_statistics` returns
     `r_mp` = ⟨R_DA⟩ — its docstring admits "same as `r_da_mean` here" — but
     R_mp is |⟨r_D⟩ − ⟨r_A⟩| and cannot be recovered from a distance sample.
     On 148l E15/E90: QuEst 47.65 Å, upstream 51.53 Å. QuEst is right;
     `dRmp` should be contributed upstream, not deleted.
  4. **`structure.py` is mostly dead.** Zero callers anywhere in the tree for
     `rmsd`, `cluster`, `calc_internal_coordinates`, `move_center_of_mass`,
     `r2i`, `make_residue_lookup_table`, `get_residue_sequence`, and for the
     `Structure` methods `internal_coordinates`, `radius_gyration`,
     `update_dist`, `update_coordinates`, `b_factors`, `atom_types`,
     `residue_dict`, `n_residues`, `phi`/`psi`/`omega`/`chi`. That is a QuEst
     cleanup needing no upstream fix. (One apparent `.phi` hit was in a
     vendored `ngl.js` — grep the source tree, not `webui/frontend/public`.)
* **The acceptor diffuses and sticks, with its own dye properties (`OBJ-01`
  item 2).** It was a static cloud that the per-frame rate averaged over — the
  fast-acceptor limit. It now walks its own accessible volume with its own
  `D`, `slow_radius` and `amino_acid_interactions` (each defaulting to the
  donor's), and `k_FRET(t)` uses the instantaneous pair separation
  (`quest/core/av.py::fret_rate_pair_trace`). The old regime stays reachable as
  `fret.acceptor_dynamics: "averaged"`.
  **This changes what QuEst computes for existing FRET projects**: on 148l
  E15→E90 with a sticky acceptor (D = 2.5 Å²/ns, slow radius 11 Å) E moved
  0.73 → 0.78. Not a bug fix on either side — the two answer different
  questions. The cloud is weighted uniformly; a trajectory is weighted by where
  the dye dwells, and a sticky dye dwells near the surface. Which way that
  moves E depends on the geometry; there is no fixed bias, and an earlier draft
  of the docstring claimed one until the numbers said otherwise.
  The acceptor's walk carries no quenching: QuEst reports the *donor's* decay,
  and the acceptor's own PET would change how bright the acceptor is. Its seed
  is offset from the donor's, or the two dyes would move in lockstep.
  New catalog entries (translated, as the completeness test demands):
  `fret.acceptor_dynamics`, `fret.dyes.1.slow_radius`,
  `fret.dyes.1.amino_acid_interactions`. Reachable from the web FRET panel and,
  via the generated view spec, from the Qt form. 18 new tests in
  `test_fret_physics.py`.
* **A cold numba cache failed the suite, and a warm one hid it.** After editing
  `quest/core/av.py`, one full run failed in `density2points_ll` with
  `RuntimeError: Cannot set NUMBA_NUM_THREADS to a different value once the
  threads have been launched (currently have 7, trying to set 8)`; the next
  three passed. It reproduces **exactly** on a cold cache
  (`find quest -name '__pycache__' -exec sh -c 'rm -f "$1"/*.nbi "$1"/*.nbc' _ {} \;`)
  and never on a warm one, because only a fresh compile makes numba re-read its
  configuration. Nothing in QuEst sets the variable — traced with an
  `os.environ` watcher and a `set_num_threads` wrapper, both silent — so the
  launched count came from elsewhere in the process. Fixed by pinning
  `NUMBA_NUM_THREADS` in `tests/conftest.py` beside `NUMBA_THREADING_LAYER`:
  an explicit value removes the recomputation. Three cold runs green in both
  environments. **A green suite proves nothing about numba here unless the
  cache was cold.**
* **FRET is reachable and *tested* on all three surfaces.** It had landed on the
  web UI earlier (enable toggle, acceptor site selectors, `R0 [Å]`, and a
  browser test that runs a full FRET simulation and asserts `0 < E ≤ 1`), and it
  survived the AutoForm rewrite — but every desktop test set
  `fret_enabled = False`, so the Qt path was working by inspection only.
  `test_gui_simulation.py::TestTheFormRunsFret` closes that: both decays, a
  partial efficiency on a chosen site pair, `E = 1 − QY_DA/QY_D` agreeing with
  the reported number, a larger R0 transferring more (a nanometre R0 would not),
  the toggle gating the channel, and save/load round-tripping the acceptor.
  E = 0.77 on 148l E15→E90 at R0 = 52 Å. Still open in
  [specs/objectives.md](/specs/objectives.md): the acceptor does not move
  (fast-acceptor limit) and κ² is not modelled.
* **`locale.getdefaultlocale()` is removed in Python 3.15**, and QuEst was
  calling it — the deprecation warning showed up in the GUI test output the day
  it was written. `quest.i18n.system_locale()` now reads `LC_ALL`, `LC_MESSAGES`,
  `LANG`, `LANGUAGE` directly, which is all that function did anyway;
  `locale.getlocale()` is *not* a substitute, since it reports the current
  locale, `(None, None)` until something calls `setlocale`. `LANG=C` means "no
  localisation", not "a language", so it resolves to English.
* **English, German and French (`quest/i18n.py`).** Because the catalog is the
  single description of every parameter, translating *it* translates the docs,
  the Qt form and the web UI at once — no gettext, no per-surface dictionary.
  `quest/settings/locales/{en,de,fr}.json` hold the translated labels and
  descriptions plus a `ui` section for everything that is not a parameter.
  `python -m quest.gui.generate_view_spec` now writes one view spec per language
  and `doc/generate_parameter_docs.py` one doc page per language;
  `/api/parameter-catalog?locale=` and a new `/api/i18n` (RPC: `quest.locales`,
  and `locale` on `quest.parameter_catalog`) serve the browser, which has a
  language selector in the header. Full picture:
  [subsystems/i18n.md](/subsystems/i18n.md).
* **The backend defaults to English, not to the server's locale.** A shared
  backend answering in whatever language its host machine is configured for
  would be a surprise no client asked for; `resolve_locale()` (env → OS →
  English) is for the *desktop*, and the RPC handlers deliberately do not call
  it. Pinned by a test that sets `QUEST_LOCALE=de` and asserts the RPC still
  answers in English.
* **German column headers did not fit the quencher table.** Every assertion was
  green; rendering the German form and looking at it showed "Löschrate kQ
  (1/ns)" clipped to "öschrate kQ (1/n". Table headers now use their own short
  `table.*` strings and keep the full wording as the tooltip. The reason
  CLAUDE.md says to look at the images.
* **Locale files are cached per process**, so editing one while uvicorn runs
  changes nothing until it restarts — the same trap as uvicorn not reloading,
  and it cost one confused round of browser checks. Recorded in
  [subsystems/i18n.md](/subsystems/i18n.md); `i18n.clear_cache()` exists for
  tests.
* **The `.ui` files are gone; the Qt form is generated (`LAY-06`, `LAY-10`,
  `LAY-11` all closed).** `quest/gui/quest.view.json` is generated from
  `quest/settings/parameter_catalog.json` by `quest/gui/generate_view_spec.py`
  and rendered by ChiSurf's `AutoForm`. Deleted: `dye_diffusion2.ui`,
  `pdb_widget.ui`, `ui/av_property.ui`, `ui/dye_diffusion.ui`,
  `ui/dye_diffusion3.ui`, `widgets.py`, `dye_diffusion.json`, and
  `ui/rescource_rc.py` (98 kB of generated base64 that nothing imported); the
  window icon survives as `quest/gui/icons/dye-diffusion.ico`.
  `dye_widget.py` went from 1,899 lines to ~130. The quencher table now edits
  `amino_acid_interactions` directly (that is LAY-10 — there is no legacy
  `{residue: {atoms, kQ}}` shape left to disagree), and the three plots are
  `plot` sections rendered through `chiplot` (LAY-11 — QuEst owns no plotting
  code at all). Details in [subsystems/gui.md](/subsystems/gui.md).
* **The form's state *is* a project (`quest/gui/form_model.py`).** The old
  widget kept state in ~40 Qt controls and converted to a project dict and back
  by hand — and the conversion is exactly where the GUI drifted from the CLI
  before (it once shipped different quenching chemistry). `ProjectFormModel`
  presents a project dict as flat attributes, so what the desktop saves is
  byte-for-byte what `quest simulate` runs and the web UI posts. It imports no
  Qt: `tests/test_form_model.py` runs in the base environment, which has no Qt
  binding, with a subprocess guardrail that importing it pulls none in.
* **One description per parameter, now including the desktop.** The catalog
  already generated `doc/parameters.md` and fed the web UI; it generates the Qt
  form too. `tests/test_view_spec.py` regenerates in memory and fails if the
  committed `quest.view.json` differs; `tests/test_form_model.py` fails if the
  catalog documents a parameter the form does not bind, or the reverse. A new
  input can no longer reach the docs and the web UI while missing from the
  desktop. Also normalised two catalog entries that said `"Angstrom"` where
  every other said `"Å"`, and regenerated `doc/parameters.md` (which was stale
  by 323 lines).
* **`save_outputs` was doing two jobs, and the GUI lost its plots to it.**
  Writing `jobs/<uuid>/` and building the thinned trajectory/AV clouds were
  behind the same flag. The Qt form passes `save_outputs=False` so an
  interactive run leaves nothing behind — and silently got no trajectory, AV or
  autocorrelation data to plot. `simulate_project` now takes a separate
  `render_data` (default `True`); `quest/scan.py` passes `render_data=False`,
  which is what actually wanted the fast path. Found by a GUI test, not by
  reading the code.
* **`tests/test_gui_simulation.py` was testing an unquenched site.** It labelled
  148l chain E residue 55, where nothing quenches: QY came out exactly 1.0 and
  the assertion `0 < QY <= 1` passed anyway. Moved to residue 15 (one of
  `QUENCHED_SITES` in `test_simulation_performance.py`) and tightened to
  `0 < QY < 1`, so the test now fails if the quenching path breaks.
* **The AutoForm seam exists in the tree (`LAY-06`).**
  `quest/gui/autoform_panel.py::build_method_panel(name)` renders any of the
  eleven methods declared in `quest/manifest.json` through ChiSurf's
  `AutoForm.from_rpc_method`, and the form it returns yields
  `model.params()` shaped exactly like the RPC params — so a panel and its
  handler cannot drift apart, and the parameter half of the form needs no
  QuEst-side UI code. Additive on purpose: `dye_diffusion2.ui` still loads,
  because the manifest cannot describe the structure picker, the quencher
  table, the plots or the 3D view. Those need a `quest.view.json` scheme
  (AutoForm has `path_list` and `state_table` for the first two); the `.ui`
  files go when it covers them. Four tests added
  (`tests/test_chisurf_autoform.py::TestQuestPanelHelper`), including one that
  asserts the helper lists *exactly* what the manifest declares, so adding a
  method without a renderable schema fails.
* **`tests/test_pdb_id_support.py` imported FastAPI unguarded**, so it aborted
  collection of the whole suite in `arm64` — which has IMP and Qt but not the
  web extra. Now an `importorskip`, matching every other optional-dependency
  test. This was masked until the full `tests/` tree was run in `arm64`.
* **The Qt GUI uses chimol, and now finds it by itself.** QuEst's own PyMOL
  viewer was deleted earlier today; what remained was that chimol was only
  discovered when ChiSurf happened to be on `sys.path`, so the default silently
  fell back to the placeholder. `quest/gui/dye_widget.py` now calls
  `enable_chisurf()` at import — the same bridge everything else uses — and the
  viewer preference defaults to `chimol` rather than `pymol`, which named a
  widget that no longer exists.

  Verified from `/tmp`, with no `PYTHONPATH` set: `_HAS_CHIMOL` is true and the
  live widget reports `_ChimolAdapter`. 208 tests pass in `arm64`.


* **QuEst now resolves to this checkout from anywhere — and that immediately
  found a break.** The base environment's editable install mapped `quest` to
  `chisurf/modules/quest/quest`, a separate older checkout, via a meta-path
  finder that outranks the current directory; `arm64` had no QuEst installed and
  worked only because the shell sat in the repo. Both are fixed: the editable
  install re-points here, and `arm64` gets a `quest-src.pth`. Verified from
  `/tmp` in both.

  `chisurf/modules/quest` is a **git submodule with its own `.git`**, so it was
  *not* replaced by a symlink — it may carry unpushed work, and deleting a
  checkout to save a path lookup is the wrong trade. It is simply no longer what
  gets imported.

  With the live tree actually in play, ChiSurf's `quenching_estimator` plugin
  failed to import: it still referenced `quest.lib.tools.dye_diffusion`, removed
  by today's relocation. Pointed at `quest.gui`; **46 ChiSurf plugin tests pass**
  (quenching_estimator + fps_json_editor). The breakage had existed all day and
  was masked by the stale copy — which is the argument for linking the two
  repositories rather than vendoring a snapshot.


* **Three follow-ups recorded rather than started** (end of a long session; a
  half-done UI migration is worse than none):
  delete the `.ui` files in favour of AutoForm over the manifest, render the Qt
  plots through ChiSurf's `chiplot` instead of hand-built pyqtgraph (`LAY-11`),
  and verify that ChiSurf's own test runs use the live `src` checkout rather
  than its stale installed copy. All three are written up with what is already
  known in [plugin-integration.md](/specs/plugin-integration.md).


* **The 3D viewer now says what its colours mean.** Magenta spheres marked the
  quenching centres — the same atoms the simulation uses — but nothing on screen
  said so, which made the most informative thing in the view read as decoration.
  A three-line legend (quenching centre / labeling site / accessible volume)
  sits bottom-left of the stage, `pointer-events: none` so it never intercepts a
  drag. Verified in the browser; 22 browser tests still pass.

  Deliberately **not** changed: the default labeling site of `1 MET`. It looked
  wrong next to a "pick the largest chain" comment, but it is a legitimate site
  with a CB, and every alternative available client-side (middle of the chain,
  first non-terminal residue) is arbitrary. A real improvement would rank sites
  by solvent exposure, which is a simulation-side feature, not a default.


* **Examples finished, and tested by running them.** `02_fret_pair.py` (donor
  and acceptor, efficiency, E(t), and a check that a smaller R0 transfers less)
  and `03_labeling_scan.py` (four candidate sites ranked by yield, lifetime,
  contact and dominant quencher) join `01_donor_quenching.py`. The scan output
  is the argument for the whole tool in four lines:

  ```
      site      QY  tau [ns]   contact  quenched by
      E:15   0.997      4.19      0.2%  TYR 100%
      E:55   0.988      4.17      0.6%  PRO  60%
     E:100   0.731      4.02     27.2%  TRP  96%
     E:132   0.967      4.14      2.5%  TYR  79%
  ```

  Writing 02 caught a real trap: E(t) computed from the first histogram bins
  came out as **-11**, because the leading bins carry the excitation spike where
  the ratio of two decays means nothing. The window now starts at 1 ns, and the
  reason is in the code.

  `tests/test_examples.py` runs each script as its docstring says to and
  requires it to print something and to document how it is run — carrying the
  `notebooks` marker, so it is deselected by default and runs under
  `pixi run test-notebooks` with the notebooks. 7 tests.


* **Recorded the release state** ([release-state.md](/references/release-state.md)):
  QuEst is unreleased, so breaking changes are free — rename, delete, reshape
  formats, no deprecation periods, no compatibility branches. The page is
  explicit about the three things this does *not* license: silent changes to the
  science, breaking ChiSurf without saying so, and skipping OKF. `CLAUDE.md`
  points at it.

* **Web backend tidied against the domain it now delegates to.** `main.py` went
  from 500 to 373 lines: `POST /api/dyes` had its own read-merge-write of the
  dye library *and* its own defaults, both of which `quest.settings.save_dye`
  already does — it now delegates and maps `ValueError`/`OSError` onto 400/500.
  With that gone, `DEFAULT_DYE_STICKINESS`, `_with_dye_stickiness_defaults`,
  `_with_dye_library_defaults` and `DYE_REPOSITORY_PATH` were dead and were
  deleted, as were the direct imports of `quest.core.dye_diffusion` internals
  (`DEFAULT_QUENCHER_ATOMS`, `REFERENCE_PET_QUENCHING`,
  `STANDARD_AMINO_ACID_RESIDUES`) that `quest.settings.quenching_defaults()`
  replaced, and the unused `PARAMETER_CATALOG_PATH`. The backend no longer
  reaches past the facade for anything but `REPO_ROOT`, which it needs for the
  job directories.

  16 backend tests and 22 browser tests pass against the restarted server.


* **Web UI: verified working, and one leak fixed.** The browser suite passes
  (22), and a screenshot confirms the real thing — upload, viewer, labeling
  controls, FRET panel. Looking at it showed a defect no assertion covered: the
  structure was displayed as
  `3b7f5e05-43eb-49fb-8e0d-214d60de8bf7_148l.pdb` in **three** places — the
  topbar status, the STRUCTURE card and the viewer footer. Uploads are stored as
  `<uuid>_<name>` so two users cannot collide, and that bookkeeping was reaching
  the user.

  `POST /api/structure/metadata` now reports the uploaded name. All three
  display sites read `metadata.filename`, so one backend change fixed all of
  them, and a backend test pins it.


* **Fixed the browser-suite failure, and it was not the browser.** After the
  day's changes the 22 Playwright tests passed alone but failed at the end of a
  combined run with a bare `Page crashed`. Bisected: the pair
  `test_pdb_id_support.py` + `test_webui_e2e.py` reproduces it in 13 s, and
  `CHISURF_ROOT=/nonexistent` makes the same pair pass — so the first suspect was
  the ChiSurf import that `conftest.py` had started doing, which drags in
  tttrlib, mdtraj, pandas and pyarrow.

  That was half of it. ChiSurf is now imported **only where a Qt binding exists**,
  which is the only place it is used (the AutoForm tests) and the only place the
  SWIG import-order fix matters — the web environment gains nothing from it and
  pays a large footprint. But the full suite still crashed Chromium, because the
  cause is cumulative: IMP, numba, FastAPI and four executed notebooks in one
  interpreter leave too little for a renderer.

  So the separation the pixi tasks already implied is now **enforced**: the
  browser tests carry a `browser` marker, `pyproject.toml` sets
  `addopts = "-m 'not browser'"`, and `pixi run test-e2e` asks for them
  explicitly. A combined run now reports **201 passed, 22 deselected** rather
  than 22 red — and "deselected" is honest in a way that skipping would not be,
  because it names a suite that still has to be run.

  Verified: base combined **201 passed, 22 deselected**; browser in its own
  process **22 passed**; `arm64` **220 passed**.


* **The notebooks work again — and now they are tested by being run.** Every
  notebook in the folder imported `mfm`, a package that has not existed for
  years, and the test suite was green anyway because it only asserted that each
  file *had cells*. Documentation that claims to work, and does not, is worse
  than none.

  Rewritten on the current API, from the science the old ones contained:

  - `01_quenching.ipynb` — the donor decay, quantum yield, contact fraction and
    the per-residue attribution, contrasting a site next to a tryptophan with
    one that barely quenches.
  - `02_fret.ipynb` — donor and FRET decays, the efficiency, **E(t) = f_DA/f_D0**,
    and a check that shrinking R0 reduces transfer.
  - `03_et_distance_distribution.ipynb` — the E(t)-paper analysis: deconvolve
    E(t) into p(R_DA) with a transfer matrix and non-negative least squares plus
    a smoothness penalty (the old `mfm.fluorescence.calc_transfer_matrix` and
    `solve_nnls` are a few lines of NumPy/SciPy). It then **cross-checks the
    recovered distance against ⟨R_DA⟩_E computed from the accessible volumes** —
    two independent routes to the same number, which is the point of the
    exercise.
  - `04_diffusion_modulated_fret.ipynb` — the Haas-Steinberg PDE notebook,
    renumbered into the series with text explaining what it adds: it takes p(R)
    as given and integrates directly, where QuEst answers the same question by
    Monte Carlo.

  Eight stale files removed: the three `mfm` ones, two empty, one with invalid
  JSON, and two duplicates (`... (1).ipynb`, `... (2).ipynb`).

  Shared setup lives in `notebooks/quest_notebook.py` — checkout discovery, the
  example structure, and `quick_project` / `fret_project` builders — so the four
  notebooks do not each repeat it.

  `tests/test_notebooks.py` now **executes** each notebook through nbclient and
  additionally requires it to carry prose and a title. Nine tests, 23 s.


* **Packaging and CI migrated to pixi + rattler-build.** `pixi.toml` grew four
  Python environments beside the existing Node one — `default` (library, CLI,
  web backend), `desktop` (adds Qt), `frontend`, and a `build` environment
  holding nothing but rattler-build, so CI does not solve the science stack
  merely to package it. Tasks: `test`, `test-web`, `test-gui`, `test-e2e`,
  `serve`, `gui`, `build-pkg`. Verified with `pixi task list`; all twenty
  resolve.

  `rattler-recipe/recipe.yaml` replaces `conda-recipe/meta.yaml`. The old recipe
  had gone stale in a way that mattered: its entry point was
  `quest=quest.quest_gui:start_gui`, a module that no longer exists and which
  was in any case the *GUI* bound to the plain `quest` command. The new one
  declares `quest = quest.cli:cli` and `cs.quest = quest.gui:start_gui`, adds
  IMP (now the structure reader, not only the AV backend), requires numpy >= 2,
  and tests the built package by importing `quest.api`/`core`/`backend` and
  actually running `quest --help` and `quest template`.

  **The old CI had stopped running at all**: every trigger was filtered on
  `paths: modules/quest/**`, the path QuEst has when vendored inside ChiSurf,
  which matches nothing in this repository. `.github/workflows/ci.yml` replaces
  it with four jobs — library+backend across three OSes, the Qt GUI offscreen
  (with the X libraries a headless Qt still needs), the frontend typecheck and
  build, and the rattler package as an artifact. It also exports
  `NUMBA_THREADING_LAYER=omp`, the setting that stops the interpreter aborting
  when ChiSurf's tttrlib OpenMP runtime meets numba's.

  Neither the recipe nor the workflow can be executed here — no rattler-build in
  either environment, and no GitHub runner — so both are **unverified beyond
  parsing**: the manifest through `pixi task list`, the workflow through a YAML
  load. First CI run should be treated as the real test.


* **The RCSB fetch is now shared, in ChiSurf core.** Moved out of the
  `fps_json_editor` plugin into **`chisurf/core/fio/structure/fetch.py`** —
  library code, wanted by more than one plugin. The move also merged the two
  behaviours: ChiSurf's version only tried `.pdb`, QuEst's fell through to
  `.cif` for entries too large to have a PDB file, and the shared one does the
  latter. The plugin re-exports from core and keeps its own cache directory, so
  its 45 tests still pass.

  QuEst's `structure_fetch.download_pdb_id` delegates to it whenever a ChiSurf
  checkout is reachable, keeping its own implementation as the standalone
  fallback. The two tests that cover that fallback now pin themselves to it
  explicitly, rather than silently testing whichever path happened to be live.

  This follows the user's ruling that **plugins may depend on ChiSurf core and
  on core plugins** — which is what made the sharing legitimate rather than a
  coupling hack.

* **Correction to yesterday's claim: MRC writing is *not* duplicated.**
  ChiSurf's `save_av_mrc` voxelises a **point cloud** through `IMP.em`; QuEst's
  `write_mrc` writes an already-computed **density grid** (the AV, the ACV, the
  diffusion-coefficient and quenching-rate grids). Same file format, different
  inputs — not a drop-in, and not a duplicate. `DUP-07` is narrowed to fps.json
  payload handling, which does still overlap.

* Suites after the day's relocation: **216 passed, 1 skipped** in `arm64`,
  **222 passed** in base, **22 passed** for the browser suite. Note again that
  uvicorn does not reload — the browser suite reported 22 failures until the
  backend was restarted on the relocated package.


* **`quest/lib` is gone; the domain is `quest/core/`.** Ten modules relocated
  and renamed for what they are — `simulation`, `dye_diffusion`, `photon`, `av`,
  `structure`, `pdb`, `elements`, `imp_av`, `fps_json`, `io` — with
  `quest/core/__init__.py` keeping the service-layer surface (`simulate_project`
  and friends) so `from quest.core import …` still reads the same. The tree now
  says what the architecture is: `core/` the domain, `backend/` the RPC,
  `gui/` the form, `cli.py`, `api.py`, `manifest.json` — the layout ChiSurf's
  own plugins use (`burst_selection` is the fullest example: `api/ backend/
  cli/ gui/ server/ manifest.json`).

* **The PDB parser is IMP's now (`DUP-06` closed for reading).**
  `quest/core/pdb.py` went from ~400 hand-written lines of PDB/PQR/mmCIF
  parsing to a reader over `IMP.atom.read_pdb` / `read_mmcif`. IMP is present in
  **both** environments (2.24 in `arm64`, 2.17 in base), so this needed no new
  dependency.

  Two details decided the swap rather than assumed it. **Selection:** none of
  IMP's stock selectors matches what QuEst read — `AllPDBSelector` gives 1525
  atoms on 148L, `NonWater*` 1363, while QuEst's parser accepted **ATOM records
  only**, 1322. A five-line `PDBSelector` subclass reproduces exactly 1322, and
  it has to answer both APIs: IMP 2.17 hands `get_is_selected` a raw line, 2.24
  a `PDBRecord`. **Radii:** IMP assigns CHARMM-style values (N 1.85 Å) where
  QuEst uses Bondi-style (N 1.65 Å), and the radius is what the accessible
  volume tests against — so the parsing is IMP's and the chemistry stays
  QuEst's. Verified field by field against the old parser on 148L: chain,
  res_id, res_name, atom_name, element, radius and mass **identical**,
  coordinates within 4e-06 Å.

  Two problems surfaced and were fixed: IMP builds decorators through SWIG,
  which is **not thread-safe** (a scan's worker pool hit *"Wrong number or type
  of arguments for overloaded function 'new_Atom'"*), and parsing became the
  dominant cost because a scan re-reads the structure per site. One lock plus a
  small parse cache fixed both — and made the suite **faster than before the
  swap**: base 70 s → 20 s.

* **ChiSurf now runs in QuEst's test process, and the abort is understood.**
  Importing ChiSurf is harmless; *building its Qt forms and then running QuEst's
  numba-parallel kernels* aborted the interpreter. The cause is the threading
  layer: ChiSurf brings tttrlib with its own OpenMP runtime, and numba's
  **`workqueue`** layer in the same process aborts, while **`omp`** and `tbb`
  are stable. `tests/conftest.py` now sets `NUMBA_THREADING_LAYER=omp`, exports
  `CHISURF_ROOT` and `IMP_TRICKS_SRC`, and imports ChiSurf **before** anything
  touches IMP. The AutoForm tests are back in-process: 216 passed in `arm64`.

* **More duplication found, this time against ChiSurf (`DUP-07`).** Its
  `fps_json_editor` plugin carries `payload.py` (fps.json normalise/validate),
  `pdb.py` (RCSB download and cache) and `mrc.py` (`save_av_mrc`) — the same
  three things as `quest/core/fps_json.py`, `quest/structure_fetch.py` and
  `write_mrc`. It cannot be consumed where it is: a plugin is not a library, and
  QuEst's domain importing another plugin would couple them through the host's
  plugin tree and pull tttrlib/mdtraj/pandas into a headless run. They belong in
  `chisurf.core.*`; then QuEst deletes its copies.

* Suites: **216 passed, 1 skipped** in `arm64` (library + Qt GUI + IMP +
  AutoForm in-process) and **222 passed** in base; the browser suite passes
  **22** when the servers are up and it is not racing a backend restart.


* **More of `quest/lib` removed by relying on the host (7673 → 4460 lines).**
  With the presentation arrow reversed, three things stopped being QuEst's job:

  - **`lib/math`** — 38 lines of FFT autocorrelation. ChiSurf has
    `core.math.signal.autocorr`, but importing a host application to correlate
    an array is absurd, so the twelve lines that matter are now inlined in
    `quest/core.py` beside their only caller and the module is gone. Removing a
    module, not adding a dependency.
  - **`gui/plots/MolView.py`** (415 lines of embedded PyMOL) — ChiSurf's chimol
    renders the same thing. Deleted; the viewer now comes from the host, and
    `_NoMolView` says so plainly when there is no host instead of letting the
    calls fail one at a time.
  - **`gui/plots/lineplot.py`, `plotbase.py`, `genealogy.py`** and their `.ui`
    files — zero references. Deleted.

  **Deleting QuEst's viewer exposed two dormant bugs in the chimol adapter**,
  which had never run because the PyMOL fallback always won:
  `_ChimolAdapter.cmd` was a property returning `self.cmd` (instant
  `RecursionError`), shadowing the correct one defined earlier in the same
  class; and `_ChimolCmdShim` had no `orient()`, which the form calls on every
  3D update. Both fixed — the chimol path now works for the first time.

* **The suite runs in three configurations, all green**: `arm64` standalone
  (204 passed, no host — null viewer), `arm64` with the ChiSurf checkout on the
  path (204 passed — chimol viewer *and* AutoForm), and base with the web layer
  (222 passed).

* **ChiSurf must not be imported into QuEst's test process.** The first version
  of `tests/test_chisurf_autoform.py` imported it at collection, which pulls
  tttrlib, mdtraj, pandas and pyarrow into the interpreter; the run then
  **aborted** (`Fatal Python error: Aborted`) once the numba-parallel simulation
  kernels ran alongside IMP and Qt. Plain import order is not the problem —
  `IMP` then `chisurf` and the reverse both import cleanly — so the renderer now
  runs in a **subprocess** and reports back as JSON. That is also the more
  honest test: it checks that a host can render QuEst, not that the two
  libraries can share an interpreter.

* **Still redundant, and blocked on one decision: `lib/io/PDB.py` +
  `lib/common.py` (~670 lines).** Both ChiSurf
  (`core.fio.structure.coordinates`) and IMP (`IMP.atom.read_pdb`) can read a
  structure, so QuEst's parser is a third copy. Neither is a free swap:
  ChiSurf is absent from `arm64` unless `CHISURF_ROOT` is set and drags
  tttrlib/mdtraj/pandas into every headless simulation for the sake of a PDB
  parse; IMP is absent from the base environment, which is where the web layer
  runs. **Recommendation: IMP.atom**, since QuEst already depends on IMP for
  accessible volumes and it costs nothing extra in the environment that does the
  science — but it means the web/base environment needs IMP, which is a real
  installation decision rather than a refactor.


* **The dependency arrow is reversed for presentation, and the UI dedup is
  proven.** QuEst may now import ChiSurf — optionally — so the host's
  **AutoForm** renders QuEst's forms instead of QuEst carrying `.ui` files and
  hand-built Qt widgets. The fluorescence arrow is unchanged: QuEst → imp-tricks,
  never the reverse.

  The lever is that `AutoForm.from_rpc_method()` consumes an `rpc_methods` entry
  **from `quest/manifest.json`** — the very declaration a host reads to learn
  what QuEst offers. QuEst already ships all eleven methods with full JSON-Schema
  parameters, so the same artefact serves as the UI description: one declaration,
  not two. Verified in `arm64` with the ChiSurf checkout on the path — every one
  of the eleven renders; `quest.simulate` comes out as project / pdb_path /
  nbins / tac_range / project_dir plus a `save_outputs` checkbox, with **no
  QuEst-side UI code**. Screenshotted and read back through
  `form.model.params()`.

  `quest/chisurf_host.py` reaches the checkout the same way `quest.imp_tricks`
  does (source tree, `CHISURF_ROOT`, never fatal), and deliberately **never
  overrides a ChiSurf already imported** — a host that loaded QuEst as a plugin
  is already running its own copy. Nothing outside `quest.gui` imports it, so
  QuEst still runs standalone; the in-tree form remains the fallback.

  Environment note worth keeping: no environment has both out of the box. `arm64`
  has Qt and IMP but no ChiSurf; the `chisurf` env's installed ChiSurf predates
  AutoForm; base has ChiSurf but no Qt. `PYTHONPATH=~/dev/chisurf` (or
  `CHISURF_ROOT`) closes it in `arm64`, which is now the documented way to run
  the AutoForm tests. `tests/test_chisurf_autoform.py` skips cleanly without it.

  Staged from here, not big-bang: the manifest covers parameter entry; the
  structure picker, quencher table, plots and 3D view still need a
  `quest.view.json` scheme before the `.ui` files and the hand-built form can be
  deleted (LAY-06, now in progress).


* **`quest/lib` slimmed from 7673 to 4511 lines, and the legacy `lib.*` import
  spelling is gone (LAY-08).** The library was a self-contained toolkit
  shadowing both imp-tricks and ChiSurf; a minimal one is legitimate for a
  standalone app, but this was neither minimal nor compatible with either.

  **The Qt parts left**: `lib/plots/`, `lib/widgets.py`, `lib/ui/` and
  `lib/genealogy.py` (~2400 lines) moved to `quest/gui/`, where their only
  callers already were. The orphaned `avWidget.ui` went with the dead
  `AvWidget`.

  **668 unreachable lines were deleted**, found by a reachability analysis from
  the real entry points rather than by eye: `AvPotential` (an AV restraint —
  `IMP.bff.restraints` territory), the unused `Pdb` wrapper, three curve
  utilities in `datatools`, nine structure-analysis functions (`Universe`,
  RMSD, superposition, clustering, internal coordinates — IMP/ChiSurf work),
  and the two legacy boolean-collision photon kernels superseded by
  `simulate_photon_trace_rate`.

  **The `lib.*` spelling is retired.** Every internal import is now
  `quest.lib.…` and `quest/__init__.py` no longer injects the package directory
  onto `sys.path`. This was not cosmetic: with both spellings live,
  `lib.structure.Structure` and `quest.lib.structure.Structure` were **two
  different classes**, so an `isinstance` check across the boundary failed and a
  `Structure` was wrapped twice — a real `TypeError` that surfaced the moment
  the GUI moved to `quest.gui`.

  The full module-by-module verdict — keep, move, or blocked on imp-tricks — is
  in [imp-tricks-dedup](/specs/imp-tricks-dedup.md).

* Suites after the slimming: **202 passed, 1 skipped** in `arm64` (library +
  Qt GUI + IMP) and **222 passed, 3 skipped** in base (library + web backend +
  browser).


* **LAY-02 closed, then made structural: the GUI and the domain are separate
  files, and importing QuEst no longer imports Qt.** Two steps.

  First the class split. `TransientDecayGenerator` stopped inheriting
  `DonorDecay`; it now **builds** one per run from the form
  (`build_model()`, driven by an explicit `MODEL_ARGUMENT_SOURCES` table mapping
  the model's 28 constructor arguments to the 17 Qt-backed properties and 11
  plain attributes that supply them) and reads results back through delegating
  properties. Rebuilding per run is deliberate: the controls are the authority,
  so a stale model can never be what gets simulated.

  Then the file split, which the class split made possible: 1600 lines of widget
  moved to **`quest/gui/dye_widget.py`**, `quest_gui.py` became
  `quest/gui/app.py`, the `.ui` and settings resources moved with them, and
  `quest/lib/tools/` stopped re-exporting the widget — which was what dragged Qt
  into every import of the library. The dead `AvWidget` in `quest/lib/fps` went
  too; it was unused and the only reason that module imported a GUI toolkit.

  The last Qt in the domain was a **`QProgressDialog` driven from inside
  `DonorDecay.update_all`** — the simulation rendering its own UI. It now takes
  a `progress(fraction, message)` callback; the widget passes one that drives
  the dialog. A subprocess guardrail asserts that importing `quest.api`,
  `quest.backend`, `quest.core` and `quest.lib.tools` loads **no** Qt binding,
  so "the service layer is GUI-free" is now checked rather than intended.

  Verified before and after by 11 characterisation tests written first
  (`tests/test_gui_simulation.py`), plus a rendered screenshot of a real run:
  decay drawn, QY 0.91, 16.2 % contact.

* **Two GUI defects the characterisation tests exposed.**
  `PDBSelector.chain_id`, `.residue_id` and `.atom_name` had setters that were
  literally `pass` — so nothing could select a labeling site programmatically,
  and **loading a saved project silently ignored its donor and acceptor sites**.
  They now select by index (which is also what repopulates the dependent
  combos); picking a value the structure does not contain is refused rather than
  accepted as free text. Separately, `quantum_yield` raised `ZeroDivisionError`
  when the photon count was zero — reachable from the GUI, since its spin box
  counts photons in *millions* and rounds 4 000 to 0.00.

* **Examples started.** `examples/01_donor_quenching.py` runs in ~3 s and shows
  the whole story at one site: decay, quantum yield, mean lifetime, contact
  fraction, and which residue types did the quenching. It labels T4 lysozyme
  residue 100, next to a tryptophan, where the answer is unambiguous —
  QY 0.73, 27 % contact, **96 % of the quenching from TRP**. `examples/_common.py`
  puts the repo root first on `sys.path`, because this machine has `quest`
  installed editable from *another checkout* (`chisurf/modules/quest`) and an
  example run from `examples/` would otherwise silently exercise that one.


* **LAY-09 closed, and with it LAY-03: the RPC surface is complete at eleven
  methods.** Two capabilities existed only inside the web backend and were
  therefore reachable only from a browser. Both are now domain modules:

  - `quest/structure_info.py::structure_metadata` — the ~150-line PDB/mmCIF
    chain–residue–atom extraction that populates labeling-site pickers, moved
    verbatim (extracted programmatically rather than retyped) and given a real
    error for a missing file;
  - `quest/jobs.py` — listing, summarising, reading and archiving the
    `jobs/<uuid>/` directories, including the paging clamp and the rule that a
    run *is* a UUID directory (anything else in `jobs/` is ignored rather than
    half-parsed).

  `quest.structure.metadata`, `quest.jobs.list` and `quest.jobs.get` are
  registered and declared in the manifest, so **the eleven methods in the
  registry, in `METHODS`, and in `manifest.json` now agree** — which the
  manifest tests enforce. The FastAPI routes became adapters; the ZIP download
  route lost its hand-rolled archiver to `jobs.archive_job`, which also fixed a
  latent oddity — the archive no longer risks containing a previous copy of
  itself.

  14 new domain tests replace what could previously only be exercised through
  HTTP. One backend test pointed at `backend_main.MAX_JOB_LIST_LIMIT`, which now
  lives in `quest.jobs`; updated rather than deleted.

* **imp-tricks now works from source in both environments, with no install.**
  ChiSurf's mechanism turned out to be two things: imp-tricks ships a
  `sitecustomize.py` that extends `IMP.__path__` when `src` is on `PYTHONPATH`,
  and its release installer does `pip install <checkout> --no-deps`. The first
  is enough for subpackages IMP does not ship (`IMP.cgmol`, `IMP.finite`, …) but
  **not** for `IMP.bff`: a compiled regular package beats a namespace directory
  of the same name whatever the path order, so `import IMP.bff.av` still fails.
  A real install *was* tried in `arm64`, confirmed working, and then **reverted**
  on the user's instruction — imp-tricks is changing, so the source tree is what
  has to work.

  `quest.lib.imp_av` therefore calls `enable_imp_tricks()` **on import**, which
  extends both `IMP.__path__` and the compiled `IMP.bff.__path__`. Importing
  QuEst is now enough to reach `IMP.bff.av`, `IMP.bff.distance_metrics` and
  `IMP.cgmol.*` from the live checkout — verified in `arm64` (compiled
  `IMP.bff.AV` still present alongside) and in base (no compiled bff at all).

* **But imp-tricks' own AV cannot be used yet, and now we know exactly why.**
  Calling `IMP.bff.av.compute_av` on 148L in `arm64` fails on **both** backends,
  each from version skew inside imp-tricks:

  - `labellib` — its guard requires `LabelLib.AV`; the installed LabelLib
    exposes only `dyeDensityAV1` (which is what QuEst's own working path calls),
    so the backend declares itself unavailable;
  - `imp_bff` — `'Particle' object has no attribute 'get_particle_index'`;
    IMP 2.24 spells it `get_index()`.

  So `DUP-01` — delete QuEst's duplicate and delegate to `compute_av` — is
  **blocked on imp-tricks**, not on QuEst, and the fix belongs in that
  repository. Recorded with the evidence in
  [imp-tricks-dedup](/specs/imp-tricks-dedup.md) and
  [known-issues](/references/known-issues.md). The symmetry is worth stating:
  *both* codebases carry an IMP AV integration written against an API that has
  moved, and in both cases it went unnoticed because the environment that would
  exercise it was not the one being tested.

  Two new tests pin the distinction that caused the original mess —
  `HAS_IMP_BFF` (compiled decorator) is **not** the same question as "is
  imp-tricks' `compute_av` importable", and `imp_tricks_av_available()` now
  answers the second one explicitly.

* **Ran the suite in the canonical `arm64` environment for the first time: 34
  failures.** The user identified `/Users/tpeulen/mambaforge/envs/arm64/bin/python`
  as the environment to use — Python 3.12, numpy 2.4.6, PyQt5, LabelLib **and a
  compiled `IMP.bff`** (IMP 2.24). That last one matters: `HAS_IMP_BFF` is true
  there, so QuEst selected its IMP accessible-volume backend, a path that had
  never executed anywhere. It does not work. Three faults:

  1. `path_map.get_path_map_header()` raises `AttributeError` — IMP 2.24 binds
     it to a C symbol absent from the compiled `_IMP_bff`. **Fixed** by using the
     writable accessor, which is bound to a different symbol.
  2. `get_xyz_density()` returns a NumPy array that the code truth-tested.
     **Fixed.**
  3. With those out of the way the backend returns an **empty** volume: every
     tile's `PM_TILE_PATH_LENGTH` is the 50000 "unreachable" sentinel. Stopped
     there rather than reverse-engineering IMP's current contract.

  So **backend selection is now explicit**: `resolve_av_backend()` defaults to
  **LabelLib** — what QuEst has always actually run and what its tests pin — with
  IMP.bff opt-in through `QUEST_AV_BACKEND=imp_bff`. An unavailable or
  misspelled backend raises immediately, naming the usable ones, instead of
  producing an empty volume later; `AV.av_backend` records the choice. This is
  the strongest argument yet for
  [imp-tricks-dedup](/specs/imp-tricks-dedup.md): the duplicated integration was
  broken and invisible, because the environment that would have exercised it was
  never the one being tested. 34 failures → 1.

* **The last failure was a genuinely flaky test, and pre-existing.**
  `test_an_unseeded_project_still_varies` compared `quantum_yield_donor` across
  four unseeded runs — but at a site where the dye rarely reaches a quencher the
  yield saturates at exactly 1.0, so four independent runs legitimately report
  the same scalar. Confirmed by isolating it: the flake occurs with *and*
  without today's changes. It now compares the **decay curves**, which differ
  every run because the photon arrival times do — a sharper statement of "the
  Monte-Carlo run did not repeat". Stable over three consecutive runs in both
  environments.

  While chasing it, a real defect surfaced in `arm64`: unseeded runs there were
  **byte-identical** every time, because numba initialises each worker thread's
  RNG state deterministically per process — four "independent" runs returning
  the same answer reads as precision that is not there.
  `_seed_worker_threads()` now seeds one prange iteration per thread from
  `os.urandom` before each unseeded trace, which keeps the fast parallel kernel
  *and* restores independence. Seeded runs are untouched and still reproduce.

* **imp-tricks can be spliced into the live interpreter** —
  `quest/imp_tricks.py::enable_imp_tricks()`, at the user's request. The obvious
  approach does not work: imp-tricks ships `IMP/<name>/` as *namespace*
  directories, and in `arm64` the compiled `IMP` and `IMP.bff` are **regular**
  packages that shadow them completely, so `PYTHONPATH` alone leaves
  `import IMP.bff.av` failing. The helper appends to the already-imported
  package's `__path__` instead, so the checkout's `IMP.bff.av` and
  `IMP.bff.distance_metrics` resolve *alongside* the compiled `IMP.bff.AV`
  rather than replacing it. Verified in both environments (compiled `AV` still
  present in `arm64` afterwards), idempotent, and returns `[]` instead of
  raising when there is no checkout. Six tests.

* **Environment and testing concepts rewritten around the two-interpreter
  reality.** `arm64` is canonical and the only place Qt *and* IMP.bff exist, but
  it has **no fastapi/uvicorn/playwright**, so the web suites still run in the
  mambaforge base interpreter. Both must be green, and the commands for each are
  in [workflows/testing.md](/workflows/testing.md). Running only the base
  environment hides real breakage — that is precisely how the IMP backend stayed
  broken. `CLAUDE.md` now says to use `arm64`, and — at the user's instruction —
  that **OKF is updated in the same change as the code, always**, including
  correcting entries that turn out to be wrong.

* Suites: **186 passed, 1 skipped** in `arm64` (library + Qt GUI, IMP present)
  and **221 passed, 2 skipped** in base (library + web backend + browser, servers
  up and restarted after the route changes).

* **FRET now follows the diffusion (OBJ-01 items 1–4).** `calc_fret_rate`
  reduced both accessible volumes to **one scalar** rate and added it to every
  frame, so the FRET channel was completely independent of the trajectory QuEst
  exists to simulate — the donor could be anywhere and transfer would not
  change. It is replaced by `calc_fret_rate_trace`, a per-frame rate

      k_FRET(t) = (1/tau0) · ⟨ (R0 / |r_D(t) − r_A|)^6 ⟩_A

  computed by a new numba kernel `fps.fret_rate_trace` (parallel over frames,
  acceptor cloud subsampled by an even stride to bound the `n_frames ×
  n_acceptor` cost). The donor is resolved in time; the acceptor is averaged
  over its volume, which is the fast-acceptor limit and is now stated as such
  rather than implied. `calc_photons_fret` adds `k_quench + k_fret` frame by
  frame — parallel de-excitation channels — and feeds the existing photon
  kernel.

  On T4L E55 → E132 the efficiency moves **0.377 → 0.662** at the same R0: the
  rate average over ⟨(R0/r)^6⟩ is not the efficiency implied by ⟨R_DA⟩_E, and
  only the former uses the simulated dynamics.

  Two more OBJ-01 items closed with it. **One efficiency definition**: the
  analytic `_fret_efficiency` (from ⟨R_DA⟩_E, ignoring quenching entirely) took
  precedence over the photon-based one whenever it had been computed, and the
  two disagreed; it is deleted, `fret_efficiency` is now always
  `1 − QY_DA/QY_D`, and `rda_mean_e()` survives purely as a diagnostic distance
  nothing consumes. **No more silent degradation**: a missing acceptor raised
  nothing and produced a donor-only result inside a bare `except Exception`; it
  now raises with an error naming what to set.

  Also handled: both clouds hold dye *centres* and can overlap in space, where
  `(R0/r)^6` diverges. The distance is floored at the sum of the two dye radii,
  since the spheres cannot interpenetrate.

  `tests/test_fret_physics.py` (14 cases) pins the kernel against the Förster
  relation — rate at r = R0 is exactly 1/tau0, twice the distance is 64× slower,
  the *rate* is averaged over the acceptor cloud rather than the distance,
  coincident clouds stay finite — and pins the regression itself: the rate
  **varies along the trajectory** by more than 1.5×, which the old scalar could
  never do.

* **LAY-05 closed: QuEst declares itself.** `quest/manifest.json` gives the
  identity, entry points (`gui`, `cli`, `services`) and all eight RPC methods
  with params/result schemas and `long_running` flags — what a host reads to
  learn what QuEst offers **without importing it**. Five tests keep it honest:
  the declared method set must equal the registry exactly, summaries must match,
  every method must document its shapes, the long-running pair must be flagged,
  and the entry points must resolve. Version is checked against
  `pyproject.toml`. (Note: the ChiSurf-side `quenching_estimator` plugin folder
  still has no manifest of its own — that is a change in the other repository
  and was deliberately not made from here.)

* **Wrote the imp-tricks dependency direction to OKF, at the user's
  instruction**: [specs/imp-tricks-dedup.md](/specs/imp-tricks-dedup.md).
  **General fluorescence machinery belongs in imp-tricks under `IMP.bff`**;
  QuEst is the application on top — project schema, PET workflow, surfaces,
  scanning, artefacts. The arrow points one way: nothing in imp-tricks may
  depend on QuEst.

  Surveying `~/dev/imp-tricks/src/IMP/bff/` showed the overlap is already large:
  `av.compute_av` performs **the same IMP.bff → LabelLib → raise dispatch QuEst
  reimplements** (plus a `backend=` override QuEst lacks); `av.BasicAV`/`ACV`
  model AV and contact volumes; `av._kernels` has `density2points`,
  `average_distance`, `mean_fret_distance`, `split_av_acv`;
  `distance_metrics.av_pair_statistics` returns ⟨R_DA⟩, R_mp, R_E and ⟨E⟩ in one
  weighted pass where QuEst has three separately-resampling functions; and
  `cgdye.analysis.fret` already handles **κ²** and exact efficiencies from rate
  matrices. Six rows opened as `DUP-01`…`DUP-06`. The `fret_rate_trace` written
  today is itself general code in the wrong repository (`DUP-03`) — its natural
  home is beside `cgdye.analysis.fret`, which would bring κ² with it, the thing
  OBJ-01 item 6 still wants.

  **Corrected a known-issue while checking this.** The recorded reason the
  preferred AV backend never runs — "`IMP.bff` is missing" — was only half of
  it. `IMP.bff` is a *namespace package provided by imp-tricks*: with
  `PYTHONPATH=~/dev/imp-tricks/src` it imports and exposes `av.compute_av` and
  `distance_metrics`. QuEst still would not use it, because
  `HAS_IMP_BFF = hasattr(IMP.bff, "AV")` gates on a **compiled top-level
  attribute the package does not have**. So there are two compounding faults:
  imp-tricks is not installed, and the capability check asks the wrong question.

* Suites: **199 passed, 2 skipped** (library + backend + browser, servers up,
  backend restarted after the FRET change so the browser really exercised it)
  and **8 passed** for the Qt GUI.

## 2026-07-26

* **The Qt GUI was simulating different physics from every other surface.**
  Rendering the widget headlessly for the first time (the default interpreter
  has no Qt; the `arm64` and `chisurf` conda environments do, and both import
  quest straight from this tree) and **reading the screenshot** showed a
  quencher table listing *MET with atom CB* — which the domain model puts on SD.
  `dye_diffusion.json` turned out to carry a legacy chemistry block: PRO, MET,
  TRP, TYR all on **CB**, a single global `kQ = 2.0`, `critical_distance = 6.5`,
  and no HIS or CYS at all. The CLI, API and web UI meanwhile run the
  residue-resolved model (TRP 3.5 on the indole ring, TYR 2.0 on the phenol,
  MET 1.67 on SD, HIS 1.0, CYS 0.8, PRO 2.0). **The same project gave different
  answers depending on which surface you opened it in.**

  The settings file now describes the *session* only — paths, times, photon
  counts. `default_quencher_table()` derives the chemistry from the same
  `REFERENCE_PET_QUENCHING` and `DEFAULT_QUENCHER_ATOMS` everything else uses,
  and the widget fills `quencher` from it unless a user's own settings file says
  otherwise; `critical_distance` moved 6.5 → 8.5 to match the project template.
  Verified in the re-rendered GUI: TRP with the indole atoms at 3.5, quench
  radius 8.50. The residual divergence — the legacy `quencher` shape has no
  per-residue radius or stickiness, and the widget still runs
  `stickiness_mode="global"` — is now **LAY-10**.

* **FRET is reachable from the browser for the first time.** The web UI had *no
  FRET controls at all*: it could plot a FRET curve, but nothing could switch
  FRET on, choose an acceptor site, or set R0 (`projectHelpers.ts` shipped
  `R0_matrix: []`). The Qt GUI had acceptor geometry but no R0 either. Both now
  have it — an `R0 [Å]` spin box in the Qt FRET group, and a FRET acceptor panel
  in the web UI with an enable toggle, chain/residue/atom selectors, R0 and the
  acceptor diffusion coefficient.

  **A full FRET run now works end to end in the browser**: T4L donor E55,
  acceptor auto-seeded, donor QY 0.911 → FRET QY 0.416, **E = 0.553**, with both
  decays drawn. I screenshotted the chart and looked at it: the two curves are
  clearly separated, the FRET channel decaying much faster, which is what E≈0.55
  should look like.

  Looking at that screenshot also caught a bug the assertions had passed: the
  acceptor seeded on **`542 HOH` atom O** — a water molecule — because "last
  residue of the chain" is a water in most structures. `lastLabelableResidue`
  now walks back to the last residue with a CB (162 LYS in 148L), and an e2e
  case pins that a water is never chosen.

* **Testing is now defined as all three surfaces.** Recorded in
  [workflows/testing.md](/workflows/testing.md) and `CLAUDE.md`: library + web
  backend, the browser suite with both servers up, and the Qt widget under
  `QT_QPA_PLATFORM=offscreen` in the `arm64` environment — plus the rule to
  render the GUI and the charts and *read the images*, which is what found both
  defects above. New `tests/test_gui_widget.py` (8 cases: quencher atoms and
  rates match the reference chemistry, all PET-active residues present, contact
  radius 8.5, the R0 control edits the model and round-trips into a project,
  its range covers Ångström values, and the widget paints); it skips where Qt is
  absent. Two Qt-free guardrails in `test_service_layer.py` pin that the GUI
  settings file carries no chemistry and that the derived table matches the
  reference. Five new e2e cases cover the FRET panel and a full FRET run.

  Suites: **179 passed, 2 skipped** (library + backend + browser, servers up)
  and **8 passed** for the GUI. Also noted: uvicorn does not reload, so a
  backend change needs a restart before the browser suite means anything.

* **R0 is in Ångström, and the template no longer switches FRET off.** The unit
  question raised by yesterday's finding is settled by the user: Ångström, like
  every other length in a project. `template_project` wrote
  `R0_matrix: 5.5` — a nanometre value — into a field consumed as Ångström, so a
  template-derived FRET project ran at a transfer rate of order 1e-6 and
  reported no transfer, with no error anywhere. It now writes
  `DEFAULT_FORSTER_RADIUS = 52.0`, a named constant in `quest/project.py` whose
  docstring states the unit and the failure mode.

  Measured end to end on T4L (148L), donor E55 → acceptor E132: **E = 0.377**
  (donor QY 0.940, donor-with-acceptor QY 0.585). At the old value the same
  geometry gives E < 1e-4 — indistinguishable from having no acceptor.

  The unit is now written down where people and agents actually look: a new
  `fret.R0_matrix` entry in `quest/settings/parameter_catalog.json` (regenerated
  into `doc/parameters.md`), a units section in
  [architecture/project-schema.md](/architecture/project-schema.md), and
  comments at the two consuming sites — `calc_fret_rate`, where R0 meets the
  Ångström AV clouds, and `RDAMeanE`, which also gained a docstring saying it
  returns the *Förster-averaged* distance and is not interchangeable with
  `RDAMean` or `dRmp`.

  **The tests were the reason this survived**: `test_use_case_fret` and
  `test_simulate_project_with_fret` assert `0 <= E <= 1`, which is true of every
  number the model can emit. `tests/test_fret_units.py` (5 cases) now pins the
  unit instead of the range — R0 within the 20–80 Å band a real pair occupies,
  the matrix symmetric with an empty diagonal, no matrix without an acceptor,
  `build_donor_from_project` actually picking the value up, and the Förster
  relation giving E = 0.5 at r = R0 while a nanometre value gives < 1e-4 at 40 Å.

  Still open, tracked in [OBJ-01](/specs/objectives.md): the web UI has no R0
  field at all — `projectHelpers.ts` ships `R0_matrix: []`, so a project built in
  the browser silently uses the 52 Å code default and a user cannot enter the
  Förster radius of their actual dye pair. Suite: 171 passed, 1 skipped.

* **LAY-01 closed: project logic left the CLI, and a triplicated scan
  implementation collapsed into one.** `template_project`, `validate_project`
  and `ProjectValidationError` now live in `quest/project.py` and are exported
  from `quest.api`. The CLI keeps `_validate_project` as a four-line translator
  to `ClickException`, so its behaviour and tests are unchanged; the web backend
  no longer imports `quest.cli` at all; and `quest/lib/fps_json.py` — a
  *library* module that was importing the CLI — now calls the domain.

  Chasing that import surfaced the real damage: **three copies** of "simulate one
  labeling site and report it" (CLI `scan`, the web backend's `simulate_site`,
  and `run_fps_scan`), which had **already drifted**. The library copy omitted
  `dominant_quencher`, `dominant_quencher_fraction` and `quenchers`, so
  `api.run_fps_scan` silently reported less than `quest scan` did; and the CLI
  copy wrote fps.json AV overrides into the project without also applying them to
  `av_parameter`, so per-site linker geometry was partly ignored. All three now
  call `quest/scan.py::simulate_site`, which also fixes both defects. Concurrency
  stays with the callers, because how many sites run at once is a transport
  concern (the web backend keeps its GIL-aware `SCAN_WORKERS` pool).

* **LAY-03 in progress: QuEst has an RPC surface.** `quest/backend/` adds a
  `ServiceDispatcher`, an `InProcessClient` and `register_services(dispatcher)`
  registering **eight methods** — `quest.template`, `quest.validate`,
  `quest.simulate`, `quest.scan`, `quest.parameter_catalog`, `quest.dyes.list`,
  `quest.dyes.save`, `quest.quenching_defaults` — behind one envelope
  (`{"ok": True, "result"}` / `{"ok": False, "error"}`). Registration is
  duck-typed on `register(name, handler)`, so ChiSurf's own dispatcher works
  without QuEst importing the host. Re-registering a name raises rather than
  overwriting: a silent overwrite would make which implementation runs depend on
  import order.

  Two shared definitions came out of it. `quest/settings/__init__.py` now loads
  the parameter catalog and dye library **package-relative** — the web backend
  had been resolving them through a `parents[3]` repo-root guess with a
  cwd-relative fallback, which cannot work from an installed wheel or a frozen
  bundle. And `result_payload()` is now the single definition of a simulation
  result on the wire, used by both `quest.simulate` and `POST /api/simulate`.
  `quest.structure.metadata` and `quest.jobs.*` are **not** registered yet: their
  implementations are still inside the web backend (now tracked as LAY-09).

* **Three pre-existing backend test failures fixed.**
  `webui/backend/tests/test_backend_api.py::TestTopQuenchers` called
  `backend_main._top_quenchers`, which has not existed since `top_quenchers`
  moved into `quest.core` — the tests had never been updated and the whole class
  was erroring. They now call the real name and pass.

* **Suites: 166 passed, 1 skipped**, with the web UI running
  (`tests/` + `webui/backend/tests/`, `-p no:pytest-qt`). The 16 browser tests
  were re-run against a **restarted** backend afterwards — uvicorn does not
  reload, so the first green run had exercised the pre-refactor process — and
  passed 16/16 against the new code, including the scan workflow that now goes
  through the shared `simulate_site` and the simulate route that now returns
  `result_payload`. New: `tests/test_service_layer.py` (19 cases) covering the
  envelope, the registry, the method behaviours, the scan record shape, and
  three layering guardrails (no surface imports another surface's helpers; the
  library never imports the CLI; the service layer imports no Qt).

* **Objectives recorded** in [specs/objectives.md](/specs/objectives.md), at the
  user's request and deliberately **not implemented**:

  **OBJ-01 — make FRET first class.** It is degraded, not dead: two tests pass
  and the T4L project gives E = 0.96 at R0 = 52 Å. But `calc_fret_rate` computes
  **one scalar** `k_fret` from ⟨R_DA⟩_E over the *static* AV clouds and adds it
  to every frame, so the donor's diffusion — the entire point of QuEst — does not
  modulate FRET at all; the acceptor never moves; `fret_efficiency` has two
  definitions that can disagree (analytic 0.9598 vs photon-based 0.9578 in the
  same run, and the analytic one ignores quenching); a failure to build the
  acceptor degrades **silently** to a donor-only result inside a bare
  `except Exception`; and R0 has no parameter-catalog entry, so it is neither
  documented nor editable in the UI.

  While measuring that, found a concrete defect now in
  [known-issues](/references/known-issues.md): the shipped template writes
  `R0_matrix: 5.5`, which `calc_fret_rate` consumes as Ångström — at realistic
  D–A distances that is a transfer rate of ~1e-6, i.e. **no FRET**. 5.5 is a
  plausible Förster radius in *nanometres*, so it reads as a unit mismatch. The
  tests miss it because they only assert `0 <= E <= 1`.

  **OBJ-02 — fit dye parameters to measured decays.** Load an experimental decay
  and optimise `kQ`, contact radii, `slow_factor`, `D` and linker geometry until
  the simulation describes it — turning the shipped chemistry from "starting
  values" into calibrated ones. Two routes: rely on the host's fitting/sampling
  stack as a first-class ChiSurf plugin (preferred — it already has optimisers,
  error analysis, global analysis and IRF/background handling), or implement an
  optimiser in QuEst. The user's observation that **first-class plugin status
  serves this objective** is now written into the plugin-integration spec: the
  RPC/manifest split is what lets a host drive QuEst as a *model* instead of
  launching it as a *window*. The spec also lists what this implies for work done
  before it starts — cheap repeatable parameterised runs, controllable noise,
  name-addressable parameters, and a decay comparable to a measured curve.

* **The web UI layer is verified for the first time in this environment: 16/16
  browser tests pass.** Brought the stack up — `uvicorn
  webui.backend.app.main:app` on :8000 and `pixi run -e frontend dev` on :3000
  (Next.js 16.2.6, ready in 357 ms; the frontend proxies `/api/*` and `/health`
  to the backend via `next.config.ts` rewrites) — and ran
  `tests/test_webui_e2e.py`: **16 passed in 26 s**, no console errors, no
  horizontal overflow. That covers page rendering, the quenching editor
  (PET-active filter, kQ editing, atom editing and restore, clear/reset), the
  full upload → simulate → read-results workflow including that quench shares
  sum to 100 %, the scan workflow with dominant-quencher reporting and column
  filtering, and the run-picker's server-side cap. Combined with the 116 unit
  tests, **132 of the repository's tests now pass with nothing failing**.

  One trap worth recording: `--timeout=` is not available — pytest-timeout is not
  installed, and passing it makes pytest exit with an argument error that looks
  like a test failure.

* **Adopted the ChiSurf plugin split as QuEst's target architecture, and opened
  the gap backlog.** New [specs](/specs/index.md) group:
  [plugin-integration.md](/specs/plugin-integration.md) is the north star —
  `core/` (Qt-free domain) → `backend/services.py` (named RPC methods with one
  `{"ok", "result"|"error"}` shape) → facade → thin `cli/` and `gui/`, with a
  `manifest.json` declaring identity, entry points and method shapes, and a
  declarative view spec. The reference implementation studied was
  `chisurf/plugins/fcs/fcs_calculator/`.

  The method table falls out of what the FastAPI backend already exposes — nine
  routes map one-to-one onto `quest.simulate`, `quest.scan`, `quest.validate`,
  `quest.template`, `quest.parameter_catalog`, `quest.dyes.*`,
  `quest.quenching_defaults`, `quest.structure.metadata`, `quest.jobs.*` — which
  is the evidence that this surface is real and merely under-spelled. QuEst is
  unusually cheap to convert because its boundary contract is already plain
  serializable data ([the project JSON](/architecture/project-schema.md)) and its
  parameter catalog is already half a view spec.

  [assessment.md](/specs/assessment.md) itemises **8 open gaps**, verified
  against the tree rather than guessed: the web backend importing the *private*
  CLI helpers `_validate_project`/`_template_project` at four call sites
  (LAY-01); `TransientDecayGenerator` inheriting `DonorDecay`, so the Qt widget
  *is* the model (LAY-02); no RPC registry at all (LAY-03); `quest.api` being a
  re-export shim that callers reach past anyway (LAY-04); the ChiSurf
  `quenching_estimator` plugin shipping **no `manifest.json`** — only
  `__init__.py`, `README.md` and `test/`, so it loads through the host's legacy
  AST discovery, exactly the case ChiSurf's own plugin spec complains about
  (LAY-05); two hand-built forms describing the same parameters (LAY-06); three
  different long-running mechanisms, none cancelable from outside (LAY-07); and
  the `sys.path` injection that would collide inside a host application
  (LAY-08). Migration order is in the spec; nothing was implemented yet.

* **Created the QuEst OKF bundle — the repository now has a persistent memory
  layer.** `okf/` is an Open Knowledge Format bundle modelled on the sibling
  ChiSurf repository's: plain-markdown concepts with YAML frontmatter, grouped
  as [architecture](/architecture/index.md), [subsystems](/subsystems/index.md),
  [workflows](/workflows/index.md) and [references](/references/index.md) under
  a root [index](/index.md), plus this append-only log. A root `CLAUDE.md` points
  every session at it and states the loop in
  [workflows/change-tracking.md](/workflows/change-tracking.md): update the
  owning concept, append here, keep `doc/` in step, commit.

  The bundle was **populated from the code**, not from the README: the
  three-stage pipeline and its invariants, the project-JSON contract and the
  parameter catalog that documents it, the four surfaces over one service layer,
  the job-artefact layout, and one concept per subsystem (core service layer,
  accessible volume, dye diffusion, quenching model, photons/decay, structure
  I/O, CLI, web UI, Qt GUI).

  Things worth having written down that are not obvious from the source:
  the AV backend priority is **IMP.bff → LabelLib → RuntimeError** and is not
  configurable; quenching centres sit on the redox-active moiety while
  stickiness stays on the residue, which is why `calc_slow_av` takes two sets of
  centres; quenching rates **add** across overlapping contacts while stickiness
  factors **multiply**; the photon seed is deliberately offset from the walk's
  (`base + 7919 mod 2³¹−1`) so photons do not replay the trajectory they score;
  `update_all` avoids `gc.collect()` because the ~20 ms it cost is paid hundreds
  of times by a scan; `save_outputs=False` exists because job artefacts dominate
  scan runtime; and `quest/__init__.py` inserts the package directory onto
  `sys.path`, which is what makes the legacy `import lib.fps` spellings resolve.

* **Environment and suite state recorded as a baseline.** Suite:
  **116 passed, 17 skipped** in ~20 s, via
  `python -m pytest tests/ -q -p no:pytest-qt`. The flag is mandatory here — no
  Qt binding is installed and pytest-qt otherwise aborts collection before any
  test runs; `-p no:qt` does not match the plugin name. Of the skips, 16 are the
  Playwright web-UI tests, which skip unless the backend and frontend are
  already running: **a green run does not mean the browser layer was
  exercised.** Interpreter: mambaforge CPython 3.10.14 with numpy 1.26.4;
  LabelLib, numba, fastapi and playwright present; `IMP.bff` and PyQt5 absent.

* **Six known issues captured** in
  [references/known-issues.md](/references/known-issues.md) rather than left in
  a chat log: `IMP.bff` missing so the preferred AV backend silently never runs;
  `pyproject.toml` declaring `numpy>=2.0` while the installed IMP extensions are
  compiled against numpy 1.x and abort under numpy 2 (the failure is recorded in
  `tests/sim_test.log`); the pytest-qt collection abort; 806 stray `jobs/`
  run directories and 30 `.coverage.*` files accumulating in the working tree
  because `simulate_project` writes to `cwd` and `quest/utils.py` installs a
  Python-2-era coverage hook; and README links pointing at the vendored
  `modules/quest/notebooks/…` path that only exists inside the ChiSurf tree.

* **State of the tree at bundle creation.** The working tree carries a large
  amount of uncommitted work — 20 modified files (~4.2k insertions, notably
  `dye_diffusion.py`, `core.py`, `fps/__init__.py`, `cli.py`) plus untracked
  `webui/`, `quest/structure_fetch.py`, `quest/settings/*.json`, `pixi.toml` and
  twelve new test modules. None of it was committed as part of creating this
  bundle. Anything that touches git here must assume other work is in flight
  ([workflows/change-tracking.md](/workflows/change-tracking.md)).
