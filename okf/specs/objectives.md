---
type: Specification
title: Objectives
description: Where QuEst is going as a scientific tool — restoring FRET as a first-class observable, and fitting dye parameters to measured decays.
resource: okf/specs/plugin-integration.md
tags: [objectives, roadmap, fret, optimization, fitting]
status: stable
generated: { by: process:okf-v0.1-import, at: 2026-07-26T00:00:00Z }
stale_after: 2027-01-26
---

The scientific direction, as distinct from
[the architectural target](plugin-integration.md). **Nothing here is
implemented yet.** They are recorded so that every refactor and every new
feature is shaped to make them cheaper rather than harder — a change that
blocks one of these is the wrong change, even if it is locally tidy.

# OBJ-01 — FRET as a first-class simulated observable

QuEst simulated FRET historically, and the machinery is still present:
`DonorDecay.calc_acceptor_av`, `calc_fret_rate`, `calc_photons_fret`,
`get_histogram_fret`, and the `fret` block of the
[project schema](/architecture/project-schema.md). Two tests exercise it
(`test_use_cases.py::test_use_case_fret`, `test_api.py::test_simulate_project_with_fret`)
and it does produce a second decay curve. **It is not dead — it is degraded**,
and the degradations are quiet ones.

Measured on 2026-07-26 with the T4L test project at R0 = 52 Å: donor QY 0.9944,
FRET-channel QY 0.042, E = 0.96. So the path runs end to end.

What is wrong with it today:

1. ~~**FRET does not see the diffusion.**~~ **Done 2026-07-27.**
   `calc_fret_rate_trace` evaluates `k_FRET(t) = (1/tau0)·⟨(R0/|r_D(t) − r_A|)^6⟩_A`
   per frame; the donor is resolved in time and the acceptor averaged over its
   volume. E on T4L E55→E132 moved 0.377 → 0.662 as a result.
2. ~~**The acceptor never moves.**~~ **Done 2026-07-27.** The acceptor is a dye
   in its own right: it walks its own accessible volume with its own diffusion
   coefficient, its own slowing radius and its own per-residue interaction
   table, and `k_FRET(t)` uses the instantaneous separation of the two
   trajectories (`fret_rate_pair_trace`). The fast-acceptor limit is still
   reachable as `fret.acceptor_dynamics: "averaged"`. **The default changed**,
   so FRET numbers move: on 148l E15→E90 with a sticky acceptor
   (D = 2.5 Å²/ns, slow radius 11 Å) E went 0.73 → 0.78. The two differ because
   a cloud is weighted uniformly while a trajectory is weighted by where the
   dye dwells — the direction depends on geometry, not on a fixed bias.
3. ~~**`fret_efficiency` has two definitions.**~~ **Done 2026-07-27.** The
   analytic one is deleted; efficiency is always `1 − QY_DA/QY_D` from the two
   simulated decays. `rda_mean_e()` remains a diagnostic distance only.
4. ~~**Failure is silent.**~~ **Done 2026-07-27.** A missing acceptor raises with
   a message naming what to set, instead of yielding a donor-only result.
5. ~~**R0 is barely reachable.**~~ **Done 2026-07-26.** The unit is settled
   (Ångström), documented in the parameter catalog, and editable in both UIs:
   an `R0 [Å]` spin box in the Qt FRET group and an `R0 [Å]` field in the web
   UI's FRET acceptor panel, which also gained the enable toggle and acceptor
   site selectors it never had. A browser FRET run now reports E = 0.55 on T4L.
6. ~~**No orientation factor.**~~ **Closed 2026-07-28 as *chosen and
   documented*, which is not the same as modelled — read this before assuming
   κ² is simulated.** `fret.kappa2` is an explicit project field, validated to
   `[0, 4]` when the project is read, translated in all three languages, and
   defaulting to the isotropic dynamic average **2/3**. Because a published
   Förster radius is already quoted at 2/3, the default reproduces every number
   computed before the field existed; the rate carries `κ²/(2/3)`, the
   literature's `1.5·κ²`.

   **It cannot be modelled in the current dye model, and that is the real
   finding.** QuEst's dye is a structureless point diffusing in an accessible
   volume: it has no transition dipole, so there is nothing to orient. Adopting
   code does not help — `kappa2_from_vectors` upstream *takes* the two dipole
   vectors. Modelling κ² needs a **new degree of freedom**: a rotational
   diffusion of a dipole per dye, its own correlation time, its own schema
   entries, and its own calibration. That is a project the size of the
   acceptor-diffusion work, and it is what would move item 6 from *documented*
   to *modelled*.

"Working" means, at minimum: a per-frame D–A distance, one definition of
efficiency, an error when the acceptor cannot be built, R0 in the catalog and
the UI, and a test that pins a known geometry to a known efficiency rather than
asserting `0 <= E <= 1`.

**Reachability, confirmed 2026-07-27 after the AutoForm migration.** FRET is
live on all three surfaces and each is covered by a test that *runs* it, not
merely one that builds a project:

| surface | how it is reached | test |
|---|---|---|
| library / CLI | the `fret` block of the project | `test_fret_physics.py`, `test_fret_units.py` |
| web UI | enable toggle, acceptor chain/residue/atom selectors, `R0 [Å]` | `test_webui_e2e.py::TestFretPanel` (5) and `::TestFretSimulation` — a full browser run asserting `0 < E ≤ 1` |
| Qt form | the FRET panel of `quest.view.json`, bound to the same keys | `test_gui_simulation.py::TestTheFormRunsFret` (8) |

The desktop tests were the gap: the rebuilt form bound FRET through the shared
project keys, but every GUI test set `fret_enabled = False`, so nothing ran it.
They now pin both decays, an efficiency strictly between 0 and 1 on a chosen
site pair, `E = 1 − QY_DA/QY_D` agreeing with the reported value, a larger R0
transferring more (a nanometre R0 would not), the toggle actually gating the
channel, and a FRET project surviving save/load. Measured E = 0.77 on
148l E15→E90 at R0 = 52 Å.

**Progress.** *2026-07-26* — reachability: R0 in the catalog and both UIs, the
web UI can switch FRET on and choose an acceptor site, the unit mismatch that
zeroed transfer is fixed, and a full FRET run works from the browser.
*2026-07-27* — the physics of items 1, 3 and 4: a per-frame transfer rate, one
efficiency definition, and a loud failure when the acceptor is missing; then
item 2, a diffusing and sticking acceptor with per-dye properties.

**All six items are now closed** — item 6 as *documented*, not as *modelled*;
see above, and do not read the checked box as "QuEst simulates κ²".

The claim that `IMP.bff.cgdye.analysis.fret` "already handles κ², which is one
more reason the rate kernel should move there" was **wrong on both counts** and
is retracted: that module solves a Markov kinetic ensemble over rotamer
libraries and merely *accepts* a κ² matrix. See `DUP-03` in
[assessment.md](assessment.md); the direction of that migration is now
reversed.

# OBJ-02 — Fit dye parameters to measured decays

The direction that makes QuEst quantitative rather than illustrative:

> experiment on a system → optimise the model parameters until the simulation
> describes the measurement.

Concretely: load a measured fluorescence decay, vary the dye and quenching
parameters (`kQ` per residue type, contact radii, `slow_factor`, the diffusion
coefficient, linker geometry), and find the values that reproduce it. The
shipped chemistry is explicitly *starting values to be calibrated*
([references/pet-quenching-theory.md](/references/pet-quenching-theory.md)) —
this objective is what turns them into calibrated ones, and it is the same loop
that would calibrate an accessible contact volume against a donor lifetime.

**Decided 2026-07-28: the host route.** QuEst contributes a *model*; ChiSurf's
existing optimiser, error analysis, global analysis and decay-data handling
drive it. Nothing is implemented — the decision is the deliverable
([specs/programme-2026-07.md](programme-2026-07.md), decision 1) — but it is
binding on work done before it starts, and it has already shaped two things:
`quest.api` becomes a verb facade over the RPC client rather than a re-export
shim (`LAY-04`), and **serialization moves to the transport boundary** so an
in-process call returns native arrays. That second one is not tidiness: an
optimiser calls `simulate` thousands of times, and converting every decay
channel to a Python list per iteration is a tax paid forever by the caller this
objective exists to serve.

The two routes as they were weighed:

- **Rely on the host.** As a first-class ChiSurf plugin, QuEst reaches an
  existing fitting and sampling stack — optimisers, error analysis, global
  analysis over several datasets, and a decay-data model that already knows
  about IRFs, background and scatter. QuEst would contribute a *model* whose
  parameters the host varies. This is the reason
  [plugin-integration.md](plugin-integration.md) matters beyond tidiness: the
  RPC/manifest split is what lets the host drive QuEst as a model rather than
  launch it as a window.
- **Own optimiser.** Keeps QuEst standalone at the cost of reimplementing
  fitting, uncertainties and data handling.

**The host route was taken on 2026-07-28**, as recorded above.

## What this implies for work done before it starts

- **A simulation must be callable as a function of its parameters**, cheaply and
  repeatedly. `save_outputs=False` already exists for that reason; anything that
  makes a run write, log or allocate unconditionally works against it.
- **Runs must be reproducible.** Seeding is already threaded through the walk
  and the photon trace ([subsystems/dye-diffusion.md](/subsystems/dye-diffusion.md));
  an optimiser needs the noise to be controllable, or it will chase Monte-Carlo
  scatter.
- **Parameters need to be addressable by name.** The dotted paths of the project
  schema are already the right shape for "vary `amino_acid_interactions.TRP.kQ`".
- **The decay is the fit target**, so it must be comparable to a measured curve:
  the same time axis, and eventually convolution with an IRF plus background.
  That machinery exists in the host — another argument for the plugin route.
- **The service layer is how a fitter drives QuEst.** `quest.simulate` returning
  a plain payload ([subsystems/core-api.md](/subsystems/core-api.md)) is already
  most of the interface an optimiser needs.
