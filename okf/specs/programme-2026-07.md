---
type: Specification
title: The 2026-07 programme
description: Every decision taken in the interview that scoped the current programme of work — what was chosen, what was rejected, and what each choice reverses.
resource: okf/handover.md
tags: [programme, decisions, roadmap, scope]
timestamp: '2026-07-28T12:00:00Z'
---

The work agreed on 2026-07-28, decided question by question before any code was
written. Each row is a decision with its alternative, because the alternative is
what a later session will otherwise re-propose.

**Read this with [assessment.md](assessment.md)** (the `LAY-*`/`DUP-*` tables)
and [objectives.md](objectives.md) (the science). This file says what was
*decided*; those say what the gaps *are*.

# Scope

Everything open in [handover.md](../handover.md) — items 1–7 — plus `LAY-04`,
`LAY-07` and the repository-hygiene entries in
[known-issues.md](../references/known-issues.md), across **three repositories**:
`quest`, `imp-tricks` and `chisurf`.

Out of scope, deliberately: implementing `OBJ-02` (the fitting loop itself), and
`DUP-07`'s remaining half (moving ChiSurf's fps.json `payload.py` — the owner
is not convinced `chisurf/core` is the right destination, so it stays open).

# Decisions

| # | Decision | Rejected alternative |
|---|---|---|
| 1 | **`OBJ-02` takes the host route.** QuEst contributes a *model*; ChiSurf's existing optimiser, error analysis and decay-data stack drive it. Recorded only — no fitting code in this programme | An own optimiser, which means reimplementing fitting, uncertainties and IRF handling |
| 2 | **One grid-offset convention: the integer `(ng-1)//2`, everywhere** | The float corner `(ng-1)/2`, which is geometrically prettier and is what LabelLib reports, but cannot be expressed by integer voxel indexing |
| 3 | **Residue identity comes from IMP's hierarchy, at the reader** — `(chain, res_id, insertion_code)` — and `Structure` keeps plain numpy arrays. `DUP-06` is promoted from "record only" to work | Wrapping a live IMP hierarchy in `Structure` (puts IMP objects in the domain layer, against the optional-dependency rule); or deleting `Structure` outright (collides with `LAY-04`) |
| 4 | **`arm64` is the only environment.** fastapi, uvicorn, playwright and pytest-asyncio move into it; base is retired | Keeping base for the web/browser layer, which is what "both must stay green" required |
| 5 | **The optional-dependency rule survives the environment it was enforced by.** Qt, LabelLib and IMP stay behind `try`/`except`; absence is *simulated* in tests instead of provided by an interpreter that genuinely lacks them | Making them hard requirements and deleting the LabelLib backend — which contradicts fixing that backend's registration in the same programme |
| 6 | **`DUP-03` is rejected, and κ² is documented rather than modelled.** Upstream's `calculate_fret_exact` is a Markov ensemble model over rotamer libraries; QuEst's trajectory kernel is the general one and should go *upstream* instead. κ² = 2/3 becomes an explicit project field | Adopting upstream (a different model, infeasible at cloud scale); or giving the dye an orientation now (real physics, a programme of its own) |
| 7 | **`quest.api` becomes a verb facade over a client**, and **serialization moves to the transport boundary** so an in-process call returns native objects | Keeping `result_payload` in the handlers, which taxes every local call — including a fitting loop, which decision 1 just made the primary consumer |
| 8 | **The contract mirrors ChiSurf's**: `quest/api/contract.py` with `PLUGIN_ID`, `CONTRACT_VERSION`, `service_success`/`service_error` carrying `error_code`, a `contract_descriptor()` and a `quest.contract.describe` method — guarded by a per-method round-trip test asserting the payload validates against the manifest schema | Trusting review to keep local and remote behaviour identical |
| 9 | **Long-running work gets one manager, mirroring ChiSurf's `JobManager` contract** — same statuses, `progress`, cooperative `should_cancel` — implemented in QuEst so it still runs standalone | Importing ChiSurf's manager (a plugin depending on its host); or a bare cancellation token, which leaves `LAY-07`'s three mechanisms standing |
| 10 | **"job" is freed for the async sense; run artefacts are renamed `runs`** | Keeping `quest.jobs.*` for artefacts and calling async work `quest.tasks.*` — no churn now, a permanent vocabulary mismatch with the host |
| 11 | **The ChiSurf plugin is rewritten from scratch** on the `modelling/fps_json_editor` template — manifest, `api/contract.py`, `rpc/services.py` registering QuEst's methods into the host dispatcher, lazy import, and a dockable widget instead of a `QMainWindow`. It reuses ChiSurf's own machinery rather than duplicating it. **`quest/gui/` stays in QuEst**; the plugin embeds it | Manifest-only (leaves the host unable to drive QuEst as a model); or moving `quest/gui/` into ChiSurf (QuEst loses its standalone GUI) |
| 12 | **imp-tricks is NOT published — everything stays local** (owner ruling, 2026-07-28, reversing the interview decision). The recipe and the target channel are recorded for whenever that changes; the version is decided **at release**, not bumped per change. Consequence, accepted: **the `test` and `gui` CI jobs cannot solve** and stay red | Publishing to anaconda.org (`tpeulen`, noarch) and PyPI, which is what the interview chose and what would turn those jobs green |
| 13 | **Verification is a committed numeric baseline.** `tests/baselines/*.json` over a fixed set; a number-moving phase diffs against it and regenerates it **in the same commit whose log entry accounts for every moved number**. Every fix must be *seen to fail* first. Cold numba cache before each baseline run; the IMP-backend half gates locally, CI covers LabelLib | Continuing with a green suite plus ad-hoc spot measurements — which is what let a transposed grid overlap the truth by 83 % and pass |
| 14 | **Run artefacts are opt-in and never land in `cwd`** | A cleanup script alone, which treats the symptom |
| 15 | **i18n gets a coverage test** across en/de/fr and the parameter catalog; UI chrome is translated; scientific help text falls back to English *visibly* | Machine-translating parameter help with no reviewer; or dropping de/fr |

# What these reverse

Three of the above contradict something this bundle previously asserted. They
are called out so a reader who trusts the older text is corrected:

- **Decision 4 reverses "both environments must stay green"**
  ([workflows/environment.md](../workflows/environment.md),
  [workflows/testing.md](../workflows/testing.md), and the root `CLAUDE.md`).
  The owner ruled base out of scope. Recorded honestly: **no defect in base was
  reproduced here** — its web backend tests gave 16 passed and `import
  quest.core` succeeded on 2026-07-28. This is an owner ruling, not a
  measurement.
- **Decision 3 reverses `DUP-06`'s "low severity, long-term, has not bitten
  yet"** in [assessment.md](assessment.md). It has bitten: every multi-chain
  structure gets one chain's side chains folded into another's CB.
- **Decision 6 reverses `DUP-03`'s premise** in [assessment.md](assessment.md)
  and [objectives.md](objectives.md), both of which say adopting
  `IMP.bff.cgdye.analysis.fret` "would bring κ² with it". It would not.

# Phases

Ordered by dependency, not by size. One commit each; OKF updated in the same
commit.

| # | Phase | Depends on |
|---|---|---|
| 0 | Hygiene and the write-side default | — |
| 1 | Consolidate on `arm64` | 0 |
| 2 | Numeric baseline; run the two unverified suites | 1 |
| 3 | One grid-offset convention | 2 |
| 4 | Residue identity from IMP | 2 |
| 5 | FRET respec and explicit κ² | 3, 4 |
| 6 | Remaining dedup rows | 3, 7 |
| 7 | imp-tricks upstream fixes | — |
| 8 | `LAY-04` facade and the JSON-RPC contract | 1 |
| 9 | `LAY-07` cancellation; `runs`/`jobs` rename | 8 |
| 10 | Rewrite the ChiSurf plugin | 8, 9 |
| 11 | i18n coverage | — |
| 12 | Publish imp-tricks; CI green | 7 |
| 13 | OKF closeout | all |

The baseline sits at phase 2 **before** anything moves numbers, on purpose: the
browser suite and the ChiSurf plugin had gone two sessions unverified across
changes that touched every accessible volume, and discovering a break after
three number-moving commits makes attribution impossible.

# Still undecided

- Where ChiSurf's fps.json `payload.py` belongs (`DUP-07` remainder).
- Whether the leftover scratch files in the tree (`out.xyz`, `out_slow.xyz`,
  `test.pdb`, `test.cif`, `webui-viewer-testpdb.png`, `tests/sim_test.log`) go.
