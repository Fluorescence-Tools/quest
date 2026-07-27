---
type: Reference
title: Release state
description: QuEst is unreleased — breaking changes are free, and backwards compatibility is not a constraint.
resource: pyproject.toml
tags: [process, versioning, compatibility]
timestamp: '2026-07-27T00:00:00Z'
---

# QuEst is not released

There are no users on a published build, so **there is nothing to stay
compatible with**. Prefer the clean change over the compatible one:

- Rename anything. Module paths, function names, project keys, RPC method names
  and manifest fields may all change; no deprecation period is owed.
- Delete anything unused. Shims, aliases and "kept for compatibility" wrappers
  are debt with no creditor. `quest/lib` was removed outright on 2026-07-27 for
  exactly this reason.
- Change file formats. The project JSON, `fps.json` handling and the job
  directory layout can be reshaped when the current shape is wrong — the
  `R0_matrix` unit fix landed as a straight correction rather than a migration.
- No back-compat branches in code. A second code path "for old projects" is a
  second thing to test and a second thing to get wrong.

What this does **not** license:

- **Silent behaviour changes in the science.** Renaming a function is free;
  quietly changing what a simulation computes is not. When the physics moves,
  measure it, say so in [the log](../log.md), and pin it with a test — as with
  the per-frame FRET rate (E 0.377 → 0.662) and the IMP reader (verified
  field-for-field *identical* before it was accepted).
- **Breaking a sibling repository without saying so.** ChiSurf imports QuEst.
  Changes that move an entry point are fine, but they must be recorded so the
  host can follow — the GUI entry point moved to `quest.gui:TransientDecayGenerator`
  on 2026-07-27 and the manifest was updated in the same change.
- **Leaving OKF behind.** The freedom to change fast only works if what changed
  is written down.

Revisit this page the moment a version is published; from then on the rules
invert.
