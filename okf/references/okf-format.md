---
type: Convention
title: OKF format and shared conventions
description: This bundle is OKF v0.2; the rules it follows are shared with FPSIMP, and this is where they live.
resource: https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md
tags: [okf, convention, fpsimp, migration]
status: stable
generated: { by: claude-code/opus-5, at: 2026-08-06T09:10:00Z }
stale_after: 2027-02-06
sources:
  - id: spec
    resource: https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md
    title: Open Knowledge Format v0.2 specification
    author: team:google-cloud-knowledge-catalog
    last_modified: 2026-07-01
  - id: fpsimp-conventions
    resource: ../../../fpsimp/okf/references/okf-conventions.md
    title: FPSIMP -- the shared convention, written out in full
    author: claude-code/opus-5
    last_modified: 2026-08-06
---

# This bundle is v0.2

Migrated from v0.1 on 2026-08-06. The full rule set -- taxonomy, frontmatter,
actor convention, link style, and how attested computations work -- lives once,
in FPSIMP's
[`okf/references/okf-conventions.md`](../../../fpsimp/okf/references/okf-conventions.md).
The two projects are heading for one server and deliberately follow the same
rules; duplicating them here would only let them drift.

# What the migration changed

| v0.1 | v0.2 |
|---|---|
| `timestamp: '2026-07-26T00:00:00Z'` | `generated: { by: <actor>, at: 2026-07-26T00:00:00Z }` |
| a `# Citations` list in the body | `sources:` in the frontmatter |
| — | `status:` on every concept |
| — | `stale_after:` on every concept |
| — | `sources:` derived from each concept's `resource`, with `last_modified` |
| `okf_version: "0.1"` | `okf_version: "0.2"` |

**`generated.by` is `process:okf-v0.1-import` on every migrated concept.** The
v0.1 bundle recorded no actor, and inventing one -- a person, a model -- would
corrupt the exact signal the field exists to carry. The import marker says only
what is known: this text came from the v0.1 bundle, producer unrecorded.
Re-attribute a concept when you next materially change it.

`stale_after` was set to six months from each concept's own timestamp, and three
months for the ones that track work in flight: `handover.md`,
`references/known-issues.md`, `references/release-state.md`,
`specs/programme-2026-07.md`, `specs/assessment.md`.

# Checking the bundle

The validator is vendored into this bundle (canonical copy lives in FPSIMP;
sync by copying, and `diff` them when either changes):

```bash
python3 okf/references/tools/okf_validate.py okf --version 0.2
python3 okf/references/tools/okf_validate.py okf --version 0.2 \
        --check-paths "quest/,webui/,tests/,doc/"
```

Errors are the spec's section 11 conformance rules only. `--check-paths` warns
about source paths a concept names that no longer exist -- which is how the
post-refactor drift in [known-issues.md](known-issues.md) was found. `log.md` is
exempt from that check, because history is supposed to mention paths that are
gone.

# What v0.2 buys that is worth using

- **`stale_after`** makes "is this still true?" a date comparison instead of a
  judgement. When it passes, the concept is not wrong -- it is *unreviewed*.
- **`sources` with `last_modified`** lets a reader see that a concept was
  written against a file that has changed since.
- **`verified`** records that a human or a named process actually checked a
  concept. Nothing in this bundle carries it yet, which is honest: the trust
  tier of every concept here is *unverified*.
- **`Attested Computation`** would let QuEst's own numbers -- quantum yield,
  mean lifetime, transfer efficiency -- be re-derived from a receipt and checked
  mechanically. `tests/baselines/` already does half of this informally. See
  FPSIMP's [computations](../../../fpsimp/okf/computations/index.md) for a
  worked pair.
