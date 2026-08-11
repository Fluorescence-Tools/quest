---
type: Subsystem
title: CLI
description: quest simulate/scan/template/validate/gui, dotted-path overrides and grid sweeps.
resource: quest/cli.py
tags: [cli, click, batch]
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

# Commands

```bash
quest template --with-fret > project.quest.json   # emit a valid starter
quest validate -p project.quest.json              # check required keys + structure
quest simulate -p project.quest.json -o decay.csv # one run
quest scan -p project.quest.json -l fps.json -o scan.json   # one run per labeling site
quest gui                                          # launch the Qt GUI
```

Entry point `quest = quest.cli:cli` (Click group). Output format follows the
`-o` extension: `.csv` or `.json`.

# Overrides and sweeps

- `--set a.b.c=value` — override any project field by dotted path. Values are
  parsed as JSON when possible, so `--set fret.enabled=true` and
  `--set attachment.residue=42` do the right thing.
- `--grid a.b.c=v1,v2,v3` — sweep. Multiple `--grid` options form the
  **Cartesian product**, and each run's output path is tagged with the values
  that produced it (`_resolve_output_path` / `_sanitize_tag_value`).

`_set_path`, `_clone_with_overrides`, `_expand_grid`, `_cartesian` and
`_build_runs` implement this; they operate on the plain project dict and are the
reason the [project schema](/architecture/project-schema.md) keeps dotted paths
addressable.

# `scan`

Takes a project **and** an `fps.json` labeling file, then simulates every
position in it, reporting per site: lifetime, quantum yield, contact fraction,
AV and contact volumes, contact-to-free ratio, and the dominant quencher with
its `quench_rate_fraction` (from `top_quenchers`). Progress is a `tqdm` bar.

Two design points:

- **Failures are data, not exits.** A site that raises is recorded with
  `status: "failed"` and the error string, with the metric fields present and
  null. A buried residue must not abort a 300-site scan.
- **`save_outputs=False`.** The scan reads only scalars and `aa_residence`; the
  job directory would dominate the runtime
  ([job-artefacts.md](/architecture/job-artefacts.md)).

`tests/test_scan_reporting.py` pins both the quencher attribution and the
failure diagnostics.

# Validation

`_validate_project` is shared with the web backend, so the browser and the
terminal agree on what a valid project is. It checks the required keys, resolves
the structure (path **or** PDB ID), requires a donor dye with `D`, and sanity
checks `t_max > t_start`.
