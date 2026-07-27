---
type: Reference
title: User documentation
description: Where the human-facing manual lives and which parts are generated.
resource: doc/index.md
tags: [documentation, generated]
timestamp: '2026-07-26T00:00:00Z'
---

`okf/` is the knowledge layer for agents. `doc/` and `README.md` are the manual
for people, and a change to user-visible behaviour updates both
([workflows/change-tracking.md](/workflows/change-tracking.md)).

| Path | Content | Generated? |
|---|---|---|
| `README.md` | Project overview, install, quickstart for every surface, citations | hand-written |
| `doc/index.md` | Documentation entry point | hand-written |
| `doc/cli.md` | CLI reference | hand-written |
| `doc/quenching_manual.md` | The quenching model explained for users | hand-written |
| `doc/parameters.md` | Full parameter reference | **generated** |
| `doc/generate_parameter_docs.py` | The generator | — |
| `doc/img/`, `doc/logo.png` | Figures | — |
| `notebooks/` | Library-usage examples, checked by `tests/test_notebooks.py` | — |

# The generated page

`doc/parameters.md` is produced from `quest/settings/parameter_catalog.json`:

```bash
python doc/generate_parameter_docs.py
```

Edit the catalog, then regenerate. Hand edits to `doc/parameters.md` are
overwritten and will also drift from the web UI, which reads the same catalog
([architecture/project-schema.md](/architecture/project-schema.md)).
