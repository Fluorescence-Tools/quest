---
type: Subsystem
title: RPC service layer
description: quest/backend/ — the named methods every surface shares, the result envelope, and how a host registers them.
resource: quest/backend/services.py
tags: [rpc, service-layer, plugin, boundary]
timestamp: '2026-07-26T00:00:00Z'
---

# One operation, one name

`quest/backend/` exposes QuEst's capabilities as named methods so the CLI, the
web backend, the GUI and a host application call the *same* operation with the
*same* result shape. It is the concrete first half of
[specs/plugin-integration.md](/specs/plugin-integration.md).

```python
from quest.backend import local_client

client = local_client()
project = client.result("quest.template", {"with_fret": False})["project"]
response = client.call("quest.validate", {"project": project})
```

# The envelope

Every handler returns exactly one of:

```python
{"ok": True,  "result": <plain serialisable data>}
{"ok": False, "error": "<message>"}
```

Callers tell success from failure by `ok`, never by sniffing for a field.
`InProcessClient.call` returns the envelope; `.result` unwraps it and raises
`RemoteError` on failure. A handler that raises anyway is converted by
`ServiceDispatcher.dispatch`, so a caller never has to handle both styles.

# The methods

| Method | Returns |
|---|---|
| `quest.template` | `{"project": …}` — a minimal valid project |
| `quest.validate` | `{"valid": True, "project": …}`, or a failure naming what is wrong |
| `quest.simulate` | The full result payload (see below) |
| `quest.scan` | `{"results": {name: record}}` — one record per labeling site |
| `quest.parameter_catalog` | The shipped catalog |
| `quest.dyes.list` / `quest.dyes.save` | The dye presets |
| `quest.quenching_defaults` | Residue-type PET chemistry |
| `quest.structure.metadata` | Chains, residues and atoms of a structure file |
| `quest.jobs.list` / `quest.jobs.get` | Past runs: summaries, and one run's result |

`METHODS` in `services.py` is the table with one-line summaries — the same table
a manifest declares. `tests/test_service_layer.py` asserts the registry matches
it.

All eleven are registered, and the manifest declares the same eleven — the
surface is complete as of 2026-07-27.

# `result_payload` is the boundary shape

`result_payload(result)` converts a `DecaySimulationResult` to plain data —
arrays to lists, absent channels to `None`. Both `quest.simulate` and
`POST /api/simulate` return it, so there is one definition of what a simulation
result *is* on the wire. Add a field there, not in a route.

# Registering with a host

`register_services(dispatcher)` needs nothing but `register(name, handler)`, so
a host application's dispatcher works unchanged and QuEst never imports the
host:

```python
from quest.backend import register_services
register_services(host_dispatcher)     # ChiSurf's ServiceDispatcher, say
```

Registering a name twice raises. That is deliberate: a silent overwrite would
make which implementation runs depend on import order.

# Rules this layer keeps

- Nothing here computes. A handler that grows logic means that logic belongs in
  the domain, where the CLI and notebooks can reach it too.
- No Qt, no FastAPI, no host imports — pinned by a guardrail test.
- Long-running work (`quest.simulate`, `quest.scan`) is currently **blocking**;
  a caller that needs progress or cancellation drives `quest.scan.scan_positions`
  itself. Making that uniform is `LAY-07`.
