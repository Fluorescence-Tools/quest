---
type: Subsystem
title: Tasks
description: The one job mechanism every surface drives -- observable progress, cooperative cancellation, and a status vocabulary shaped to match the host.
resource: quest/tasks.py
tags: [jobs, cancellation, progress, threading, chisurf]
status: stable
generated: { by: claude-code/opus-5, at: 2026-08-06T09:10:00Z }
stale_after: 2027-02-06
sources:
  - id: quest-tasks-py
    resource: ../../quest/tasks.py
    title: quest/tasks.py
    author: human:tpeulen
    last_modified: 2026-07-28
  - id: quest-rpc-services
    resource: ../../quest/rpc/services.py
    title: quest/rpc/services.py -- the only caller today
    author: human:tpeulen
    last_modified: 2026-07-28
---

# Why it exists

`LAY-07` recorded three unrelated ways of running long work -- a thread pool in
the web backend, a blocking call in the CLI, a `QProgressDialog` in the GUI --
none of which could be observed or stopped from outside. `quest/tasks.py` is the
one they are meant to drive.

It **mirrors ChiSurf's job contract without importing it**: the status
vocabulary, `progress`, the cooperative `should_cancel` and the `to_dict()`
shape match `chisurf/server/jobs.py` deliberately, so the host recognises what
QuEst reports -- but a plugin that needs its host in order to run standalone is
not a plugin.

# The vocabulary

```
queued ──start──> running ──finish──> completed
   │                 │      └─fail──> failed
   │              cancel
   │                 ↓
   └──cancel──> cancelling ──worker notices──> cancelled
```

`CANCELLING` is a distinct state from `CANCELLED` on purpose. Setting the flag
*asks*; the worker notices at its next checkpoint, and until then it may still
be touching shared state. A cancel button that implies more than it delivers is
worse than none.

`JobStatus.is_terminal` is `completed | failed | cancelled` -- the states after
which nothing more may be written.

# Cancellation is cooperative, and one simulation is nearly atomic

A **scan** checks between sites, so it stops within one site. A single
`simulate` has essentially one checkpoint: `_simulate_traj` computes the whole
trajectory in a single `njit` call, and a jitted kernel cannot poll a Python
flag. Cancelling therefore means *"do not start the next stage"*, not *"stop
now"*. That limitation is stated rather than hidden.

Two consequences worth knowing:

- A job cancelled while still `QUEUED` goes straight to `CANCELLED` -- there is
  no worker to notice anything.
- A worker that returns *after* observing cancellation is recorded as
  `CANCELLED`, not `COMPLETED`, and its partial result is dropped.

# The domain never learns about jobs

`JobManager.canceller(job_id)` returns a zero-argument predicate and
`reporter(job_id)` returns a `(percent, message)` callback. The domain takes
those **callables**, not a `Job` or the manager, which is what keeps
[`quest/core/`](core-api.md) free of job bookkeeping and lets a caller with its
own cancellation source -- a host, a Qt dialog -- pass its own.

This is the same rule as [surfaces.md](../architecture/surfaces.md), one level
down: bookkeeping lives at the edge, the domain stays plain.

# `to_dict()` carries no result

Deliberately JSON-only. `result` may be a `DecaySimulationResult` full of numpy,
and the dict crosses a transport; a caller fetches the result through the method
that produced it, where `to_payload` applies.

# State, threading, history

`JobManager` guards everything with an `RLock` and keeps insertion order so
`list()` can return newest-first, like every other listing in the codebase.
`_trim()` drops the oldest **terminal** jobs beyond `max_history` (default 100);
running jobs are never trimmed. `default_manager()` is a lazily-created
process-wide singleton the surfaces share -- a host may use its own.

# Adoption is not finished

`quest/rpc/services.py` is the only caller today. The web backend still uses its
own `ThreadPoolExecutor` ([webui.md](webui.md)), so cancellation and progress
stop at the RPC surface. Finishing that is also what the shared-server work with
FPSIMP needs -- this vocabulary is the strongest candidate for the common job
model, since FPSIMP's states carry no cancellation concept at all
([FPSIMP combined-stack](../../../fpsimp/okf/architecture/combined-stack.md),
OPEN-3).
