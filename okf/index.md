---
okf_version: "0.2"
---

# QuEst Knowledge Bundle

An [Open Knowledge Format](https://github.com/GoogleCloudPlatform/knowledge-catalog)
**v0.2** bundle describing the QuEst codebase — the simulation pipeline, its
subsystems, the project-file contract, and the developer workflows. Authored for
agents and humans who need durable context on how this repository is organized.

This bundle is the project's **persistent memory**: what is learned about the
code, the environment, and the physics is written here rather than left in a
chat log. Start here, not in the source tree.

The conventions are shared with [FPSIMP](../../fpsimp/okf/index.md), which the
two projects follow so that a reader moving between them finds the same shape —
both are heading for one server. Format, validation and the 2026-08-06 migration
from v0.1: [references/okf-format.md](references/okf-format.md).

# Concepts

* [**Session handover**](handover.md) - **Read first.** Where the tree stands, what is uncommitted in both repositories, which decisions are waiting on a human, and the traps that cost time.
* [Overview](overview.md) - What QuEst is, the source layout, and the entry points.
* [Update log](log.md) - The running, dated narrative of what changed and why.

# Subdirectories

* [architecture](architecture/index.md) - The three-stage simulation pipeline, the project-JSON contract that every surface shares, and the four user-facing surfaces.
* [subsystems](subsystems/index.md) - The major code areas: core service layer, RPC, tasks, accessible volumes, dye diffusion, the quenching model, photon/decay generation, structure I/O, and the CLI/web/GUI front ends.
* [specs](specs/index.md) - The target architecture — the core/rpc/api/cli/gui split adopted from the ChiSurf plugin contract — and the backlog tracking where today's code diverges from it.
* [workflows](workflows/index.md) - Developer workflows: change tracking (the memory loop), environment, and testing.
* [references](references/index.md) - PET-quenching theory, known issues, the OKF format itself, and pointers to the maintained user documentation.
