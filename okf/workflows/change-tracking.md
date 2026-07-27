---
type: Playbook
title: Change tracking
description: The loop for every material change — update OKF, keep it traceable, commit. This is how the memory layer stays true.
resource: okf/log.md
tags: [process, okf, provenance, git, memory]
timestamp: '2026-07-26T00:00:00Z'
---

# OKF is the memory; the loop is what keeps it true

`okf/` is the durable, agent-readable knowledge layer beside the code. It is
only worth reading if it stays in sync with the tree, so **every material change
runs the same loop**.

## The loop

1. **Make the change** — prefer a clean implementation over a patch. QuEst is
   under active development; a coherent restructuring beats a workaround.
2. **Update the matching OKF concept** — the `architecture/`, `subsystems/`,
   `workflows/` or `references/` page that owns the changed behaviour. If the
   change invalidates a sentence in a concept, fix the sentence in the *same*
   change. A wrong memory is worse than no memory.
3. **Append a dated bullet to [okf/log.md](../log.md)** under today's
   `## YYYY-MM-DD` heading — one bullet per landed unit of work, saying *what*
   changed, *why*, how it was verified, and which concepts it touches. On a
   conflict between a code comment and OKF, OKF wins (it is newer).
4. **Update the user documentation** — `okf/` is the knowledge layer for
   *agents*; `doc/` and `README.md` are the manual for *people*. A change to
   user-visible behaviour is not finished until both are updated. Parameter help
   text is edited in `quest/settings/parameter_catalog.json` and then
   regenerated: `python doc/generate_parameter_docs.py` — never hand-edit
   `doc/parameters.md`.
5. **Verify** — run the suite (see [testing.md](testing.md)) and put the number
   in both the log bullet and the commit body.
6. **Commit** — a focused commit per coherent unit.

## Learning is part of the loop

The point of a memory layer is that a discovery costs once. When you learn
something that was not obvious from the code — a dependency that is not
installed, a kernel that must stay sequential, a parameter whose default was
chosen by measurement, a physical assumption — it goes into the concept that
owns it, or into
[references/known-issues.md](/references/known-issues.md) when it is a defect
that cannot be fixed now. Never leave it only in the chat.

A finding that is not worth writing down is either trivial or not really
understood. Decide which.

## Git conventions

These are carried over from the sibling ChiSurf repository, which the same
author works the same way:

- **Commit locally; do not push.**
- **No commit trailers of any kind** — no `Co-Authored-By`, no tool attribution.
  Commits are authored by tpeulen; the message is plain subject + body.
- **Never destroy uncommitted work.** The tree may hold edits that are not
  yours. Banned: `git reset --hard`, `git checkout -- <path>` / `git restore`,
  `git clean -f`, `git stash`, any `--force`, and the sweeping
  `git add -A` / `git commit -a`. Commit only your own changes. Note the
  pathspec footgun: `git commit -- <path>` commits that path's *working-tree*
  content, so use it only for files that are wholly yours.
- The working tree currently carries a large amount of uncommitted work
  (see the log entry for 2026-07-26) — be especially careful here.

## What "traceable" means

Anyone, or any later agent, should be able to reconstruct *what changed and why*
from durable artifacts alone:

- **`okf/log.md`** — the running narrative; the first place to look.
- **The concepts** — the current state of the world, not its history.
- **Git history** — small, self-describing, verified-green commits.
