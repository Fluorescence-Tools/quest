# Specs

Target ("north star") architecture for QuEst, and the backlog tracking where
today's code diverges from it. Consult these before large refactors — a change
that moves the tree toward the target is worth more than one that merely works.

* [The 2026-07 programme](programme-2026-07.md) - The decisions scoping the current programme of work, with the alternative each one rejected — and the three places where it reverses what this bundle previously asserted. Read before proposing work in these areas.
* [Objectives](objectives.md) - Where QuEst is going scientifically: FRET as a first-class observable, and fitting dye parameters to measured decays. Not implemented — shape new work so these get cheaper, not harder.
* [Plugin integration — target](plugin-integration.md) - The core / rpc / api / cli / gui split QuEst adopts from the ChiSurf plugin contract, so the same code serves the CLI, the web UI, the desktop GUI, and a ChiSurf plugin without a second implementation.
* [Deduplication against imp-tricks](imp-tricks-dedup.md) - General fluorescence machinery belongs in imp-tricks under `IMP.bff`; QuEst is the application on top of it. What already exists there, what QuEst duplicates, and the rules for moving it.
* [Assessment](assessment.md) - The gap backlog: where the current tree diverges from the target, itemised with ids and status.
