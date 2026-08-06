# Subsystems

* [Core service layer](core-api.md) - `quest/core/` and the `quest.api` facade: project loading, `simulate_project`, `DecaySimulationResult`, residue-resolved analysis, plus the `project`/`scan`/`settings` domain modules.
* [RPC service layer](rpc.md) - `quest/rpc/`: the named methods every surface shares, the result envelope, and how a host application registers them.
* [Tasks](tasks.md) - `quest/tasks.py`: the one job mechanism every surface is meant to drive - observable progress, cooperative cancellation, and why `CANCELLING` is a state of its own.
* [Accessible volume](accessible-volume.md) - AV enumeration, the IMP.bff/LabelLib backend split, and the derived slow-factor and quenching-rate grids.
* [Dye diffusion](dye-diffusion.md) - `SimulateDiffusion` and `DonorDecay`: the Brownian walk in the AV and the model object that owns a run.
* [Quenching model](quenching-model.md) - Residue-type-specific PET quenching: contact radii, quenching centres, additive rates, multiplicative stickiness, and the per-type attribution.
* [Photons & decay](photon-decay.md) - The numba photon-trace kernels and the photon/curve decay modes.
* [Structure I/O](structure-io.md) - PDB/CIF/PQR parsing, the `Structure` object, coarse-graining, and PDB-ID fetching.
* [CLI](cli.md) - `quest simulate/scan/template/validate/gui`, dotted-path overrides and grid sweeps.
* [Web UI](webui.md) - The FastAPI backend, the Next.js frontend, and the Electron shell.
* [Qt GUI](gui.md) - The legacy desktop GUI and its optional-Qt story.
* [Translations](i18n.md) - English, German and French, translated once in the catalog's locale files and consumed by the docs, the Qt form and the web UI.
