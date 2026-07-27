# Architecture

* [Simulation pipeline](simulation-pipeline.md) - The three stages every run goes through: accessible volume, Brownian diffusion in that volume, photon/decay generation — and where each stage lives in the code.
* [Project schema](project-schema.md) - The project-JSON contract shared by the CLI, the API, the web UI and the GUI, and the parameter catalog that documents it.
* [Surfaces](surfaces.md) - The four user-facing surfaces (Python API, CLI, web UI, Qt GUI) and the rule that they are thin wrappers over one service layer.
* [Job artefacts](job-artefacts.md) - What a run writes to `jobs/<uuid>/`, and why bulk work turns it off.
