"""Tests for the scan-oriented fast path and for seeded reproducibility.

A residue scan runs the same simulation hundreds of times and only reads the
scalar metrics, so `simulate_project(save_outputs=False)` skips the per-run job
directory. Separately, a project that pins ``random_seed`` must produce the same
numbers every time, including when several sites are simulated on worker
threads.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from quest.core import simulate_project
from quest.core.av import density2points
from quest.core.dye_diffusion import default_amino_acid_interactions
from quest.core.photon import simulate_photon_trace_rate

PDB_PATH = Path(__file__).parent / "148l.pdb"

# Sites in 148L chain E that carry a CB, have room for the dye, and sit close
# enough to quenchers that the quantum yield actually moves; a site with
# QY == 1.0 would pass a reproducibility check trivially.
#
# Re-picked 2026-07-27 when radii moved to IMP's (CHARMM-style, ~19 % larger
# than the Bondi values QuEst used to carry). Fatter atoms close pockets: 15
# and 17 have no accessible volume at all now, and only 43 of 151 CB sites on
# this chain remain labelable. QY under *this* project: 18 → 0.75, 96 → 0.09,
# 36 → 0.97. (The quantum yield depends on the project, not just the site:
# 118 quenches under the template but not under this one's chemistry.)
QUENCHED_SITES = (18, 96, 36)


def _project(residue: int, *, seed: int | None = 5, **overrides):
    donor = {"name": "D0", "D": 7.0}
    if seed is not None:
        donor["random_seed"] = seed
    project = {
        "tau0": 4.2,
        "n_photons": 4000,
        "t_max": 400.0,
        "t_step": 0.02,
        "dg": 0.5,
        "critical_distance": 8.5,
        "slow_radius": 8.5,
        "av_parameter": {"linker_length": 21.5, "linker_width": 0.5, "radius1": 3.5},
        "amino_acid_interactions": default_amino_acid_interactions(slow_factor=0.1),
        "attachment": {"chain": "E", "residue": residue, "atom": "CB"},
        "pdb": str(PDB_PATH),
        "n_bins": 64,
        "parallel_trajectories": 1,
        "fret": {"enabled": False, "dyes": [donor]},
    }
    project.update(overrides)
    return project


class TestSaveOutputs:
    def test_no_job_directory_is_written(self, tmp_path: Path) -> None:
        simulate_project(_project(QUENCHED_SITES[0]), project_dir=tmp_path, save_outputs=False)
        assert not (tmp_path / "jobs").exists()

    def test_job_directory_is_written_by_default(self, tmp_path: Path) -> None:
        simulate_project(_project(QUENCHED_SITES[0]), project_dir=tmp_path)
        jobs = list((tmp_path / "jobs").iterdir())
        assert len(jobs) == 1
        written = {path.name for path in jobs[0].iterdir()}
        assert "result.json" in written
        assert "av.mrc" in written

    def test_scalar_metrics_survive_the_fast_path(self, tmp_path: Path) -> None:
        """Everything a scan reports must still be there without the artefacts."""
        full = simulate_project(_project(QUENCHED_SITES[0]), project_dir=tmp_path)
        fast = simulate_project(_project(QUENCHED_SITES[0]), project_dir=tmp_path, save_outputs=False)

        assert fast.quantum_yield_donor == pytest.approx(full.quantum_yield_donor)
        assert fast.collisions_fraction == pytest.approx(full.collisions_fraction)
        assert fast.av_volume == pytest.approx(full.av_volume)
        assert fast.contact_volume == pytest.approx(full.contact_volume)
        assert fast.contact_to_free_ratio == pytest.approx(full.contact_to_free_ratio)

        by_name = {row["residue_name"]: row for row in fast.aa_residence}
        assert by_name, "aa_residence drives the scan's 'Quenched by' column"
        for row in full.aa_residence:
            assert by_name[row["residue_name"]]["mean_quench_rate"] == pytest.approx(
                row["mean_quench_rate"]
            )

    def test_viewer_point_clouds_follow_render_data_not_save_outputs(
        self, tmp_path: Path
    ) -> None:
        """Two questions, two flags.

        The thinned clouds exist for a plot or a 3D viewer; the job directory
        exists for the filesystem. Gating both on `save_outputs` cost the Qt
        form its trajectory and AV plots, because it passes `save_outputs=False`
        to keep a GUI run from writing `jobs/` into the working directory.
        """
        scan = simulate_project(
            _project(QUENCHED_SITES[0]), project_dir=tmp_path, save_outputs=False, render_data=False
        )
        assert scan.av_points is None
        assert scan.acv_points is None
        assert scan.trajectory_points is None

        gui = simulate_project(_project(QUENCHED_SITES[0]), project_dir=tmp_path, save_outputs=False)
        assert not (tmp_path / "jobs").exists(), "a GUI run must write nothing"
        assert gui.av_points is not None
        assert gui.trajectory_distance_to_mean is not None

        full = simulate_project(_project(QUENCHED_SITES[0]), project_dir=tmp_path)
        assert full.av_points is not None
        assert full.acv_points is not None


class TestSeededReproducibility:
    @pytest.mark.parametrize("residue", QUENCHED_SITES)
    def test_repeated_runs_agree(self, residue: int, tmp_path: Path) -> None:
        """Regression: the photon trace used to ignore ``random_seed``."""
        yields = [
            simulate_project(
                _project(residue), project_dir=tmp_path, save_outputs=False
            ).quantum_yield_donor
            for _ in range(3)
        ]
        assert len(set(yields)) == 1, f"non-reproducible quantum yields: {yields}"
        assert 0.0 < yields[0] < 1.0, "pick a site where quenching actually happens"

    def test_worker_threads_agree_with_the_main_thread(self, tmp_path: Path) -> None:
        """Numba keeps RNG state per thread, so a seeded run must not drift."""

        def run(residue: int) -> float:
            return simulate_project(
                _project(residue), project_dir=tmp_path, save_outputs=False
            ).quantum_yield_donor

        sequential = [run(residue) for residue in QUENCHED_SITES]
        with ThreadPoolExecutor(max_workers=len(QUENCHED_SITES)) as executor:
            threaded = list(executor.map(run, QUENCHED_SITES))
        assert sequential == threaded

    def test_an_unseeded_project_still_varies(self, tmp_path: Path) -> None:
        """Compare the decay curves, not the quantum yield.

        At a site the dye rarely reaches a quencher the yield saturates at 1.0,
        so four independent runs legitimately report the same scalar and the
        test used to fail at random. The photon arrival times differ every run
        regardless, which is what "the Monte-Carlo run did not repeat" means.
        """
        decays = {
            simulate_project(
                _project(QUENCHED_SITES[0], seed=None), project_dir=tmp_path, save_outputs=False
            ).donor_counts.tobytes()
            for _ in range(4)
        }
        assert len(decays) == 4, "without a seed the Monte-Carlo run should not repeat"

    def test_photon_trace_honours_its_seed(self) -> None:
        rates = np.linspace(0.0, 0.4, 512).astype(np.float32)
        first = simulate_photon_trace_rate(2000, rates, t_step=0.02, tau0=4.0, random_seed=11)
        second = simulate_photon_trace_rate(2000, rates, t_step=0.02, tau0=4.0, random_seed=11)
        other = simulate_photon_trace_rate(2000, rates, t_step=0.02, tau0=4.0, random_seed=12)

        assert np.array_equal(first[0], second[0])
        assert np.array_equal(first[1], second[1])
        assert not np.array_equal(first[1], other[1])


class TestDensityToPoints:
    def test_matches_a_plain_python_scan_of_the_grid(self) -> None:
        """The jitted kernel replaced a Python triple loop; keep them equivalent."""
        rng = np.random.default_rng(3)
        nx, ny, nz = 7, 5, 6
        grid = np.asfortranarray(rng.normal(size=(nx, ny, nz)))

        expected = [
            (ix * 0.5 - 1.0, iy * 0.5 + 2.0, iz * 0.5 + 0.25)
            for iz in range(nz)
            for iy in range(ny)
            for ix in range(nx)
            if grid[ix, iy, iz] > 0.0
        ]
        points = density2points(nx, ny, nz, 0.5, grid, (-1.0, 2.0, 0.25))

        assert points.shape == (len(expected), 3)
        assert np.allclose(points, np.array(expected))

    def test_an_empty_grid_yields_no_points(self) -> None:
        grid = np.zeros((4, 4, 4))
        assert density2points(4, 4, 4, 0.5, grid, (0.0, 0.0, 0.0)).shape == (0, 3)
