"""The Qt form must run a simulation and show the result.

Characterisation tests for the form's ▶ button. They exist because the desktop
surface is the one that drifted: it used to hold its state in Qt controls and
convert it to a project by hand, and the conversion is where it diverged from
the CLI. Now the form's state *is* the project and the run goes through
`quest.api`, so what these check is that the whole path — controls to service
layer to plots — still carries the numbers.

Qt and ChiSurf only; run in the canonical environment:

    QT_QPA_PLATFORM=offscreen \\
      /Users/tpeulen/mambaforge/envs/arm64/bin/python -m pytest tests/test_gui_simulation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("qtpy", reason="no Qt binding in this environment")
QtWidgets = pytest.importorskip("qtpy.QtWidgets", reason="no Qt binding")
QtCore = pytest.importorskip("qtpy.QtCore", reason="no Qt binding")

from quest.hosts import autoform_available  # noqa: E402

pytestmark = [
    pytest.mark.skipif(
        not autoform_available(),
        reason="the Qt form is rendered by ChiSurf's AutoForm (set CHISURF_ROOT)",
    ),
    # Every test here drives a real run, and the widget reports a failed run in
    # its status line rather than raising — so the error never reaches the
    # conftest hook that would skip it.
    pytest.mark.needs_av,
]

STRUCTURE = Path(__file__).parent / "148l.pdb"


@pytest.fixture(scope="module")
def app():
    yield QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


#: A labeling site on 148l where the dye actually meets a quencher. Picked to
#: match `tests/test_simulation_performance.py::QUENCHED_SITES`: a site with no
#: quencher in reach returns QY = 1.0, and a decay test that passes on an
#: unquenched site is not testing quenching. Re-picked when radii moved to
#: IMP's — 15 and 90 have no accessible volume with the larger atoms.
QUENCHED_SITE = {"chain": "E", "residue": 118, "atom": "CB"}


def _configure(widget, *, t_max: float = 400.0, seed: int = 5) -> None:
    """A small but non-degenerate run: enough frames to meet a quencher."""
    model = widget.model
    model.pdb = str(STRUCTURE)
    model.attachment_chain = QUENCHED_SITE["chain"]
    model.attachment_residue = QUENCHED_SITE["residue"]
    model.attachment_atom = QUENCHED_SITE["atom"]
    model.t_max = t_max
    model.t_step = 0.02
    model.n_photons = 100_000
    model.random_seed = seed
    model.fret_enabled = False


@pytest.fixture(scope="module")
def simulated(app):
    """A form driven through one small, seeded simulation."""
    from quest.gui import TransientDecayGenerator

    widget = TransientDecayGenerator()
    _configure(widget)
    assert widget.model.validate() == [], "the form built an invalid project"
    widget.update_all()
    yield widget
    widget.close()


class TestTheFormRunsASimulation:
    def test_it_produces_a_decay(self, simulated) -> None:
        result = simulated.model.results
        assert result is not None
        assert np.asarray(result.donor_counts).sum() > 0
        assert len(result.time) == len(result.donor_counts)

    def test_the_quantum_yield_shows_quenching(self, simulated) -> None:
        """Strictly below 1: the site was chosen so quenching happens."""
        quantum_yield = simulated.model.results.quantum_yield_donor
        assert 0.0 < quantum_yield < 1.0, (
            f"QY = {quantum_yield}: no quenching at all, so this run proves "
            "nothing about the quenching path"
        )

    def test_the_dye_visits_the_protein_surface(self, simulated) -> None:
        assert 0.0 <= simulated.model.results.collisions_fraction <= 1.0

    def test_a_trajectory_exists_and_moves(self, simulated) -> None:
        distances = np.asarray(simulated.model.results.trajectory_distance_to_mean)
        assert distances.shape[0] > 100
        assert distances.min() < distances.max()

    def test_the_accessible_volume_is_populated(self, simulated) -> None:
        assert np.asarray(simulated.model.results.av_points).shape[0] > 100

    def test_the_result_reaches_the_form(self, simulated) -> None:
        """The point of a GUI: the numbers are shown, not merely computed."""
        status = simulated.model.status
        assert "QY" in status and "τ" in status
        assert f"{simulated.model.results.quantum_yield_donor:.3f}" in status

    def test_the_plots_are_filled(self, simulated) -> None:
        decay = simulated.model.decay_series()
        assert decay and len(decay[0]["y"]) > 0

        trajectory = simulated.model.trajectory_series()
        assert trajectory and len(trajectory[0]["y"]) > 100

        autocorrelation = simulated.model.autocorrelation_series()
        assert autocorrelation and len(autocorrelation[0]["y"]) > 0

    def test_the_pyqtgraph_canvases_carry_curves(self, simulated) -> None:
        """The pyqtgraph PlotWidgets must actually hold data after refresh.

        The model data series being populated (the test above) proves the
        simulation ran; this proves ``refresh()`` pushed it to the canvases
        the user sees.
        """
        for attr, min_points, n_series in (
            ("decay_plot", 2, 1),
            ("traj_plot", 100, 1),
            ("acf_plot", 2, 1),
        ):
            canvas = getattr(simulated, attr)
            assert canvas.plot_widget is not None, f"{attr} has no PlotWidget"
            items = canvas.plot_widget.getPlotItem().listDataItems()
            assert len(items) == n_series, (
                f"{attr} expected {n_series} curve(s), got {len(items)}"
            )
            x, y = items[0].getData()
            assert len(y) >= min_points, (
                f"{attr} curve has {len(y)} points, expected >= {min_points}"
            )

    def test_a_fret_run_populates_two_decay_curves(self, fret_simulated) -> None:
        """FRET adds a second decay curve — verify it reaches the canvas."""
        canvas = fret_simulated.decay_plot
        items = canvas.plot_widget.getPlotItem().listDataItems()
        assert len(items) == 2, f"expected donor + FRET, got {len(items)} curves"

    def test_a_screenshot_is_non_blank(self, simulated, tmp_path) -> None:
        """Grab the widget to a PNG — a blank image means nothing was drawn."""
        simulated.resize(1400, 900)
        from qtpy.QtGui import QImage

        pixmap = simulated.grab()
        assert not pixmap.isNull(), "grab() produced a null pixmap"
        image = pixmap.toImage()
        # A completely white or black image means nothing rendered.
        pixels = set()
        for y in range(0, image.height(), max(1, image.height() // 50)):
            for x in range(0, image.width(), max(1, image.width() // 50)):
                pixels.add(image.pixel(x, y))
        assert len(pixels) > 10, (
            f"screenshot has only {len(pixels)} distinct colours; "
            "the widget is probably blank"
        )

    def test_it_repaints_after_the_run(self, simulated) -> None:
        simulated.resize(1400, 900)
        assert not simulated.grab().isNull()


class TestWhatWasSimulatedIsWhatTheFormShows:
    def test_the_project_is_the_ui_state(self, simulated) -> None:
        project = simulated.to_project_dict()
        assert project["t_max"] == pytest.approx(400.0)
        assert project["t_step"] == pytest.approx(0.02)
        assert project["n_photons"] == 100_000
        assert project["attachment"]["residue"] == QUENCHED_SITE["residue"]

    def test_the_result_carries_the_project_it_ran(self, simulated) -> None:
        """`results.project` is the resolved project — not a different one."""
        ran = simulated.model.results.project
        assert ran["attachment"]["residue"] == QUENCHED_SITE["residue"]
        assert ran["t_max"] == pytest.approx(400.0)

    def test_a_gui_run_leaves_no_artefacts(self, simulated, tmp_path) -> None:
        """`save_outputs=False`: the GUI must not litter `jobs/` where it started."""
        assert simulated.model.results.job_id is None or not (
            Path.cwd() / "jobs" / str(simulated.model.results.job_id)
        ).exists()


class TestTheSavedFileIsAProject:
    """One JSON for every surface — the GUI writes what the CLI reads."""

    def test_it_saves_and_reloads(self, simulated, tmp_path) -> None:
        path = simulated.model.save_project(tmp_path / "project.json")
        written = json.loads(path.read_text(encoding="utf8"))
        assert written["attachment"]["residue"] == QUENCHED_SITE["residue"]

        from quest.gui import TransientDecayGenerator

        other = TransientDecayGenerator()
        try:
            other.model.load_project(path)
            assert other.to_project_dict() == simulated.to_project_dict()
        finally:
            other.close()

    def test_the_service_layer_accepts_it(self, simulated, tmp_path) -> None:
        """The real check: what the form saved must validate as a project."""
        from quest.api import load_project, validate_project

        path = simulated.model.save_project(tmp_path / "project.json")
        validate_project(load_project(path))


class TestSeededRunsAreReproducible:
    def test_two_runs_with_the_same_seed_agree(self, app) -> None:
        from quest.gui import TransientDecayGenerator

        def run():
            widget = TransientDecayGenerator()
            _configure(widget, t_max=200.0, seed=17)
            widget.update_all()
            result = widget.model.results
            outcome = (
                result.quantum_yield_donor,
                np.asarray(result.donor_counts).copy(),
            )
            widget.close()
            return outcome

        first_qy, first_decay = run()
        second_qy, second_decay = run()
        assert first_qy == pytest.approx(second_qy)
        np.testing.assert_array_equal(first_decay, second_decay)


#: A donor/acceptor pair on 148l far enough apart that transfer is partial —
#: not ~0 (which would pass a "0 <= E <= 1" assertion while proving nothing) and
#: not ~1 (which would hide an R0 read in the wrong unit).
FRET_ACCEPTOR = {"chain": "E", "residue": 96, "atom": "CB"}


@pytest.fixture(scope="module")
def fret_simulated(app):
    """The form driven through a FRET run, acceptor and all."""
    from quest.gui import TransientDecayGenerator

    widget = TransientDecayGenerator()
    _configure(widget)
    model = widget.model
    model.acceptor_chain = FRET_ACCEPTOR["chain"]
    model.acceptor_residue = FRET_ACCEPTOR["residue"]
    model.acceptor_atom = FRET_ACCEPTOR["atom"]
    model.fret_enabled = True
    assert model.validate() == [], "the FRET form built an invalid project"
    widget.update_all()
    yield widget
    widget.close()


class TestTheFormRunsFret:
    """The desktop surface must reach FRET, not just the donor channel.

    The rebuilt form binds FRET through the same project keys as every other
    surface, but "it should work" is what was said about the GUI's quenching
    chemistry too. These run it.
    """

    def test_it_produces_both_decays(self, fret_simulated) -> None:
        result = fret_simulated.model.results
        assert result.fret_counts is not None, "no FRET channel was simulated"
        assert np.asarray(result.fret_counts).sum() > 0
        assert len(result.fret_counts) == len(result.donor_counts)

    def test_the_efficiency_is_partial(self, fret_simulated) -> None:
        """Strictly between 0 and 1: the site pair was chosen for that."""
        efficiency = fret_simulated.model.results.fret_efficiency
        assert efficiency is not None
        assert 0.05 < efficiency < 0.98, (
            f"E = {efficiency}: this pair should transfer partially, so this "
            "value means the geometry or R0 is not reaching the simulation"
        )

    def test_the_acceptor_quenches_the_donor(self, fret_simulated) -> None:
        """E is 1 − QY_DA/QY_D, so the FRET channel must be the darker one."""
        result = fret_simulated.model.results
        assert result.quantum_yield_fret < result.quantum_yield_donor
        derived = 1.0 - result.quantum_yield_fret / result.quantum_yield_donor
        assert derived == pytest.approx(result.fret_efficiency, abs=1e-6)

    def test_r0_reaches_the_simulation_in_angstrom(self, app) -> None:
        """A larger Förster radius must transfer more. Nanometres would not."""
        from quest.gui import TransientDecayGenerator

        def efficiency_at(r0: float) -> float:
            widget = TransientDecayGenerator()
            _configure(widget, t_max=200.0)
            widget.model.acceptor_chain = FRET_ACCEPTOR["chain"]
            widget.model.acceptor_residue = FRET_ACCEPTOR["residue"]
            widget.model.acceptor_atom = FRET_ACCEPTOR["atom"]
            widget.model.fret_enabled = True
            widget.model.fret_R0 = r0
            widget.update_all()
            value = widget.model.results.fret_efficiency
            widget.close()
            return value

        assert efficiency_at(65.0) > efficiency_at(40.0)

    def test_the_form_shows_the_efficiency(self, fret_simulated) -> None:
        """A number computed but not displayed is not a result."""
        status = fret_simulated.model.status
        assert "E =" in status
        assert f"{fret_simulated.model.results.fret_efficiency:.3f}" in status

    def test_both_curves_reach_the_decay_plot(self, fret_simulated) -> None:
        names = [series["name"] for series in fret_simulated.model.decay_series()]
        assert len(names) == 2, names
        assert all(len(series["y"]) > 0 for series in fret_simulated.model.decay_series())

    def test_turning_fret_off_removes_the_channel(self, app) -> None:
        """The toggle must actually gate it; a stale acceptor must not linger."""
        from quest.gui import TransientDecayGenerator

        widget = TransientDecayGenerator()
        try:
            _configure(widget, t_max=200.0)
            widget.model.acceptor_chain = FRET_ACCEPTOR["chain"]
            widget.model.acceptor_residue = FRET_ACCEPTOR["residue"]
            widget.model.acceptor_atom = FRET_ACCEPTOR["atom"]
            widget.model.fret_enabled = False
            widget.update_all()
            assert widget.model.results.fret_efficiency is None
            assert len(widget.model.decay_series()) == 1
        finally:
            widget.close()

    def test_a_fret_project_round_trips(self, fret_simulated, tmp_path) -> None:
        """The acceptor and R0 must survive save/load like everything else."""
        from quest.api import load_project, validate_project

        path = fret_simulated.model.save_project(tmp_path / "fret.json")
        restored = load_project(path)
        validate_project(restored)
        assert restored["fret"]["enabled"] is True
        assert restored["fret"]["dyes"][1]["attachment"]["residue"] == (
            FRET_ACCEPTOR["residue"]
        )
        assert restored["fret"]["R0_matrix"][0][1] == pytest.approx(
            fret_simulated.model.fret_R0
        )


class TestTheWindowStaysAliveDuringARun:
    """The ▶ button must not freeze the form for the length of a simulation.

    The run used to be a plain call on the GUI thread. `LAY-07` had already
    built one observable, cancelable mechanism for long work — the desktop was
    the surface that did not use it. What these pin is the *user-visible*
    consequence: events still get delivered while a run is in flight.
    """

    def test_events_are_delivered_while_a_run_is_in_flight(self, app) -> None:
        from quest.gui import TransientDecayGenerator

        widget = TransientDecayGenerator()
        try:
            _configure(widget, t_max=300.0, seed=3)
            ticks = []
            timer = QtCore.QTimer()
            timer.setInterval(10)
            timer.timeout.connect(lambda: ticks.append(1))
            timer.start()

            widget.update_all()
            timer.stop()

            assert widget.model.results is not None, "the run must still produce one"
            assert ticks, (
                "no timer event was delivered during the run — the GUI thread "
                "was blocked, which is the defect this replaced"
            )
        finally:
            widget.close()

    def test_the_run_button_is_disabled_while_running(self, app) -> None:
        from quest.gui import TransientDecayGenerator

        widget = TransientDecayGenerator()
        try:
            _configure(widget, t_max=300.0, seed=11)
            states = []
            timer = QtCore.QTimer()
            timer.setInterval(5)
            timer.timeout.connect(
                lambda: states.append(
                    (widget.btn_run.isEnabled(), widget.btn_cancel.isEnabled())
                )
            )
            timer.start()
            widget.update_all()
            timer.stop()

            assert (False, True) in states, (
                "Run stayed enabled (or Cancel stayed disabled) during the run"
            )
            assert widget.btn_run.isEnabled()
            assert not widget.btn_cancel.isEnabled()
        finally:
            widget.close()

    def test_a_failed_run_leaves_the_buttons_usable(self, app) -> None:
        """A bad project must not leave the form stuck with Run disabled.

        Clicking ▶ no longer raises out of the slot: the failure is reported in
        the status line instead. `ProjectFormModel.run_simulation` still raises
        for headless callers, which is where an exception is useful.
        """
        from quest.gui import TransientDecayGenerator

        widget = TransientDecayGenerator()
        try:
            widget.model.pdb = "/nonexistent/structure.pdb"
            widget.update_all()

            assert widget.model.results is None
            assert widget.btn_run.isEnabled()
            assert not widget.btn_cancel.isEnabled()
            assert "Failed" in widget.model.status
        finally:
            widget.close()

    def test_a_second_click_while_running_is_ignored(self, app) -> None:
        """Pumping events means the button that started this can re-enter it."""
        from quest.gui import TransientDecayGenerator

        widget = TransientDecayGenerator()
        try:
            _configure(widget, t_max=200.0, seed=9)
            reentered = []

            def click_again() -> None:
                if widget.model.simulation_running:
                    reentered.append(widget.update_all())

            timer = QtCore.QTimer()
            timer.setInterval(10)
            timer.timeout.connect(click_again)
            timer.start()

            result = widget.update_all()
            timer.stop()

            assert result is not None
            assert all(r is None for r in reentered), (
                "a re-entrant click started a second run"
            )
        finally:
            widget.close()
