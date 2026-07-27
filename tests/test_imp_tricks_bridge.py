"""Splicing an imp-tricks checkout into the live interpreter.

General fluorescence machinery lives in imp-tricks under ``IMP.bff``
(okf/specs/imp-tricks-dedup.md). Reaching it from a checkout is not just a
``sys.path`` entry: where IMP is compiled, ``IMP`` and ``IMP.bff`` are regular
packages that shadow the checkout entirely.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from quest.imp_tricks import enable_imp_tricks, imp_tricks_src

pytestmark = pytest.mark.skipif(
    imp_tricks_src() is None, reason="no imp-tricks checkout found"
)


def test_the_checkout_is_found() -> None:
    src = imp_tricks_src()
    assert src is not None and (src / "IMP" / "bff").is_dir()


def test_enabling_reports_the_subpackages() -> None:
    enabled = enable_imp_tricks()
    assert "IMP.bff" in enabled
    assert "IMP.cgmol" in enabled


def test_the_general_fluorescence_modules_become_importable() -> None:
    enable_imp_tricks()
    compute_av = importlib.import_module("IMP.bff.av").compute_av
    assert callable(compute_av)

    metrics = importlib.import_module("IMP.bff.distance_metrics")
    # At r = R0 transfer and emission are equally likely.
    assert metrics.fret_efficiency(52.0, 52.0) == pytest.approx(0.5)


def test_a_compiled_imp_bff_is_not_displaced() -> None:
    """The checkout must sit *alongside* a compiled IMP.bff, not replace it."""
    imp_bff = importlib.import_module("IMP.bff")
    had_compiled_av = hasattr(imp_bff, "AV")

    enable_imp_tricks()

    assert hasattr(importlib.import_module("IMP.bff"), "AV") == had_compiled_av
    assert callable(importlib.import_module("IMP.bff.av").compute_av)


def test_enabling_twice_changes_nothing() -> None:
    first = enable_imp_tricks()
    before = list(importlib.import_module("IMP.bff").__path__)
    second = enable_imp_tricks()
    after = list(importlib.import_module("IMP.bff").__path__)
    assert first == second
    assert before == after


def test_a_missing_checkout_is_not_an_error(tmp_path) -> None:
    assert enable_imp_tricks(tmp_path / "nowhere") == []


def test_importing_quest_bridges_the_checkout_automatically() -> None:
    """No ceremony: using QuEst is enough to reach imp-tricks from source."""
    imp_av = importlib.import_module("quest.core.imp_av")
    assert "IMP.bff" in imp_av.IMP_TRICKS_ENABLED
    assert imp_av.imp_tricks_av_available()


def test_the_compiled_and_source_avs_are_distinguished() -> None:
    """``HAS_IMP_BFF`` is about the compiled decorator, not imp-tricks' module.

    Conflating them is what made QuEst select a broken backend: the flag was
    read as "imp-tricks is available" when it means "IMP was built with bff".
    """
    imp_av = importlib.import_module("quest.core.imp_av")
    compiled = imp_av.HAS_IMP_BFF
    from_source = imp_av.imp_tricks_av_available()
    assert isinstance(compiled, bool) and isinstance(from_source, bool)
    # The checkout is present in this test session, the compiled decorator may
    # or may not be — they are independent.
    assert from_source is True


class TestTheUpstreamAvMatchesQuEsts:
    """The precondition for deleting QuEst's AV: upstream must agree exactly.

    `DUP-01` is the largest deletion QuEst has left — `imp_av.py` and the `AV`
    class both become `IMP.bff.av.compute.compute_av`. That is only safe while
    the two produce the *same volume*, so this pins it. It skips wherever
    imp-tricks or LabelLib is missing, which includes the base environment.
    """

    STRUCTURE = Path(__file__).parent / "148l.pdb"
    # Re-picked with the move to IMP radii: residue 15 has no accessible
    # volume any more, so it cannot compare two backends.
    SITE = dict(chain="E", residue=118, atom="CB")
    AV_PARAMS = dict(linker_length=21.5, linker_width=0.5, radius1=3.5)

    @pytest.fixture(scope="class")
    def clouds(self):
        pytest.importorskip("LabelLib", reason="LabelLib is not installed")
        from quest.imp_tricks import enable_imp_tricks

        enable_imp_tricks()
        compute = pytest.importorskip(
            "IMP.bff.av.compute", reason="imp-tricks checkout not reachable"
        )
        if not getattr(compute, "_HAS_LABELLIB", False):
            pytest.skip("imp-tricks reports no usable LabelLib backend")

        import numpy as np

        from quest.core.av import AV
        from quest.core.structure import Structure

        structure = Structure(str(self.STRUCTURE))
        atoms = structure.atoms
        xyz = np.ascontiguousarray(atoms["coord"], dtype=np.float64)
        vdw = np.ascontiguousarray(atoms["radius"], dtype=np.float64)

        def _text(values):
            return np.array(
                [v.decode().strip() if isinstance(v, bytes) else str(v).strip()
                 for v in values]
            )

        selected = np.flatnonzero(
            (_text(atoms["chain"]) == self.SITE["chain"])
            & (atoms["res_id"] == self.SITE["residue"])
            & (_text(atoms["atom_name"]) == self.SITE["atom"])
        )
        assert selected.size, "the labeling site is not in the structure"

        upstream = np.asarray(
            compute.compute_av(
                xyz, vdw, xyz[selected[0]],
                linker_length=self.AV_PARAMS["linker_length"],
                linker_width=self.AV_PARAMS["linker_width"],
                dye_radii=(self.AV_PARAMS["radius1"], 0.0, 0.0),
                grid_resolution=0.5,
                backend="labellib",
            ).points
        )[:, :3]

        own = AV(
            structure,
            residue_seq_number=self.SITE["residue"],
            atom_name=self.SITE["atom"],
            chain_identifier=self.SITE["chain"],
            simulation_grid_resolution=0.5,
            save_av=False,
            verbose=False,
            **self.AV_PARAMS,
        ).points
        return upstream, own

    def test_the_volumes_are_not_empty(self, clouds) -> None:
        """Upstream used to return nothing at all, silently."""
        upstream, own = clouds
        assert upstream.shape[0] > 1000
        assert own.shape[0] > 1000

    def test_the_volumes_are_identical(self, clouds) -> None:
        import numpy as np

        upstream, own = clouds
        assert upstream.shape == own.shape
        a = upstream[np.lexsort(upstream.T)]
        b = own[np.lexsort(own.T)]
        np.testing.assert_array_equal(a, b)


class TestTheImpBackendIsNotReadyToBePrimary:
    """Why `AV_BACKENDS` still starts with LabelLib.

    The intended order is IMP.bff first, LabelLib only where IMP.bff is not
    built. These pin the two defects that stop it, so the day they are fixed the
    tests fail and say so, rather than the ordering staying wrong by inertia.
    """

    def test_the_imp_backend_uses_the_structure_it_is_given(self) -> None:
        """Fixed 2026-07-27: it used to re-read the file from disk.

        That discarded every in-memory transformation, so a coarse-grained
        project silently got the all-atom volume, and `.cif` failed because the
        re-read was `read_pdb` only. It now builds the IMP model from the atom
        array, which makes both impossible — and means format support lives in
        exactly one place (okf/specs/structure-io.md).
        """
        import inspect

        from quest.core import imp_av

        source = inspect.getsource(imp_av.build_imp_accessible_volume)
        assert "read_pdb" not in source and "read_mmcif" not in source, (
            "the IMP backend is parsing files again; it must take the atom array"
        )
        assert "structure.atoms" in source

    def test_labellib_is_still_first(self) -> None:
        from quest.core.av import AV_BACKENDS

        assert AV_BACKENDS[0] == "labellib", (
            "IMP.bff was promoted; make sure the coarse-graining and mmCIF "
            "defects above are actually fixed first"
        )

    def test_the_allowed_sphere_floor_is_enforced(self) -> None:
        """Below ~2 A IMP's path search starts nowhere and returns nothing."""
        from quest.core.av import MIN_IMP_ALLOWED_SPHERE_RADIUS

        assert MIN_IMP_ALLOWED_SPHERE_RADIUS >= 2.0


class TestTheImpAvBackendRuns:
    """Exercise the IMP.bff path itself, which the suite otherwise never enters.

    `AV_BACKENDS` starts with LabelLib, so every other test takes that branch.
    `build_imp_accessible_volume` shipped a `Model.get_particles()` call that
    does not exist and nothing noticed — a whole backend with no coverage.
    """

    STRUCTURE = Path(__file__).parent / "148l.pdb"
    SITE = dict(residue_seq_number=36, atom_name="CB", chain_identifier="E")
    PARAMS = dict(
        linker_length=21.5,
        linker_width=0.5,
        radii=(3.5, 0.0, 0.0),
        allowed_sphere_radius=2.0,
        simulation_grid_resolution=0.5,
    )

    @pytest.fixture(scope="class")
    def structure(self):
        self._require_compiled_imp_bff()
        from quest.core.structure import Structure

        return Structure(str(self.STRUCTURE), make_coarse=False)

    @staticmethod
    def _require_compiled_imp_bff():
        """`import IMP.bff` succeeding is not enough.

        In the base environment the *source* `IMP.bff` from the imp-tricks
        checkout imports fine — there is no compiled one to shadow it — but the
        C++ `IMP.bff.AV` this backend needs is absent. `HAS_IMP_BFF` is the
        flag that actually asks that question.
        """
        from quest.core.imp_av import HAS_IMP_BFF

        if not HAS_IMP_BFF:
            pytest.skip("no compiled IMP.bff (the C++ AV class) in this interpreter")

    def _build(self, structure):
        from quest.core.imp_av import build_imp_accessible_volume

        return build_imp_accessible_volume(structure, **self.SITE, **self.PARAMS)

    def test_it_produces_a_volume(self, structure) -> None:
        result = self._build(structure)
        assert result.points.shape[0] > 10_000
        assert result.points.shape[1] == 3

    def test_it_agrees_with_labellib_now_that_radii_are_shared(self, structure) -> None:
        """Both backends read the same atom array, so they can only differ by
        algorithm. They came 22 % apart while QuEst kept its own radius table;
        with IMP's radii used for both, the gap is ~10 %."""
        from quest.core.av import AV

        labellib = AV(
            structure,
            residue_seq_number=self.SITE["residue_seq_number"],
            atom_name=self.SITE["atom_name"],
            chain_identifier=self.SITE["chain_identifier"],
            simulation_grid_resolution=0.5,
            save_av=False,
            verbose=False,
            linker_length=21.5,
            linker_width=0.5,
            radius1=3.5,
        ).points
        imp = self._build(structure).points

        ratio = imp.shape[0] / labellib.shape[0]
        assert 0.8 < ratio < 1.2, f"backends diverged: IMP/LabelLib = {ratio:.2f}"
        centroid_gap = float(
            ((imp.mean(axis=0) - labellib.mean(axis=0)) ** 2).sum() ** 0.5
        )
        assert centroid_gap < 2.0, f"backends disagree on position by {centroid_gap:.1f} A"

    def test_it_honours_the_structure_it_is_given(self) -> None:
        """The defect this replaced: it re-read the file, so coarse-graining
        was silently discarded and coarse came out identical to all-atom."""
        self._require_compiled_imp_bff()
        from quest.core.structure import Structure

        all_atom = self._build(Structure(str(self.STRUCTURE), make_coarse=False))
        coarse = self._build(Structure(str(self.STRUCTURE), make_coarse=True))
        assert coarse.points.shape[0] != all_atom.points.shape[0], (
            "coarse-grained and all-atom gave the same volume — the backend is "
            "reading the file again instead of the structure"
        )
