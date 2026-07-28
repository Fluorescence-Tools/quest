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
    av = importlib.import_module("quest.core.av")
    assert "IMP.bff" in av.IMP_TRICKS_ENABLED
    assert av.imp_tricks_av_available()


def test_the_compiled_and_source_avs_are_distinguished() -> None:
    """``HAS_IMP_BFF`` is about the compiled decorator, not imp-tricks' module.

    Conflating them is what made QuEst select a broken backend: the flag was
    read as "imp-tricks is available" when it means "IMP was built with bff".
    """
    av = importlib.import_module("quest.core.av")
    compiled = av.HAS_IMP_BFF
    from_source = av.imp_tricks_av_available()
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

    **Each side is pinned to the same backend on purpose.** This compares two
    *implementations of the same algorithm*, so it must not be left to pick up
    whatever `AV_BACKENDS` prefers — when IMP.bff became primary it silently
    turned into a LabelLib-vs-IMP.bff comparison and failed for the wrong
    reason. The two backends are not expected to agree with *each other*; they
    differ by ~10 % on volume, which `TestTheImpAvBackendRuns` measures.

    Both backends are covered as of 2026-07-27. Upstream's `imp_bff` branch had
    never run — it raised on every input — and was fixed in imp-tricks along
    the way; comparing only LabelLib would leave the path QuEst actually
    defaults to unpinned.
    """

    STRUCTURE = Path(__file__).parent / "148l.pdb"
    # Re-picked with the move to IMP radii: residue 15 has no accessible
    # volume any more, so it cannot compare two backends.
    SITE = dict(chain="E", residue=118, atom="CB")
    AV_PARAMS = dict(linker_length=21.5, linker_width=0.5, radius1=3.5)

    @pytest.fixture(scope="class", params=["labellib", "imp_bff"])
    def clouds(self, request):
        backend = request.param
        from quest.imp_tricks import enable_imp_tricks

        enable_imp_tricks()
        compute = pytest.importorskip(
            "IMP.bff.av.compute", reason="imp-tricks checkout not reachable"
        )
        if backend == "labellib":
            pytest.importorskip("LabelLib", reason="LabelLib is not installed")
            if not getattr(compute, "_HAS_LABELLIB", False):
                pytest.skip("imp-tricks reports no usable LabelLib backend")
        else:
            from quest.core.av import HAS_IMP_BFF

            if not (HAS_IMP_BFF and getattr(compute, "_HAS_IMP_BFF", False)):
                pytest.skip("no compiled IMP.bff (the C++ AV class) in this interpreter")

        import numpy as np

        from quest.core.av import AV, DEFAULT_ALLOWED_SPHERE_RADIUS
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
                backend=backend,
                allowed_sphere_radius=DEFAULT_ALLOWED_SPHERE_RADIUS,
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
            av_backend=backend,
            **self.AV_PARAMS,
        ).points
        return upstream, own

    def test_the_volumes_are_not_empty(self, clouds) -> None:
        """Upstream used to return nothing at all, silently."""
        upstream, own = clouds
        assert upstream.shape[0] > 1000
        assert own.shape[0] > 1000

    def test_the_volumes_are_identical(self, clouds) -> None:
        """Same voxels, to well under a thousandth of a grid step.

        Not bit-equality, and the reason is worth keeping: on the `imp_bff`
        path the two sides reach the same points by different arithmetic.
        QuEst rebuilds its cloud from the density grid in float64, upstream
        takes IMP's own `get_xyz_density()`, which has been through float32.
        They agree to ~2e-6 Å — a millionth of the 0.5 Å voxel — so the
        tolerance is about the last bits of a float, not about slack in the
        volume. A real disagreement is a whole voxel, 5e5 times larger.
        """
        import numpy as np

        upstream, own = clouds
        assert upstream.shape == own.shape
        a = upstream[np.lexsort(upstream.T)]
        b = own[np.lexsort(own.T)]
        np.testing.assert_allclose(a, b, atol=1e-4, rtol=0)


class TestThePrimaryBackendStaysFitToBePrimary:
    """The conditions IMP.bff had to meet before it could be promoted.

    It was second for as long as it re-read structures from disk. It is first
    now, which makes these two properties load-bearing rather than aspirational:
    they are what stops the default path silently computing the wrong volume.
    The sibling class that asserted `AV_BACKENDS[0] == "labellib"` is gone --
    it existed to make the promotion deliberate, and the promotion happened.
    """

    def test_the_primary_backend_is_imp_bff(self) -> None:
        """LabelLib remains the fallback where IMP.bff is not compiled."""
        from quest.core.av import AV_BACKENDS

        assert AV_BACKENDS == ("imp_bff", "labellib")

    def test_the_backend_cannot_re_read_the_file(self) -> None:
        """Fixed 2026-07-27, and structural since `DUP-01`.

        The IMP backend used to re-read the file from disk, which discarded
        every in-memory transformation: a coarse-grained project silently got
        the all-atom volume, and `.cif` failed because the re-read was
        `read_pdb` only. This was an assertion about the *source text* of
        `build_imp_accessible_volume`; that function is gone, and the property
        is now guaranteed by the signature instead — `IMP.bff.av.compute_av`
        takes `(atoms_xyz, atoms_vdw, source_xyz)` and has no path to a file at
        all. So this checks the signature, and
        `TestTheImpAvBackendRuns.test_it_honours_the_structure_it_is_given`
        checks the behaviour.
        """
        import inspect

        from quest.core.av import _compute_av

        parameters = inspect.signature(_compute_av()).parameters
        assert "atoms_xyz" in parameters and "source_xyz" in parameters
        assert not any("file" in name or "path" in name for name in parameters), (
            "compute_av grew a file argument; the AV must take the atom array"
        )

    def test_the_allowed_sphere_floor_is_enforced(self) -> None:
        """Below ~2 A IMP's path search starts nowhere and returns nothing."""
        from quest.core.av import MIN_IMP_ALLOWED_SPHERE_RADIUS

        assert MIN_IMP_ALLOWED_SPHERE_RADIUS >= 2.0


class TestTheImpAvBackendRuns:
    """Exercise the IMP.bff path directly, and compare it against LabelLib.

    Written when `AV_BACKENDS` still started with LabelLib and no test entered
    this branch at all: `build_imp_accessible_volume` shipped a
    `Model.get_particles()` call that does not exist and nothing noticed. It is
    the default path now, so the comparison below is what says how far the
    fallback platforms sit from it.
    """

    STRUCTURE = Path(__file__).parent / "148l.pdb"
    SITE = dict(residue_seq_number=36, atom_name="CB", chain_identifier="E")
    PARAMS = dict(
        linker_length=21.5,
        linker_width=0.5,
        radius1=3.5,
        radius2=0.0,
        radius3=0.0,
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
        from quest.core.av import HAS_IMP_BFF

        if not HAS_IMP_BFF:
            pytest.skip("no compiled IMP.bff (the C++ AV class) in this interpreter")

    def _build(self, structure):
        """`AV` pinned to the IMP.bff backend.

        This went through `build_imp_accessible_volume` until `DUP-01` deleted
        it; the backend is `IMP.bff.av.compute_av` now, and `AV` is the only
        thing left in QuEst that reaches it.
        """
        from quest.core.av import AV

        return AV(
            structure,
            save_av=False,
            verbose=False,
            av_backend="imp_bff",
            **self.SITE,
            **self.PARAMS,
        )

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


class TestTheAdoptedDistanceKernelsBehave:
    """`DUP-03`: <R_DA> and <R_DA>_E come from `IMP.bff` now, not from QuEst.

    Written *with* the swap, because what protected these before was a single
    assertion that a distance lay between 10 and 100 A — which every wrong
    answer short of a unit error also satisfies. The three properties below are
    the ones the adoption could plausibly have broken.

    `dRmp` is deliberately not tested against upstream: `av_pair_statistics`
    returns R_mp wrongly by its own admission, and QuEst keeps its own. See
    :func:`quest.core.av.dRmp`.
    """

    STRUCTURE = Path(__file__).parent / "148l.pdb"
    R0 = 52.0
    #: Both sites must be labelable under **both** backends, or this class errors
    #: in whichever environment falls back to LabelLib. E15/E90 -- the pair the
    #: dedup spec quotes -- does not qualify: LabelLib labels 43 of 148l chain E's
    #: 151 CB sites against IMP.bff's 151, and 15 is not among the 43. E55/E124
    #: is the widest-separated pair that is (64.0 A between mean positions), so
    #: it also exercises the R0-relative regime these metrics are for.
    SITES = (55, 124)

    @pytest.fixture(scope="class")
    def pair(self):
        pytest.importorskip("LabelLib", reason="LabelLib is not installed")
        from quest.core.av import AV
        from quest.core.structure import Structure

        structure = Structure(str(self.STRUCTURE), make_coarse=False)
        common = dict(atom_name="CB", chain_identifier="E", linker_length=21.5)
        return tuple(
            AV(structure, residue_seq_number=site, **common) for site in self.SITES
        )

    def test_the_two_distances_are_ordered_and_physical(self, pair) -> None:
        """<R_DA>_E < <R_DA> for any real pair: the Forster average is
        weighted towards the close configurations, which is the whole reason
        the two are not interchangeable."""
        from quest.core.av import RDAMean, RDAMeanE

        av1, av2 = pair
        r_da = RDAMean(av1, av2)
        r_e = RDAMeanE(av1, av2, self.R0)
        assert 20.0 < r_e < r_da < 120.0, (r_e, r_da)

    def test_sampling_is_reproducible(self, pair) -> None:
        """Upstream seeds explicitly where QuEst's parallel kernel did not, so
        a reported distance no longer changes between identical calls. This is
        a behaviour *change*, and pinning it is the point."""
        from quest.core.av import RDAMeanE

        av1, av2 = pair
        assert RDAMeanE(av1, av2, self.R0) == RDAMeanE(av1, av2, self.R0)
        assert RDAMeanE(av1, av2, self.R0, seed=1) != RDAMeanE(av1, av2, self.R0, seed=2)

    def test_it_converges_on_the_same_answer_the_old_kernel_gave(self, pair) -> None:
        """A plain unweighted Monte-Carlo mean over random pairs — what QuEst's
        deleted `_ran_dist` computed. The clouds are binary, so upstream's
        weighted mean must reduce to exactly this."""
        import numpy as np

        from quest.core.av import RDAMean

        av1, av2 = pair
        rng = np.random.default_rng(20260728)
        i1 = rng.integers(0, av1.points.shape[0], 200_000)
        i2 = rng.integers(0, av2.points.shape[0], 200_000)
        reference = float(
            np.linalg.norm(av1.points[i1] - av2.points[i2], axis=1).mean()
        )
        # Both sides are Monte-Carlo; 0.5 A is comfortably inside the spread of
        # either at these sample counts and well below the 8 % that separates
        # genuinely different metrics.
        assert RDAMean(av1, av2, 200_000) == pytest.approx(reference, abs=0.5)
