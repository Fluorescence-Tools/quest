"""Pin the numbers, not just the absence of exceptions.

A green suite does not tell you an accessible volume is right. The AV grid was
once transposed x<->z and still overlapped the truth by **83 %** of its voxels:
point count, bounding box and total volume were all correct, only the shape was
mirrored — so the contact volume landed on the wrong side of the protein and
donors came out unquenched, and nothing in the suite noticed.

`tests/baselines/*.json` is the record of what this tree computes. Regenerate it
with `tests/baselines/generate.py`, and only in a commit whose `okf/log.md`
entry accounts for **every** moved number
([okf/specs/programme-2026-07.md](../okf/specs/programme-2026-07.md),
decision 13). Regenerating to make this file pass, without that, is precisely
the failure this is meant to prevent.

The tolerances are tight on purpose. These are seeded, deterministic runs on
fixed structures; a number that moves at all has a cause, and the cause is what
belongs in the log.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from quest.core.av import available_av_backends

BASELINE_DIR = Path(__file__).parent / "baselines"

# Loaded by path rather than imported as `tests.baselines.generate`: `tests/` is
# deliberately not a package, and making it one changes pytest's import mode for
# every module in it.
_spec = importlib.util.spec_from_file_location(
    "quest_baseline_generate", BASELINE_DIR / "generate.py"
)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

#: Geometry is exact — voxel counts and grid edges are integers, and a
#: coordinate derived from them is a sum of exact binary fractions.
GEOMETRY_TOLERANCE = 1e-9

#: Monte-Carlo scalars are seeded and reproducible, so they should be exact too;
#: the tolerance covers float accumulation order only, which numba can change
#: between builds without any change in meaning.
SCALAR_TOLERANCE = 1e-6


def _baseline(backend: str) -> dict[str, Any]:
    path = BASELINE_DIR / f"{backend}.json"
    if not path.exists():
        pytest.skip(f"no baseline for {backend}; run tests/baselines/generate.py")
    return json.loads(path.read_text(encoding="utf8"))


BACKENDS = [b for b in ("labellib", "imp_bff") if b in available_av_backends()]

pytestmark = pytest.mark.skipif(not BACKENDS, reason="no accessible-volume backend")


@pytest.fixture(autouse=True)
def _isolate_artefacts(tmp_path, monkeypatch):
    monkeypatch.setenv("QUEST_JOBS_DIR", str(tmp_path / "jobs"))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("residue", gen.SITES)
class TestTheAccessibleVolumeHasNotMoved:
    def test_the_grid_and_the_cloud_agree_with_the_baseline(
        self, backend: str, residue: int
    ) -> None:
        expected = _baseline(backend)["accessible_volumes"][f"E{residue}"]
        measured = gen.measure_av(gen.STRUCTURE, "E", residue, backend)

        assert measured["ng"] == expected["ng"]
        assert measured["n_points"] == expected["n_points"]
        assert measured["occupied_voxels"] == expected["occupied_voxels"]

    def test_the_cloud_sits_where_it_did(self, backend: str, residue: int) -> None:
        """Position, separately from size.

        A cloud shifted by half a voxel has exactly the same point count, the
        same volume and the same bounding box. Only its centroid moves — which
        is the whole reason this assertion is not folded into the one above.
        """
        expected = _baseline(backend)["accessible_volumes"][f"E{residue}"]
        measured = gen.measure_av(gen.STRUCTURE, "E", residue, backend)

        assert measured["x0"] == pytest.approx(expected["x0"], abs=GEOMETRY_TOLERANCE)
        assert measured["points_centroid"] == pytest.approx(
            expected["points_centroid"], abs=GEOMETRY_TOLERANCE
        )
        assert measured["extent"] == pytest.approx(expected["extent"], abs=GEOMETRY_TOLERANCE)


@pytest.mark.parametrize("backend", BACKENDS)
class TestTheSimulationHasNotMoved:
    @pytest.mark.parametrize("residue", gen.SITES)
    def test_the_scalar_metrics_agree(self, backend: str, residue: int, monkeypatch) -> None:
        monkeypatch.setenv("QUEST_AV_BACKEND", backend)
        expected = _baseline(backend)["simulations"][f"E{residue}"]
        measured = gen.measure_simulation(gen._project(residue))

        for key, value in expected.items():
            if value is None:
                assert measured[key] is None, f"{key} gained a value"
                continue
            assert measured[key] == pytest.approx(value, rel=SCALAR_TOLERANCE), key

    def test_the_distance_metrics_agree(self, backend: str, monkeypatch) -> None:
        monkeypatch.setenv("QUEST_AV_BACKEND", backend)
        expected = _baseline(backend)["distances"]
        measured = gen.measure_distances("E", tuple(expected["pair"]), backend)

        assert measured["rda_mean"] == pytest.approx(expected["rda_mean"], rel=1e-4)
        assert measured["rda_mean_e"] == pytest.approx(expected["rda_mean_e"], rel=1e-4)
        assert measured["rmp"] == pytest.approx(expected["rmp"], abs=GEOMETRY_TOLERANCE)

    def test_the_fret_channel_agrees(self, backend: str, monkeypatch) -> None:
        monkeypatch.setenv("QUEST_AV_BACKEND", backend)
        expected = _baseline(backend)["fret"]
        measured = gen.measure_simulation(
            gen._project(gen.FRET_PAIR[0], acceptor=gen.FRET_PAIR[1])
        )

        assert measured["fret_efficiency"] == pytest.approx(
            expected["fret_efficiency"], rel=SCALAR_TOLERANCE
        )
        assert 0.0 < measured["fret_efficiency"] < 1.0


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("structure_name", ("148l.pdb", "mc4r_dimer.cif"))
def test_coarse_graining_has_not_moved(backend: str, structure_name: str) -> None:
    """Residue identity, recorded before it is fixed.

    `n_residue_ids_with_more_than_one_cb` is **534** on the dimer today: every
    residue id appears in both chains, and `residue_dict` is keyed on the id
    alone. That number is expected to go to 0 when residue identity becomes
    `(chain, res_id)`, and the CB centroid is expected to move with it — on the
    dimer only. If it moves on `148l.pdb`, which is effectively single-chain
    for this purpose, the change is wrong.
    """
    expected = _baseline(backend)["coarse_graining"][structure_name]
    path = gen.STRUCTURE if structure_name == "148l.pdb" else gen.MULTICHAIN
    measured = gen.measure_coarse_graining(path)

    assert measured["chains"] == expected["chains"]
    assert measured["n_atoms"] == expected["n_atoms"]
    assert measured["n_cb"] == expected["n_cb"]
    assert (
        measured["n_residue_ids_with_more_than_one_cb"]
        == expected["n_residue_ids_with_more_than_one_cb"]
    )
    assert measured["cb_centroid"] == pytest.approx(
        expected["cb_centroid"], abs=GEOMETRY_TOLERANCE
    )


def test_both_backends_are_baselined() -> None:
    """A missing baseline must not pass quietly as a skip nobody reads."""
    missing = [b for b in available_av_backends() if not (BASELINE_DIR / f"{b}.json").exists()]
    assert not missing, f"no baseline for available backend(s): {', '.join(missing)}"


def test_the_backends_actually_disagree() -> None:
    """Canary against a generation bug that made the two files identical.

    The first run of `generate.py` produced byte-identical simulation blocks for
    both backends, because `simulate_project` reads `QUEST_AV_BACKEND` from the
    environment and ignores the backend passed to `AV()`. The volumes differed
    by 7000 points while the decays did not, which is impossible.
    """
    if len(BACKENDS) < 2:
        pytest.skip("only one backend in this interpreter")
    labellib = _baseline("labellib")
    imp_bff = _baseline("imp_bff")

    assert (
        labellib["simulations"]["E96"]["quantum_yield_donor"]
        != imp_bff["simulations"]["E96"]["quantum_yield_donor"]
    ), "the two baselines were measured on the same backend"
