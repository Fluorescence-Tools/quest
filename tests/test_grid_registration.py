"""One grid, one index map — on both backends.

Three separate things convert between Angstrom and a voxel index, and they must
be the same conversion:

1. `_center_grid_indices` places the quencher and stickiness centres when a grid
   is *stamped*.
2. `density2points` / `_points_from_centred_grid` turn the density back into a
   point cloud.
3. `DonorDecay._sample_grid_at_trajectory` reads a stamped grid at the walk's
   positions.

Until 2026-07-28 they did not agree. (1) and (2) used the integer offset
`(ng - 1) // 2`; (3) and the trajectory kernels used the float corner
`(ng - 1) / 2`; and the LabelLib branch anchored `x0` on the **attachment atom**
rather than on a voxel node. Those differ whenever `ng` is even, which is the
normal case — 86 and 80 on the two backends at the reference site.

The consequence was not an exception. It was quenching rates written half a
voxel from where the walk read them, which is the same class of defect as the
transposed grid that overlapped the truth by 83 % of its voxels and silently
removed all quenching.

These tests check the *mechanism* rather than a quantum yield: a number moving
tells you something changed, and only registration tells you which way is right.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from quest.core.av import (
    AV,
    _center_grid_indices,
    available_av_backends,
    grid_center_index,
)

STRUCTURE = Path(__file__).parent / "148l.pdb"
RESOLUTION = 0.5
SITE = 96

BACKENDS = [b for b in ("labellib", "imp_bff") if b in available_av_backends()]

pytestmark = pytest.mark.skipif(not BACKENDS, reason="no accessible-volume backend")


@pytest.fixture(scope="module", params=BACKENDS)
def av(request):
    return AV(
        str(STRUCTURE),
        chain_identifier="E",
        residue_seq_number=SITE,
        atom_name="CB",
        simulation_grid_resolution=RESOLUTION,
        save_av=False,
        verbose=False,
        av_backend=request.param,
    )


def test_the_grid_edge_is_even_here(av) -> None:
    """The whole defect is invisible on an odd edge; assert we are testing it.

    For odd `ng` the integer and float offsets are equal, so a suite that only
    ever saw odd grids would pass with either convention.
    """
    assert av.ng % 2 == 0, (
        f"ng={av.ng} is odd, so this file proves nothing about the offset "
        "convention; pick a site or resolution that yields an even edge"
    )


def test_every_cloud_point_indexes_to_an_occupied_voxel(av) -> None:
    """The registration check, stated as a fact about the two arrays.

    `AV.points` and `AV.density` describe the same volume. Running a point
    through the *kernels'* index formula must therefore land on a voxel the
    density calls occupied — for every point, not most of them.

    Measured on 2026-07-28 with the LabelLib branch anchored on the attachment
    atom: **37478/37859 = 98.99 %**. The missing 1 % is the half-voxel.
    """
    points = np.asarray(av.points, dtype=float)
    x0 = np.asarray(av.x0, dtype=float)
    ng = av.ng
    density = np.asarray(av.density).reshape(ng, ng, ng)

    ix, iy, iz, _ = _center_grid_indices(
        points, x0, RESOLUTION, ng, np.zeros(len(points))
    )
    inside = (
        (ix >= 0) & (ix < ng) & (iy >= 0) & (iy < ng) & (iz >= 0) & (iz < ng)
    )
    assert inside.all(), f"{(~inside).sum()} points index outside the grid"

    occupied = density[ix, iy, iz] > 0
    assert occupied.all(), (
        f"{(~occupied).sum()} of {len(points)} cloud points index to an empty "
        "voxel: the point cloud and the grid the kernels stamp are misregistered"
    )


def test_the_anchor_sits_on_a_voxel_node(av) -> None:
    """`x0` is a voxel position, not an atom position.

    Both backends size the grid themselves, and the attachment atom lands on a
    node only when the edge length is odd.
    """
    ng = av.ng
    npm = grid_center_index(ng)
    x0 = np.asarray(av.x0, dtype=float)
    points = np.asarray(av.points, dtype=float)

    # Every cloud coordinate is an integer number of voxels from the anchor.
    offsets = (points - x0) / RESOLUTION
    assert np.allclose(offsets, np.round(offsets), atol=1e-9), (
        "cloud coordinates are not on the lattice `x0` defines"
    )
    assert 0 <= npm < ng


def test_the_trajectory_and_the_kernels_use_the_same_index_map() -> None:
    """(3) against (1), without running a simulation.

    `_sample_grid_at_trajectory` computes `floor((p - x0)/dg + centre)`, and
    `_center_grid_indices` computes `floor((p - x0)/dg) + centre`. Those agree
    only when `centre` is an integer in both — the float corner shifts one of
    them by exactly half a voxel on an even edge — **and** only when both round
    the same way. They did not: the kernel used `int()`, which truncates toward
    zero and therefore rounds *up* for a negative offset, while the sampler used
    `trunc()` on an already-positive value, which rounds down. Half the grid.
    """
    ng = 86
    dg = RESOLUTION
    x0 = np.array([1.25, -3.5, 10.0])
    rng = np.random.default_rng(0)
    probes = x0 + (rng.random((2000, 3)) - 0.5) * ng * dg

    npm = grid_center_index(ng)
    kernel_ix, kernel_iy, kernel_iz, _ = _center_grid_indices(
        probes, x0, dg, ng, np.zeros(len(probes))
    )
    sampled = np.floor((probes - x0) / dg + float(npm)).astype(np.int64)

    assert np.array_equal(sampled[:, 0], kernel_ix)
    assert np.array_equal(sampled[:, 1], kernel_iy)
    assert np.array_equal(sampled[:, 2], kernel_iz)

    # And the float convention does *not* agree, which is the bug.
    float_centre = (ng - 1) / 2.0
    float_sampled = np.floor((probes - x0) / dg + float_centre).astype(np.int64)
    assert not np.array_equal(float_sampled[:, 0], kernel_ix), (
        "the float corner agrees with the kernels here, so this test cannot "
        "detect the defect it exists for -- is ng odd?"
    )


def test_grid_center_index_is_the_integer_offset() -> None:
    assert grid_center_index(86) == 42
    assert grid_center_index(85) == 42
    # Odd and even differ from the float corner in exactly one case.
    assert grid_center_index(85) == (85 - 1) / 2
    assert grid_center_index(86) != (86 - 1) / 2
