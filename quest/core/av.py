import os.path
import json
import os
import warnings
from collections import OrderedDict

import numpy as np
from numba import njit, prange
# Qt-free: the accessible-volume widget that used to live here was unused, and
# it was the only reason this module imported a GUI toolkit.

from . import pdb as PDB
from .structure import Structure
from .io import write_xyz

try:
    import LabelLib as ll
    HAS_LABELLIB = True
except Exception:  # pragma: no cover - optional dependency
    HAS_LABELLIB = False

from .imp_av import (
    HAS_IMP_BFF,
    build_imp_accessible_volume,
)

#: Resolved on first use by :func:`_density2points_kernel`, never at import.
_UPSTREAM_DENSITY2POINTS = None


def _density2points_kernel():
    """`IMP.bff.av._kernels.density2points`, imported on first use.

    **Deliberately lazy, and two separate tests depend on it.** Importing
    `IMP.bff.av` pulls in `IMP` itself, and `import quest.core` must not do that:
    `test_importing_the_domain_does_not_import_qt` and `test_the_model_needs_no_qt`
    both probe the domain layer in a subprocess, and a module-scope import here
    made both fail. Not with an ImportError either -- with the dylib conflict in
    [references/known-issues.md](../../okf/references/known-issues.md), where a
    child of a process that has already loaded IMP resolves
    `libimp_kernel` against the *base* environment's `libboost_filesystem`. So a
    module-scope import would have taken a documented latent environment fault
    and made it a hard failure of the service layer.

    Deferring costs nothing: this is reached once per accessible volume, and the
    result is cached in a module global.

    Not guarded, once it is actually needed. imp-tricks is a declared dependency
    as of 2026-07-28 ([specs/imp-tricks-dedup.md](../../okf/specs/imp-tricks-dedup.md)),
    and a fallback could only mean keeping the second copy of this kernel that
    the import exists to delete. The message names the cause, because a bare
    ImportError deep inside a simulation names nothing.
    """
    global _UPSTREAM_DENSITY2POINTS
    if _UPSTREAM_DENSITY2POINTS is None:
        try:
            from IMP.bff.av._kernels import density2points
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "QuEst's accessible volume requires `IMP.bff.av` from imp-tricks, "
                "which is not importable. Install imp-tricks, or place a checkout "
                "at ~/dev/imp-tricks (or $IMP_TRICKS_SRC) so `quest.imp_tricks` "
                "can splice it in."
            ) from exc
        _UPSTREAM_DENSITY2POINTS = density2points
    return _UPSTREAM_DENSITY2POINTS


#: Accessible-volume backends, in the order ``"auto"`` prefers them.
#:
#: **IMP.bff is primary; LabelLib is the fallback for platforms without a
#: compiled IMP.bff — Windows above all.** Promoted 2026-07-27, once the two
#: defects that had kept it second were fixed: `build_imp_accessible_volume`
#: used to re-read the file from disk with `IMP.atom.read_pdb`, which silently
#: discarded coarse-graining and made `.cif` projects fail outright. It now
#: builds the IMP model from the atom array it is handed, so it cannot do
#: either ([specs/structure-io.md](okf/specs/structure-io.md)).
#:
#: The backends do not agree exactly, and this order decides which answer QuEst
#: gives: on 148l E36 they return 136 707 (IMP.bff) against 151 869 (LabelLib)
#: points — a 0.90 ratio, with centroids within 0.5 A. Same place, ~10 % less
#: of it. That residual is algorithmic, and it is the one number that moves
#: when a platform falls back to LabelLib, so contact fractions and quenching
#: rates are not bit-comparable across the two.
#:
#: The direction of travel is still to route through
#: `IMP.bff.av.compute.compute_av` rather than carry `imp_av.py` at all
#: ([specs/imp-tricks-dedup.md](okf/specs/imp-tricks-dedup.md)); this only
#: changes which of the two in-tree paths runs first.
AV_BACKENDS = ("imp_bff", "labellib")

#: Radius around the attachment atom inside which obstacles are ignored, so the
#: linker can leave the atom it is tied to.
#:
#: **One project-level value, 2.1 A**, replacing QuEst's 0.5 default and the
#: separate 2.0 clamp IMP's path got. 2.1 A is carbon's CHARMM radius -- the
#: attachment atom is a CB in every default workflow, and the sphere that lets
#: the linker escape it should be the size of that atom. It also sits just above
#: the cliff below which IMP's path search starts nowhere and returns an **empty
#: volume with no error**; imp.bff's own reference test uses 2.0 and expects zero
#: points at 1.0.
#:
#: **Only the IMP.bff backend honours it.** `calculate1R`/`calculate3R` accept a
#: `linkersphere` argument and never pass it on -- `ll.dyeDensityAV1` has no such
#: parameter, and the wrapper zeroes the attachment atom's own vdW radius
#: instead. Measured: raising the default from 0.5 to 2.1 moves LabelLib's
#: labelable-site count on 148l chain E by exactly nothing (43/151 before and
#: after), while IMP.bff goes 147 -> 151/151 and quenching sites 60 -> 67.
#:
#: That is also the correction to a claim made when IMP.bff was promoted: the
#: 43-vs-147 gap between the backends is **not** mostly this parameter, it is
#: how each treats the attachment site. LabelLib frees only the attachment atom,
#: so a CB hemmed in by CHARMM-sized neighbours has no way out; IMP ignores every
#: obstacle within the sphere.
DEFAULT_ALLOWED_SPHERE_RADIUS = 2.1

#: Smallest `allowed_sphere_radius` that IMP.bff's path search can start from.
#: Only reachable now by asking for less than the default explicitly; the clamp
#: warns rather than silently substituting, because a quietly changed parameter
#: is how the backends drifted apart in the first place. LabelLib has no such
#: cliff -- it zeroes the attachment atom's radius instead.
MIN_IMP_ALLOWED_SPHERE_RADIUS = 2.0


def _clamped_allowed_sphere_radius(value):
    """Keep IMP's path search above the cliff, and say so when it has to.

    The old code clamped silently. That is how a caller asking for 0.5 got 2.0
    and never learned the number it reasoned about was not the number that ran.
    """
    radius = float(value)
    if radius >= MIN_IMP_ALLOWED_SPHERE_RADIUS:
        return radius
    warnings.warn(
        f"allowed_sphere_radius={radius:g} A is below "
        f"{MIN_IMP_ALLOWED_SPHERE_RADIUS:g} A, where IMP.bff's path search "
        f"starts nowhere and returns an empty volume without reporting one. "
        f"Using {MIN_IMP_ALLOWED_SPHERE_RADIUS:g} A instead; the default is "
        f"{DEFAULT_ALLOWED_SPHERE_RADIUS:g} A.",
        RuntimeWarning,
        stacklevel=3,
    )
    return MIN_IMP_ALLOWED_SPHERE_RADIUS


def available_av_backends():
    """Return the accessible-volume backends usable in this interpreter."""
    available = []
    if HAS_LABELLIB:
        available.append("labellib")
    if HAS_IMP_BFF:
        available.append("imp_bff")
    return available


def resolve_av_backend(preference=None):
    """Pick the accessible-volume backend to use.

    Parameters
    ----------
    preference : str, optional
        ``"labellib"``, ``"imp_bff"``, or ``"auto"``. Defaults to the
        ``QUEST_AV_BACKEND`` environment variable, else ``"auto"``.

    Raises
    ------
    ValueError
        If the name is not a known backend.
    RuntimeError
        If the requested backend -- or, for ``"auto"``, any backend -- is
        unavailable. The message says which ones are, rather than failing later
        with an empty volume.
    """
    choice = str(preference or os.environ.get("QUEST_AV_BACKEND") or "auto").strip().lower()
    available = available_av_backends()

    if choice == "auto":
        for backend in AV_BACKENDS:
            if backend in available:
                return backend
        raise RuntimeError(
            "No accessible-volume backend is available. Install an IMP build "
            "providing IMP.bff (preferred), or LabelLib."
        )

    if choice not in AV_BACKENDS:
        raise ValueError(
            f"Unknown accessible-volume backend {choice!r}; expected one of "
            f"{', '.join(AV_BACKENDS)} or 'auto'."
        )
    if choice not in available:
        raise RuntimeError(
            f"Accessible-volume backend {choice!r} was requested but is not "
            f"available; usable backends here: {', '.join(available) or 'none'}."
        )
    return choice


# Numba-accelerated kernels (flattened from fps.py)
def _points_from_centred_grid(density, ng, dg, centre):
    """Point cloud of a cubic grid described by its *centre* rather than a corner.

    QuEst's own AV carries ``x0`` (the grid anchor, at the middle voxel);
    LabelLib reports a corner. One kernel, two conventions, one shift.

    The offset is the **integer** ``(ng - 1) // 2``, matching
    :func:`_center_grid_indices`. For LabelLib's odd grids the two are equal, so
    this changes nothing there; for an even edge length -- which is what IMP
    produces -- a float ``(ng - 1) / 2`` would place the point cloud half a
    voxel off the grid the quenching and stickiness kernels index into.

    Plain Python on purpose: it does arithmetic on three numbers and then calls
    the jitted kernel. Compiling *this* would only make it unable to see
    :func:`density2points` as anything but an untyped global.
    """
    origin = np.asarray(centre, dtype=np.float64) - ((ng - 1) // 2) * float(dg)
    return density2points(ng, ng, ng, dg, density, origin)


#: Default seed for the AV pair sampling. Fixed on purpose -- see
#: :func:`_sample_pair_distances`.
RDA_SAMPLING_SEED = 0

#: Resolved on first use by :func:`_pair_distance_kernels`, never at import.
_UPSTREAM_PAIR_KERNELS = None


def _pair_distance_kernels():
    """`random_distances` and `av_pair_statistics`, imported on first use.

    Lazy for the same reason as :func:`_density2points_kernel` -- importing
    `IMP.bff` reaches `IMP`, and `import quest.core` must not.
    """
    global _UPSTREAM_PAIR_KERNELS
    if _UPSTREAM_PAIR_KERNELS is None:
        try:
            from IMP.bff.av._kernels import random_distances
            from IMP.bff.distance_metrics import av_pair_statistics
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "QuEst's donor-acceptor distances require `IMP.bff` from "
                "imp-tricks, which is not importable. Install imp-tricks, or "
                "place a checkout at ~/dev/imp-tricks (or $IMP_TRICKS_SRC) so "
                "`quest.imp_tricks` can splice it in."
            ) from exc
        _UPSTREAM_PAIR_KERNELS = (random_distances, av_pair_statistics)
    return _UPSTREAM_PAIR_KERNELS


def _sample_pair_distances(av1, av2, n_samples, seed=RDA_SAMPLING_SEED):
    """Sampled distances and weights between two AV point clouds.

    `IMP.bff.av._kernels.random_distances` as of 2026-07-28 (`DUP-03`); QuEst's
    own `_ran_dist` is deleted. Two consequences, both measured rather than
    assumed on 148l E15/E90 (64 969 x 71 873 points, 50 000 samples):

    **The numbers agree and the timing does not.** ⟨R_DA⟩ 55.164 ± 0.037 A
    (QuEst) against 55.171 ± 0.032 (upstream), ⟨R_DA⟩_E 54.467 ± 0.022 against
    54.473 ± 0.029 — inside the sampling noise of either. But upstream costs
    **3.4 ms a call against 0.68 ms**, because QuEst's kernel was
    ``parallel=True`` where upstream's is sequential, and because upstream takes
    ``(N, 4)`` points so the cloud has to be widened with a weight column first
    (0.87 ms of the 3.4). Accepted, not overlooked: this runs **once per
    simulation** as a diagnostic, so a 2.7 ms regression is ~0.4 s across a
    151-site scan. Do not adopt this kernel anywhere it runs per frame.

    **Results are now reproducible, which they were not.** Upstream seeds
    explicitly; QuEst's parallel kernel drew from an unseeded per-thread RNG, so
    two calls on the same clouds gave different answers. A reported distance
    that changes between runs is a defect, so the fixed default stays and the
    seed is a parameter rather than a constant.

    The weight column is all ones: QuEst's density grids are binary, so the
    weighted mean upstream computes reduces to the plain one QuEst took.
    """
    random_distances, _ = _pair_distance_kernels()
    n_samples = int(n_samples)

    def _weighted(points):
        # Upstream's signature is (x, y, z, weight); QuEst's clouds are (x, y, z).
        wide = np.ones((points.shape[0], 4), dtype=np.float64)
        wide[:, :3] = points
        return wide

    sample = random_distances(
        _weighted(av1.points), _weighted(av2.points), n_samples, int(seed)
    )
    # Contiguous copies: `av_pair_statistics` does two `np.dot`s, and numba warns
    # (rightly) that a strided column view makes them slower than they need be.
    return (
        np.ascontiguousarray(sample[:, 0]),
        np.ascontiguousarray(sample[:, 1]),
    )


def RDAMeanE(av1, av2, R0: float = 52.0, nSamples: int = 50_000, seed: int = RDA_SAMPLING_SEED):
    """Forster-averaged donor-acceptor distance <R_DA>_E, in Angstrom.

    ``R0`` is in Angstrom, matching the AV point clouds. This is the average a
    FRET-efficiency measurement reports; it is not interchangeable with the
    plain mean distance :func:`RDAMean` or the mean-position distance
    :func:`dRmp`.
    """
    distances, weights = _sample_pair_distances(av1, av2, nSamples, seed)
    _, av_pair_statistics = _pair_distance_kernels()
    # Index 2 is R_E. Index 1 is *not* R_mp -- see :func:`dRmp`.
    return float(av_pair_statistics(distances, weights, float(R0))[2])


@njit(cache=True, parallel=True)
def _fret_rate_trace(trajectory, acceptor_points, R0, tau0, r_min2):
    n_frames = trajectory.shape[0]
    n_acceptor = acceptor_points.shape[0]
    rates = np.empty(n_frames, dtype=np.float64)
    R0_6 = R0 ** 6
    inv_tau0 = 1.0 / tau0
    for frame in prange(n_frames):
        px = trajectory[frame, 0]
        py = trajectory[frame, 1]
        pz = trajectory[frame, 2]
        total = 0.0
        for j in range(n_acceptor):
            dx = px - acceptor_points[j, 0]
            dy = py - acceptor_points[j, 1]
            dz = pz - acceptor_points[j, 2]
            r2 = dx * dx + dy * dy + dz * dz
            if r2 < r_min2:
                r2 = r_min2
            total += R0_6 / (r2 * r2 * r2)
        rates[frame] = inv_tau0 * total / n_acceptor
    return rates


@njit(cache=True, parallel=True)
def _fret_rate_pair_trace(donor, acceptor, R0, tau0, r_min2):
    n_frames = donor.shape[0]
    rates = np.empty(n_frames, dtype=np.float64)
    R0_6 = R0 ** 6
    inv_tau0 = 1.0 / tau0
    for frame in prange(n_frames):
        dx = donor[frame, 0] - acceptor[frame, 0]
        dy = donor[frame, 1] - acceptor[frame, 1]
        dz = donor[frame, 2] - acceptor[frame, 2]
        r2 = dx * dx + dy * dy + dz * dz
        if r2 < r_min2:
            r2 = r_min2
        rates[frame] = inv_tau0 * R0_6 / (r2 * r2 * r2)
    return rates


def fret_rate_pair_trace(
    donor_trajectory,
    acceptor_trajectory,
    R0: float,
    tau0: float,
    r_min: float = 7.0,
):
    """Per-frame FRET rate (1/ns) from *two* trajectories.

    Both dyes are resolved in time::

        k_FRET(t) = (1/tau0) * (R0 / |r_D(t) - r_A(t)|)^6

    This is the general case. :func:`fret_rate_trace` averages the acceptor over
    its accessible volume instead, which is the *fast-acceptor limit* — correct
    only when the acceptor explores its volume quickly compared with the donor's
    excited-state lifetime.

    The two do not agree, and the reason is **occupancy**, not the averaging
    order. The cloud is weighted uniformly: every accessible voxel counts the
    same. A trajectory is weighted by where the dye actually spends its time,
    and a sticky dye dwells near the protein surface rather than out in the
    free volume. Which way that moves the efficiency depends on the geometry —
    it is not a fixed bias. On 148l E15→E90 with a sticky acceptor
    (D = 2.5 Å²/ns, slow radius 11 Å) it raised E from 0.73 to 0.78, because
    the sticky region there lies toward the donor.

    The two walks are independent, so pairing frame *i* with frame *i* samples
    the joint distribution correctly **provided both were simulated on the same
    time base** — same ``t_step`` and same number of frames. The caller is
    responsible for that; a mismatch raises.

    Parameters
    ----------
    donor_trajectory, acceptor_trajectory : array_like, shape (n_frames, 3)
        Dye centre positions in Angstrom, frame-aligned.
    R0 : float
        Forster radius in Angstrom.
    tau0 : float
        Unquenched donor lifetime in nanoseconds.
    r_min : float
        Closest approach of the two dye centres, in Angstrom. Two dye spheres
        cannot interpenetrate, and ``(R0/r)^6`` diverges if they are allowed to.

    Returns
    -------
    numpy.ndarray
        Rates in 1/ns, one per frame.
    """
    donor = np.ascontiguousarray(donor_trajectory, dtype=np.float64).reshape((-1, 3))
    acceptor = np.ascontiguousarray(acceptor_trajectory, dtype=np.float64).reshape((-1, 3))
    if donor.shape[0] == 0 or acceptor.shape[0] == 0:
        raise ValueError("Both trajectories must have frames to pair.")
    if tau0 <= 0.0:
        raise ValueError("tau0 must be positive to derive a FRET rate.")
    if donor.shape[0] != acceptor.shape[0]:
        # Truncating would silently change the sampled time window; a caller
        # that meant to do that should say so.
        raise ValueError(
            "Donor and acceptor trajectories must have the same number of "
            f"frames to pair ({donor.shape[0]} vs {acceptor.shape[0]}); "
            "simulate both dyes with the same t_max and t_step."
        )
    r_min = max(float(r_min), 1e-6)
    return _fret_rate_pair_trace(donor, acceptor, float(R0), float(tau0), r_min * r_min)


def fret_rate_trace(
    trajectory,
    acceptor_points,
    R0: float,
    tau0: float,
    r_min: float = 7.0,
    max_acceptor_points: int = 512,
):
    """Per-frame FRET rate (1/ns) along a donor trajectory.

    For every donor position the transfer rate is averaged over the acceptor's
    accessible volume::

        k_FRET(t) = (1/tau0) * < (R0 / |r_D(t) - r_A|)^6 >_A

    This resolves the donor in time -- the point of simulating its diffusion at
    all -- while treating the acceptor as sampling its own volume quickly
    compared with the donor's excited-state lifetime. That is the fast-acceptor
    limit; it is not valid if the acceptor is immobilised.

    Parameters
    ----------
    trajectory : array_like, shape (n_frames, 3)
        Donor positions in Angstrom.
    acceptor_points : array_like, shape (n, 3)
        Acceptor accessible-volume cloud in Angstrom.
    R0 : float
        Forster radius in Angstrom.
    tau0 : float
        Unquenched donor lifetime in nanoseconds.
    r_min : float
        Closest approach of the two dye centres, in Angstrom. Both clouds are
        centre positions and can overlap in space, where ``(R0/r)^6`` would
        otherwise diverge; two dye spheres cannot actually interpenetrate, so
        the distance is floored at the sum of their radii.
    max_acceptor_points : int
        Cap on the acceptor points averaged over, taken by an even stride. The
        average converges long before the full cloud, and the cost is
        ``n_frames * n_acceptor``.

    Returns
    -------
    numpy.ndarray
        Rates in 1/ns, one per frame.
    """
    trajectory = np.ascontiguousarray(trajectory, dtype=np.float64).reshape((-1, 3))
    points = np.ascontiguousarray(acceptor_points, dtype=np.float64).reshape((-1, 3))
    if points.shape[0] == 0:
        raise ValueError("Acceptor accessible volume is empty; cannot compute FRET rates.")
    if tau0 <= 0.0:
        raise ValueError("tau0 must be positive to derive a FRET rate.")
    if points.shape[0] > max_acceptor_points:
        stride = int(np.ceil(points.shape[0] / float(max_acceptor_points)))
        points = np.ascontiguousarray(points[::stride])
    r_min = max(float(r_min), 1e-6)
    return _fret_rate_trace(trajectory, points, float(R0), float(tau0), r_min * r_min)


def RDAMean(av1, av2, nSamples: int = 50_000, seed: int = RDA_SAMPLING_SEED):
    """Mean donor-acceptor distance <R_DA>, in Angstrom."""
    distances, weights = _sample_pair_distances(av1, av2, nSamples, seed)
    _, av_pair_statistics = _pair_distance_kernels()
    return float(av_pair_statistics(distances, weights, 52.0)[0])


def dRmp(av1, av2):
    """Distance between the two clouds' mean positions, in Angstrom.

    **Stays in QuEst deliberately, and the arrow points upstream here.**
    `av_pair_statistics` returns this as its second value and it is wrong: its
    own docstring concedes "same as ``r_da_mean`` here", i.e. it returns the mean
    *pair distance* again. R_mp is |<r_D> - <r_A>|, which cannot be recovered
    from a sample of pair distances at all — the sampling destroys exactly the
    information it needs. Measured on 148l E15/E90: 47.65 A here against
    51.53 A from upstream's field, 8 % apart, and the smaller number is the
    correct one.

    So index 1 of `av_pair_statistics` must never be read. QuEst reads 0 and 2.
    """
    return float(np.sqrt(((av1.Rmp - av2.Rmp) ** 2).sum()))


@njit(cache=True)
def _center_grid_indices(rs, r0, dg, ng, radius):
    """Voxel coordinates and integer radii of the sphere centres."""
    n = rs.shape[0]
    ix0 = np.empty(n, dtype=np.int64)
    iy0 = np.empty(n, dtype=np.int64)
    iz0 = np.empty(n, dtype=np.int64)
    radius_idx = np.empty(n, dtype=np.int64)
    offset = (ng - 1) // 2
    for i in range(n):
        ix0[i] = int((rs[i, 0] - r0[0]) / dg) + offset
        iy0[i] = int((rs[i, 1] - r0[1]) / dg) + offset
        iz0[i] = int((rs[i, 2] - r0[2]) / dg) + offset
        radius_idx[i] = int(radius[i] / dg)
    return ix0, iy0, iz0, radius_idx


# The three grid kernels below only visit the voxels inside each centre's
# bounding box instead of testing every voxel against every centre. The outer
# loop stays over x-slabs so the work is parallel and free of write races.
@njit(cache=True, parallel=True)
def _subav(density, ng, slow_radius, rs, r0, dg):
    subav_density = density.copy()
    n_slow_center = rs.shape[0]
    ix0s, iy0s, iz0s, radius_idxs = _center_grid_indices(rs, r0, dg, ng, slow_radius)

    for ix in prange(ng):
        is_slow = np.zeros((ng, ng), dtype=np.uint8)
        for isa in range(n_slow_center):
            r_idx = radius_idxs[isa]
            dx = ix - ix0s[isa]
            remaining = r_idx * r_idx - dx * dx
            if remaining <= 0:
                continue
            iy0 = iy0s[isa]
            iz0 = iz0s[isa]
            y_lo = max(0, iy0 - r_idx)
            y_hi = min(ng - 1, iy0 + r_idx)
            for iy in range(y_lo, y_hi + 1):
                dy = iy - iy0
                span2 = remaining - dy * dy
                if span2 <= 0:
                    continue
                span = int(np.sqrt(span2))
                # The membership test is strict (d^2 < r^2), so drop the
                # boundary voxel when span2 is a perfect square.
                if span * span >= span2:
                    span -= 1
                if span < 0:
                    continue
                z_lo = max(0, iz0 - span)
                z_hi = min(ng - 1, iz0 + span)
                for iz in range(z_lo, z_hi + 1):
                    is_slow[iy, iz] = 1
        for iy in range(ng):
            for iz in range(ng):
                if is_slow[iy, iz] == 0:
                    subav_density[ix, iy, iz] = 0
    return subav_density


def subav(density, ng, dg, slow_radius, rs, r0):
    return _subav(
        np.asarray(density, dtype=np.uint8),
        int(ng),
        np.asarray(slow_radius, dtype=np.float64),
        np.asarray(rs, dtype=np.float64),
        np.asarray(r0, dtype=np.float64),
        float(dg),
    )


@njit(cache=True, parallel=True)
def _slow_factor_grid(density, ng, slow_radius, rs, r0, dg, slow_fact):
    factors = np.ones(density.shape, dtype=np.float64)
    n_slow_center = rs.shape[0]
    ix0s, iy0s, iz0s, radius_idxs = _center_grid_indices(rs, r0, dg, ng, slow_radius)

    for ix in prange(ng):
        slab = np.ones((ng, ng), dtype=np.float64)
        touched = False
        for isa in range(n_slow_center):
            r_idx = radius_idxs[isa]
            dx = ix - ix0s[isa]
            remaining = r_idx * r_idx - dx * dx
            if remaining <= 0:
                continue
            factor = slow_fact[isa]
            iy0 = iy0s[isa]
            iz0 = iz0s[isa]
            y_lo = max(0, iy0 - r_idx)
            y_hi = min(ng - 1, iy0 + r_idx)
            for iy in range(y_lo, y_hi + 1):
                dy = iy - iy0
                span2 = remaining - dy * dy
                if span2 <= 0:
                    continue
                span = int(np.sqrt(span2))
                # The membership test is strict (d^2 < r^2), so drop the
                # boundary voxel when span2 is a perfect square.
                if span * span >= span2:
                    span -= 1
                if span < 0:
                    continue
                z_lo = max(0, iz0 - span)
                z_hi = min(ng - 1, iz0 + span)
                for iz in range(z_lo, z_hi + 1):
                    slab[iy, iz] *= factor
                    touched = True
        if touched:
            for iy in range(ng):
                for iz in range(ng):
                    if density[ix, iy, iz] != 0:
                        factors[ix, iy, iz] = slab[iy, iz]
    return factors


@njit(cache=True, parallel=True)
def _additive_factor_grid(density, ng, radius, rs, r0, dg, values):
    factors = np.zeros(density.shape, dtype=np.float64)
    n_center = rs.shape[0]
    ix0s, iy0s, iz0s, radius_idxs = _center_grid_indices(rs, r0, dg, ng, radius)

    for ix in prange(ng):
        slab = np.zeros((ng, ng), dtype=np.float64)
        touched = False
        for isa in range(n_center):
            value = values[isa]
            if value == 0.0:
                continue
            r_idx = radius_idxs[isa]
            dx = ix - ix0s[isa]
            remaining = r_idx * r_idx - dx * dx
            if remaining <= 0:
                continue
            iy0 = iy0s[isa]
            iz0 = iz0s[isa]
            y_lo = max(0, iy0 - r_idx)
            y_hi = min(ng - 1, iy0 + r_idx)
            for iy in range(y_lo, y_hi + 1):
                dy = iy - iy0
                span2 = remaining - dy * dy
                if span2 <= 0:
                    continue
                span = int(np.sqrt(span2))
                # The membership test is strict (d^2 < r^2), so drop the
                # boundary voxel when span2 is a perfect square.
                if span * span >= span2:
                    span -= 1
                if span < 0:
                    continue
                z_lo = max(0, iz0 - span)
                z_hi = min(ng - 1, iz0 + span)
                for iz in range(z_lo, z_hi + 1):
                    slab[iy, iz] += value
                    touched = True
        if touched:
            for iy in range(ng):
                for iz in range(ng):
                    if density[ix, iy, iz] != 0:
                        factors[ix, iy, iz] = slab[iy, iz]
    return factors


def slow_factor_grid(density, ng, dg, slow_radius, rs, r0, slow_fact):
    return _slow_factor_grid(
        np.asarray(density, dtype=np.uint8),
        int(ng),
        np.asarray(slow_radius, dtype=np.float64),
        np.asarray(rs, dtype=np.float64),
        np.asarray(r0, dtype=np.float64),
        float(dg),
        np.asarray(slow_fact, dtype=np.float64),
    )


def additive_factor_grid(density, ng, dg, radius, rs, r0, values):
    return _additive_factor_grid(
        np.asarray(density, dtype=np.uint8),
        int(ng),
        np.asarray(radius, dtype=np.float64),
        np.asarray(rs, dtype=np.float64),
        np.asarray(r0, dtype=np.float64),
        float(dg),
        np.asarray(values, dtype=np.float64),
    )


# Sequential on purpose: the random walk is a strict chain of dependent steps,
# so parallel=True buys no speed-up and splits the random draws across threads,
# which makes `random_seed` non-reproducible.
#
# ``nogil`` is what makes running *whole* trajectories concurrently worthwhile:
# without it the callers' thread pools serialise on the GIL. Numba's RNG state
# is thread-local, so each worker still reproduces its own ``random_seed``.
@njit(cache=True, nogil=True)
def _simulate_traj(d, ds, dg, t_max, t_step, D, slow_fact, random_seed):
    if random_seed >= 0:
        np.random.seed(random_seed)
    n_samples = int(t_max / t_step)
    ng = d.shape[0]
    sigma = np.sqrt(2 * D * 3 * t_step) / dg
    pos = np.zeros((n_samples, 3), dtype=np.float64)
    accepted = np.zeros(n_samples, dtype=np.uint8)
    r = np.random.normal(0.0, sigma, size=(n_samples, 3))
    slow_fact_sqrt = np.sqrt(slow_fact)

    # pick a starting point in AV
    found = False
    for _ in range(1000):
        x_idx = np.random.randint(0, ng)
        y_idx = np.random.randint(0, ng)
        z_idx = np.random.randint(0, ng)
        if d[x_idx, y_idx, z_idx] > 0:
            pos[0, 0] = x_idx
            pos[0, 1] = y_idx
            pos[0, 2] = z_idx
            found = True
            break
    if not found:
        return pos, accepted, 0, 0

    n_accepted = 1
    n_rejected = 0
    i_accepted = 0
    accepted[0] = 1
    for i in range(1, n_samples):
        x_idx = int(pos[i_accepted, 0])
        y_idx = int(pos[i_accepted, 1])
        z_idx = int(pos[i_accepted, 2])
        if ds[x_idx, y_idx, z_idx] > 0:
            r[i, 0] *= slow_fact_sqrt
            r[i, 1] *= slow_fact_sqrt
            r[i, 2] *= slow_fact_sqrt

        pos[i, 0] = pos[i_accepted, 0] + r[i, 0]
        pos[i, 1] = pos[i_accepted, 1] + r[i, 1]
        pos[i, 2] = pos[i_accepted, 2] + r[i, 2]

        x_idx = int(pos[i, 0])
        y_idx = int(pos[i, 1])
        z_idx = int(pos[i, 2])
        if 0 <= x_idx < ng and 0 <= y_idx < ng and 0 <= z_idx < ng and d[x_idx, y_idx, z_idx] > 0:
            i_accepted = i
            n_accepted += 1
            accepted[i] = 1
        else:
            pos[i, 0] = pos[i_accepted, 0]
            pos[i, 1] = pos[i_accepted, 1]
            pos[i, 2] = pos[i_accepted, 2]
            n_rejected += 1
            accepted[i] = 0
    return (pos - (ng - 1) / 2.0) * dg, accepted, n_accepted, n_rejected


# Sequential on purpose, and ``nogil`` for concurrency: see `_simulate_traj`.
@njit(cache=True, nogil=True)
def _simulate_traj_grid(d, slow_factor, dg, t_max, t_step, D, random_seed):
    if random_seed >= 0:
        np.random.seed(random_seed)
    n_samples = int(t_max / t_step)
    ng = d.shape[0]
    sigma = np.sqrt(2 * D * 3 * t_step) / dg
    pos = np.zeros((n_samples, 3), dtype=np.float64)
    accepted = np.zeros(n_samples, dtype=np.uint8)
    r = np.random.normal(0.0, sigma, size=(n_samples, 3))

    found = False
    for _ in range(1000):
        x_idx = np.random.randint(0, ng)
        y_idx = np.random.randint(0, ng)
        z_idx = np.random.randint(0, ng)
        if d[x_idx, y_idx, z_idx] > 0:
            pos[0, 0] = x_idx
            pos[0, 1] = y_idx
            pos[0, 2] = z_idx
            found = True
            break
    if not found:
        return pos, accepted, 0, 0

    n_accepted = 1
    n_rejected = 0
    i_accepted = 0
    accepted[0] = 1
    for i in range(1, n_samples):
        x_idx = int(pos[i_accepted, 0])
        y_idx = int(pos[i_accepted, 1])
        z_idx = int(pos[i_accepted, 2])
        factor = slow_factor[x_idx, y_idx, z_idx]
        if factor < 1.0:
            factor_sqrt = np.sqrt(factor)
            r[i, 0] *= factor_sqrt
            r[i, 1] *= factor_sqrt
            r[i, 2] *= factor_sqrt

        pos[i, 0] = pos[i_accepted, 0] + r[i, 0]
        pos[i, 1] = pos[i_accepted, 1] + r[i, 1]
        pos[i, 2] = pos[i_accepted, 2] + r[i, 2]

        x_idx = int(pos[i, 0])
        y_idx = int(pos[i, 1])
        z_idx = int(pos[i, 2])
        if 0 <= x_idx < ng and 0 <= y_idx < ng and 0 <= z_idx < ng and d[x_idx, y_idx, z_idx] > 0:
            i_accepted = i
            n_accepted += 1
            accepted[i] = 1
        else:
            pos[i, 0] = pos[i_accepted, 0]
            pos[i, 1] = pos[i_accepted, 1]
            pos[i, 2] = pos[i_accepted, 2]
            n_rejected += 1
            accepted[i] = 0
    return (pos - (ng - 1) / 2.0) * dg, accepted, n_accepted, n_rejected


def simulate_traj(d, ds, dg, t_max, t_step, D, slow_fact, random_seed=None):
    seed = -1 if random_seed is None else int(random_seed)
    slow_fact_array = np.asarray(slow_fact)
    if slow_fact_array.ndim == 3:
        return _simulate_traj_grid(
            np.asarray(d, dtype=np.uint8),
            np.asarray(slow_fact, dtype=np.float64),
            float(dg),
            float(t_max),
            float(t_step),
            float(D),
            seed,
        )
    return _simulate_traj(
        np.asarray(d, dtype=np.uint8),
        np.asarray(ds, dtype=np.uint8),
        float(dg),
        float(t_max),
        float(t_step),
        float(D),
        float(slow_fact),
        seed,
    )


def spherePoints(nSphere: int):
    inc = np.pi * (3.0 - np.sqrt(5.0))
    offset = 2.0 / float(nSphere)
    k_vals = np.arange(nSphere, dtype=np.float32)
    y = k_vals * offset - 1.0 + (offset / 2.0)
    rd = np.sqrt(1.0 - y * y)
    phi = k_vals * inc
    x = np.cos(phi) * rd
    z = np.sin(phi) * rd
    return np.stack([x, y, z], axis=1).astype(np.float32)


@njit(cache=True)
def _asa(r, vdw, probe_atom_indices, sphere_points, probe, radius):
    nSphere = sphere_points.shape[0]
    n_probe_atoms = probe_atom_indices.shape[0]
    n_atoms = r.shape[0]
    if n_probe_atoms == 0 or nSphere == 0 or n_atoms == 0:
        return np.zeros(n_probe_atoms, dtype=np.float32)

    neighbor_indices = np.zeros(n_atoms, dtype=np.uint32)
    probe_atom_asa = np.zeros(n_probe_atoms, dtype=np.float32)
    c = 4.0 * np.pi / float(nSphere)
    for i in range(n_probe_atoms):
        probe_atom_index = probe_atom_indices[i]
        n_neighbor = 0
        aX, aY, aZ = r[probe_atom_index]
        for atom_index in range(n_atoms):
            if probe_atom_index != atom_index:
                bX, bY, bZ = r[atom_index]
                dist2 = (aX - bX) ** 2 + (aY - bY) ** 2 + (aZ - bZ) ** 2
                if dist2 < 2 * (vdw[probe_atom_index] + probe):
                    neighbor_indices[n_neighbor] = atom_index
                    n_neighbor += 1

        n_accessible_point = 0
        for j in range(nSphere):
            is_accessible = True
            aX = sphere_points[j, 0] * radius + r[probe_atom_index, 0]
            aY = sphere_points[j, 1] * radius + r[probe_atom_index, 1]
            aZ = sphere_points[j, 2] * radius + r[probe_atom_index, 2]
            for k in range(n_neighbor):
                bX, bY, bZ = r[neighbor_indices[k]]
                dist2 = (aX - bX) ** 2 + (aY - bY) ** 2 + (aZ - bZ) ** 2
                if dist2 < (radius + probe) * (radius + probe):
                    is_accessible = False
                    break
            if is_accessible:
                n_accessible_point += 1
        probe_atom_asa[i] = c * n_accessible_point * vdw[probe_atom_index] ** 2
    return probe_atom_asa


def asa(r, vdw, probe_atom_indices, sphere_points, probe=1.0, radius=2.5):
    r = np.asarray(r, dtype=np.float32)
    vdw = np.asarray(vdw, dtype=np.float32)
    probe_atom_indices = np.asarray(probe_atom_indices, dtype=np.uint32)
    if sphere_points is None or np.asarray(sphere_points).ndim != 2:
        sphere_points = spherePoints(590)
    sphere_points = np.asarray(sphere_points, dtype=np.float32)
    return _asa(r, vdw, probe_atom_indices, sphere_points, float(probe), float(radius))


def density2points(nx, ny, nz, dg, density, r0):
    """Occupied voxels of a density grid, as an (n, 3) cloud in Angstrom.

    *r0* is the centre of voxel ``(0, 0, 0)``. A grid described by its *centre*
    is the same thing with the origin shifted by ``-(n - 1) // 2 * dg`` per axis;
    :func:`_points_from_centred_grid` does that, so this stays one kernel.

    **The loop itself is `IMP.bff.av._kernels.density2points`** as of 2026-07-28
    (`DUP-02`). QuEst carried its own for years; upstream's is a strict superset,
    keeping the per-voxel weight this signature discards, and the two select
    exactly the same voxels. Verified on 5x7x9, 8x8x8 and 11x11x11 grids: the
    point *sets* are equal.

    What differs is the **enumeration order** — upstream runs ``ix`` outermost,
    QuEst's ran ``iz`` outermost — so the rows come back permuted. That is safe
    only because nothing reads ``AV.points`` positionally: the walk runs on the
    density grid, :func:`RDAMean` samples indices at random, :meth:`AV.Rmp`
    averages, and the acceptor cloud is averaged over in full. The one visible
    effect is the row order of a saved ``.xyz``.

    See [imp-tricks-dedup](../../okf/specs/imp-tricks-dedup.md).
    """
    # Keep whatever memory layout the caller has: the Fortran-ordered grid
    # LabelLib produces is passed straight through rather than copied to C order,
    # which would add a pass over every voxel.
    n, points = _density2points_kernel()(
        int(nx), int(ny), int(nz),
        float(dg),
        np.asarray(density, dtype=np.float64),
        np.asarray(r0, dtype=np.float64),
    )
    # Upstream returns an oversized (nx*ny*nz, 4) buffer of (x, y, z, weight)
    # with only the first *n* rows valid. QuEst's grids are binary, so the
    # weight column carries nothing here; drop it and the slack.
    return points[:n, :3]


def calculate1R(l: float,
                w: float,
                r: float,
                atom_i: int,
                x: np.array,
                y: np.array,
                z: np.array,
                vdw: np.array,
                linkersphere: float = DEFAULT_ALLOWED_SPHERE_RADIUS,
                linknodes: int = 3,
                vdwRMax: float = 1.8,
                dg: float = 0.5,
                verbose: bool = False
                ):
    """
    :param l: linker length
    :param w: linker width
    :param r: dye-radius
    :param atom_i: attachment-atom index
    :param x: Cartesian coordinates of atoms (x)
    :param y: Cartesian coordinates of atoms (y)
    :param z: Cartesian coordinates of atoms (z)
    :param vdw: Van der Waals radii (same length as number of atoms)
    :param linkersphere: Initial linker-sphere to start search of allowed dye positions (not used anymore)
    :param linknodes: By default 3 (not used anymore)
    :param vdwRMax: Maximal Van der Waals radius (not used anymore)
    :param dg: Resolution of accessible volume in Angstrom
    :param verbose: If true informative output is printed on std-out
    :return:
    """
    if verbose:
        print("AV: calculate1R")
    n_atoms = len(vdw)
    x0, y0, z0 = x[atom_i], y[atom_i], z[atom_i]
    r0 = np.array([x0, y0, z0])

    vdw_copy = np.copy(vdw)
    vdw_copy[atom_i] = 0.0
    atoms = np.vstack([x, y, z, vdw_copy])
    source = r0
    av1 = ll.dyeDensityAV1(
        atoms,
        source,
        l, w, r, dg
    )
    nx, ny, nz = av1.shape
    # LabelLib hands back a plain Python list of nx*ny*nz floats. ``fromiter``
    # with an explicit count preallocates and skips the type probing np.array
    # does, which is worth ~5 ms on a typical 92^3 grid.
    grid = av1.grid
    g = np.fromiter(grid, dtype=np.float64, count=len(grid)).reshape([nx, ny, nz], order='F')
    if verbose:
        print(nx, ny, nz, dg, dg, dg, x0, y0, z0)
    ox = x0 - (nx - 1) / 2.0 * dg
    oy = y0 - (ny - 1) / 2.0 * dg
    oz = z0 - (nz - 1) / 2.0 * dg
    points = density2points(nx, ny, nz, dg, g, (ox, oy, oz))
    g[g < 0] = 0
    g[g > 0] = 1
    density = g.astype(np.uint8)
    ng = nx

    if verbose:
        print("Number of atoms: %i" % n_atoms)
        print("Attachment atom coordinates: %s" % r0)

    return points, density, ng, r0


class AV(object):

    def __init__(self, structure, residue_seq_number=None, atom_name=None, chain_identifier=None,
                 linker_length=20.0, linker_width=0.5, radius1=5.0,
                 radius2=4.5, radius3=3.5, simulation_grid_resolution=0.5, save_av=False, output_file='out',
                 attachment_atom_index=None, verbose=False,
                 allowed_sphere_radius=DEFAULT_ALLOWED_SPHERE_RADIUS,
                 simulation_type='AV1', position_name=None, residue_name=None,
                 av_backend=None):

        """
        Parameters
        ----------
        :param structure: Structure of str
            either string with path of pdb-file or pdb object
        :param residue_seq_number:
        :param atom_name:
        :param chain_identifier:
        :param linker_length:
        :param linker_width:
        :param radius1:
        :param radius2:
        :param radius3:
        :param simulation_grid_resolution:
        :param save_av:
        :param output_file:
        :param attachment_atom_index:
        :param verbose:
        :param allowed_sphere_radius:
        :param simulation_type:
        :param position_name:
        :param residue_name:
        :raise ValueError:

        Examples
        --------
        Calculating accessible volume using provided pdb-file
        >>>         >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
        >>> residue_number = 18
        >>> atom_name = 'CB'
        >>> attachment_atom = 1
        >>> av = lib.fps.AV(pdb_filename, attachment_atom=1, verbose=True)

        Calculating accessible volume using provided Structure object
        >>>         >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
        >>> structure = lib.Structure(pdb_filename)
        >>> av = lib.fps.AV(structure, attachment_atom=1, verbose=True)
        Calculating accessible volume
        -----------------------------
        Loading PDB
        Calculating initial-AV
        Linker-length  : 20.00
        Linker-width   : 0.50
        Linker-radius  : 5.00
        Attachment-atom: 1
        AV-resolution  : 0.50
        AV: calculate1R
        Number of atoms: 2647
        Attachment atom: [ 33.28   58.678  40.397]
        Points in AV: 111911
        Points in total-AV: 111911

        Using residue_seq_number and atom_name to calculate accessible volume, this also works without
        chain_identifier. However, only if a single-chain is present.
        >>>         >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
        >>> structure = lib.Structure(pdb_filename)
        >>> av = lib.fps.AV(structure, residue_seq_number=11, atom_name='CB', verbose=True)

        If save_av is True the calculated accessible volume is save to disk. The filename of the calculated
        accessible volume is determined by output_file
        >>>         >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
        >>> structure = lib.Structure(pdb_filename)
        >>> av = lib.fps.AV(structure, residue_seq_number=11, atom_name='CB', verbose=True, save_av=True, output_file='test')
        Calculating accessible volume
        -----------------------------
        Loading PDB
        Calculating initial-AV
        Linker-length  : 20.00
        Linker-width   : 0.50
        Linker-radius  : 5.00
        Attachment-atom: 174
        AV-resolution  : 0.50
        AV: calculate1R
        Number of atoms: 2647
        Attachment atom: [ 41.606  44.953  36.625]
        Points in AV: 22212
        Points in total-AV: 22212

        write_xyz
        ---------
        Filename: test.xyz
        -------------------

        """
        self.verbose = verbose
        self.points_slow = None
        self.position_name = position_name
        self.residue_name = residue_name
        if verbose:
            print("")
            print("Calculating accessible volume")
            print("-----------------------------")
        if isinstance(structure, Structure):
            atoms = structure.atoms
        else:
            if verbose:
                print("Loading PDB")
            atoms = PDB.read(structure, verbose=verbose)
        x = atoms['coord'][:, 0]
        y = atoms['coord'][:, 1]
        z = atoms['coord'][:, 2]
        vdw = atoms['radius']
        self.structure = structure
        self.dg = simulation_grid_resolution
        self.output_file = output_file

        # Determine Labeling position
        if verbose:
            print("Labeling position")
            print("Chain ID: %s" % chain_identifier)
            print("Residue seq. number: %s" % residue_seq_number)
            print("Residue name: %s" % residue_name)
            print("Atom name: %s" % atom_name)

        def _decode(value):
            if isinstance(value, bytes):
                return value.decode().strip()
            if isinstance(value, np.bytes_):
                return value.tobytes().decode().strip()
            if isinstance(value, np.str_):
                return str(value).strip()
            return str(value).strip()

        if attachment_atom_index is None:
            if residue_seq_number is None or atom_name is None:
                raise ValueError("either attachment_atom number or residue and atom_name have to be provided.")
            self.attachment_residue = residue_seq_number
            self.attachment_atom = atom_name
            if chain_identifier is None:
                attachment_atom_index = np.where((atoms['res_id'] == residue_seq_number) &
                                                (atoms['atom_name'] == atom_name))[0]
            else:
                attachment_atom_index = np.where((atoms['res_id'] == residue_seq_number) &
                                                 (atoms['atom_name'] == atom_name) &
                                                 (atoms['chain'] == chain_identifier))[0]
            if len(attachment_atom_index) != 1:
                # Name the selection: a project with two dyes hits this for the
                # acceptor as readily as for the donor, and the bare message
                # gave no way to tell which one was wrong.
                site = "%s%s:%s" % (
                    "" if chain_identifier is None else "%s:" % chain_identifier,
                    residue_seq_number,
                    atom_name,
                )
                if len(attachment_atom_index) == 0:
                    raise ValueError(
                        "No attachment atom %s in the structure." % site
                    )
                raise ValueError(
                    "Attachment atom %s is ambiguous: it matches %d atoms."
                    % (site, len(attachment_atom_index))
                )
            else:
                attachment_atom_index = attachment_atom_index[0]
            if verbose:
                print("Atom index: %s" %attachment_atom_index)
        else:
            if residue_seq_number is None:
                residue_seq_number = int(atoms['res_id'][attachment_atom_index])
            if atom_name is None:
                atom_name = _decode(atoms['atom_name'][attachment_atom_index])
            if chain_identifier is None:
                chain_identifier = _decode(atoms['chain'][attachment_atom_index])
            self.attachment_residue = residue_seq_number
            self.attachment_atom = atom_name

        if verbose:
            print("Calculating initial-AV")
            print("Linker-length  : %.2f" % linker_length)
            print("Linker-width   : %.2f" % linker_width)
            print("Linker-radius  : %.2f" % radius1)
            print("Attachment-atom: %i" % attachment_atom_index)
            print("AV-resolution  : %.2f" % simulation_grid_resolution)
        backend = resolve_av_backend(av_backend)
        self.av_backend = backend
        if verbose:
            print("AV backend    : %s" % backend)
        if backend == "imp_bff":
            av_result = build_imp_accessible_volume(
                structure=structure,
                residue_seq_number=residue_seq_number,
                atom_name=atom_name,
                chain_identifier=chain_identifier,
                linker_length=linker_length,
                linker_width=linker_width,
                radii=(radius1, radius2, radius3),
                allowed_sphere_radius=_clamped_allowed_sphere_radius(
                    allowed_sphere_radius
                ),
                simulation_grid_resolution=simulation_grid_resolution,
            )
            density = av_result.density
            ng = av_result.grid_size
            npm = int((ng - 1) // 2)
            # Anchor on the map's own origin rather than on the attachment atom.
            # IMP sizes the grid itself and the atom lands on the middle voxel
            # only when the edge length is odd; on the 148l reference site it is
            # 86, so assuming otherwise put `x0` -- and therefore every grid the
            # quenching and stickiness kernels index with it -- one voxel out.
            x0 = np.asarray(av_result.origin, dtype=np.float64) + npm * float(
                simulation_grid_resolution
            )
            n = density.sum()
            points = _points_from_centred_grid(
                density.reshape(ng, ng, ng), ng, simulation_grid_resolution, x0
            )
        else:
            if simulation_type == 'AV3':
                points, density, ng, x0 = calculate3R(
                    linker_length,
                    linker_width,
                    radius1,
                    radius2,
                    radius3,
                    attachment_atom_index,
                    x,
                    y,
                    z,
                    vdw,
                    dg=simulation_grid_resolution,
                    verbose=verbose,
                    linkersphere=allowed_sphere_radius,
                )
            else:
                points, density, ng, x0 = calculate1R(
                    linker_length,
                    linker_width,
                    radius1,
                    attachment_atom_index,
                    x,
                    y,
                    z,
                    vdw,
                    verbose=verbose,
                    linkersphere=allowed_sphere_radius,
                    dg=simulation_grid_resolution,
                )

        self.ng = ng
        self.points = points
        if len(points) == 0:
            chain_str = f" on chain '{chain_identifier}'" if chain_identifier else ""
            raise ValueError(
                f"Accessible Volume is empty (0 points). Residue {residue_seq_number} ({atom_name}){chain_str} "
                f"might be buried inside the protein."
            )
        self.points_fast = points.copy()
        self.density_fast = density
        self.density_slow = None
        self.slow_factor_grid = None
        self.k_quench_grid = None
        self.density = density
        self.x0 = x0
        if verbose:
            print("Points in total-AV: %i" % density.sum())
        if save_av:
            write_xyz(output_file + '.xyz', points, verbose=verbose)

    def calc_slow_av(
            self, slow_centers=None, slow_radius=None, save=True, verbose=True, replace_density=False,
            slow_fact=None, quench_radius=None, k_quench=None, quench_centers=None):
        """
        Calculates a subav given the accessible volume determined during class initiation.
        : list/numpy array of points dimension (npoints, 3)


        Parameters
        ----------
        :param slow_centers : array_like
             A list of Cartesian coordinated representing the slow centers. The shape should be (n,3)
             where n is the number of slow centers.
        :param slow_radius : array_like, or float
            A list of radii the shape
        :param slow_fact : array_like, float, or None
            Optional per-center diffusion scaling factors used to build a
            spatial slow-factor grid. Points affected by overlapping centers
            multiply all contributing factors.
        :param quench_centers : array_like, or None
            Optional Cartesian coordinates of the quenching centres, shape
            (n, 3), matching ``k_quench``. Defaults to ``slow_centers``. Passing
            separate centres lets the quenching rate be evaluated at each
            residue's redox-active moiety while stickiness stays on CB.
        :param quench_radius : array_like, float, or None
            Contact radius per quenching centre. Defaults to ``slow_radius``.
        :param k_quench : array_like, float, or None
            Quenching rate contributed by each centre inside its contact
            radius. Contributions of overlapping centres add up.
        :param replace_density : bool
            If replace_density is True the AV-density will be replaced by the density of the slow
            accessible volume
        :param verbose : bool
        :param save : bool
            If save is True the slow-accessible volume is saved to an xyz-file with the suffix '_slow'

        """
        verbose = verbose or self.verbose
        if verbose:
            print("Calculating slow-AV:")
            print(slow_centers)
        if isinstance(slow_radius, (int, float)):
            slow_radii = np.ones(slow_centers.shape[0])*slow_radius
        elif len(slow_radius) != slow_centers.shape[0]:
            raise ValueError("The size of the slow_radius doesnt match the number of slow_centers")
        else:
            slow_radii = np.asarray(slow_radius, dtype=np.float64)

        if slow_fact is None:
            slow_facts = None
        elif isinstance(slow_fact, (int, float)):
            slow_facts = np.ones(slow_centers.shape[0], dtype=np.float64) * float(slow_fact)
        elif len(slow_fact) != slow_centers.shape[0]:
            raise ValueError("The size of the slow_fact doesnt match the number of slow_centers")
        else:
            slow_facts = np.asarray(slow_fact, dtype=np.float64)

        if quench_centers is None:
            quench_points = slow_centers
        else:
            quench_points = np.asarray(quench_centers, dtype=np.float64).reshape((-1, 3))
            if quench_points.shape[0] != slow_centers.shape[0]:
                raise ValueError("The number of quench_centers doesnt match the number of slow_centers")

        if quench_radius is None:
            quench_radii = slow_radii
        elif isinstance(quench_radius, (int, float)):
            quench_radii = np.ones(quench_points.shape[0]) * quench_radius
        elif len(quench_radius) != quench_points.shape[0]:
            raise ValueError("The size of the quench_radius doesnt match the number of quench_centers")
        else:
            quench_radii = np.asarray(quench_radius, dtype=np.float64)

        if k_quench is None:
            k_quench_values = None
        elif isinstance(k_quench, (int, float)):
            k_quench_values = np.ones(quench_points.shape[0], dtype=np.float64) * float(k_quench)
        elif len(k_quench) != quench_points.shape[0]:
            raise ValueError("The size of the k_quench doesnt match the number of quench_centers")
        else:
            k_quench_values = np.asarray(k_quench, dtype=np.float64)

        if verbose or self.verbose:
            print("Calculating slow AV")
            print("slow centers: \n%s" % slow_centers)

        ng = int(self.ng)
        npm = int((ng - 1) // 2)
        density = self.density_fast
        dg = self.dg
        x0 = self.x0
        slow_density = subav(density, ng, dg, slow_radii, slow_centers, x0)
        n = slow_density.sum()
        points = _points_from_centred_grid(slow_density.reshape(ng, ng, ng), ng, dg, x0)
        if slow_facts is None:
            self.slow_factor_grid = None
        else:
            self.slow_factor_grid = slow_factor_grid(density, ng, dg, slow_radii, slow_centers, x0, slow_facts)
        if k_quench_values is None:
            self.k_quench_grid = None
        else:
            self.k_quench_grid = additive_factor_grid(density, ng, dg, quench_radii, quench_points, x0, k_quench_values)
        if verbose or self.verbose:
            print("Points in slow-AV: %i" % n)
        if save:
            write_xyz(self.output_file + '_slow.xyz', points, verbose=verbose)
        self.density_slow = slow_density
        self.points_slow = points
        if replace_density:
            self.density = slow_density

    @property
    def Rmp(self):
        return self.points.mean(axis=0)

    def dRmp(self, av):
        """
        Calculate the distance between the mean positions with respect to the accessible volume `av`

        Parameters
        ----------
        av : accessible volume object

        Example
        -------
        >>>         >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
        >>> structure = lib.Structure(pdb_filename)
        >>> av1 = lib.fps.AV(structure, residue_seq_number=72, atom_name='CB')
        >>> av2 = lib.fps.AV(structure, residue_seq_number=134, atom_name='CB')
        >>> av1.dRmp(av2)
        """
        return dRmp(self, av)

    def dRDA(self, av):
        """
        Calculate the mean distance to the second accessible volume `av`

        Parameters
        ----------
        av : accessible volume object

        Example
        -------
        >>>         >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
        >>> structure = lib.Structure(pdb_filename)
        >>> av1 = lib.fps.AV(structure, residue_seq_number=72, atom_name='CB')
        >>> av2 = lib.fps.AV(structure, residue_seq_number=134, atom_name='CB')
        >>> av1.dRDA(av2)
        """
        return RDAMean(self, av)


    def dRDAE(self, av):
        """
        Calculate the FRET-averaged mean distance to the second accessible volume `av`

        Parameters
        ----------
        av : accessible volume object

        Example
        -------
        >>>         >>> pdb_filename = '/sample_data/structure/T4L_Topology.pdb'
        >>> structure = lib.Structure(pdb_filename)
        >>> av1 = lib.fps.AV(structure, residue_seq_number=72, atom_name='CB')
        >>> av2 = lib.fps.AV(structure, residue_seq_number=134, atom_name='CB')
        >>> av1.dRDAE(av2)
        """
        return RDAMeanE(self, av)

    def __repr__(self):
        s = '\n'
        s += 'Accessible Volume\n'
        s += '-----------------\n'
        s += 'Attachment residue: %s\n' % self.attachment_residue
        s += 'Attachment atom: %s\n' % self.attachment_atom
        s += '\n'
        s += 'Fast-AV:\n'
        s += 'n-points: %s\n' % self.points_fast.shape[0]
        if self.points_slow is not None:
            s += 'Slow-AV:\n'
            s += 'n-points: %s\n' % self.points_slow.shape[0]
        else:
            s += 'Slow-AV: not calculated'
        return s


