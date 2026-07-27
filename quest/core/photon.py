"""Photon-trace and decay-curve kernels (numba).

Two entry points, both driven by a *per-frame quenching rate* array:
``simulate_photon_trace_rate`` samples individual photons, and
``simulate_decay_quench`` integrates a decay curve directly. The older
boolean-collision variants were removed once nothing called them.
"""
from __future__ import annotations

import os
from typing import Tuple

import numba
import numpy as np
from numba import njit, prange


_EPS = 2.4414062e-4  # match former C constant


@njit(cache=True)
def _ranf() -> float:
    return np.random.random()


@njit(cache=True)
def _log_exp_wait(tau0: float) -> float:
    # exponential waiting time sampling
    return np.log(1.0 / (_ranf() + _EPS)) * tau0


@njit(cache=True, nogil=True)
def _photon_rate_walk(kq, t_step, tau0, shift_nbr, n_frames):
    """Emit one photon: sample its lifetime, then race it against quenching."""
    dt = _log_exp_wait(tau0)
    n_step = int(dt / t_step)
    for j in range(shift_nbr, shift_nbr + n_step):
        if _ranf() < kq[j % n_frames] * t_step:
            return 0.0, np.uint8(0)
    return dt, np.uint8(1)


@njit(cache=True, parallel=True)
def _seed_worker_threads(base_seed: int) -> None:
    """Give every numba worker thread a fresh RNG state.

    Numba keeps one RNG state per thread and initialises it deterministically,
    so an *unseeded* run repeats itself for the whole life of the process --
    four Monte-Carlo runs in a row return byte-identical results, which reads
    as precision that is not there. Seeding one iteration per thread reaches
    each thread's own state.
    """
    for index in prange(numba.get_num_threads()):
        np.random.seed(base_seed + index)


@njit(cache=True, parallel=True)
def _simulate_trace_rate(
    n_ph: int, kq: np.ndarray, t_step: float, tau0: float
) -> Tuple[np.ndarray, np.ndarray]:
    dts = np.zeros(n_ph, dtype=np.float64)
    phs = np.zeros(n_ph, dtype=np.uint8)
    n_frames = kq.shape[0]
    if n_frames == 0:
        return dts, phs
    shift_nbrs = np.random.randint(0, n_frames if n_frames > 1 else 1, n_ph)

    for i in prange(n_ph):
        dts[i], phs[i] = _photon_rate_walk(kq, t_step, tau0, shift_nbrs[i], n_frames)
    return dts, phs


# Seeded counterpart of `_simulate_trace_rate`. It has to stay sequential:
# under `parallel=True` numba keeps one RNG state per thread and hands
# iterations to whichever thread is free, so the draws a photon sees would
# depend on the scheduler and the seed would not pin the result down.
@njit(cache=True, nogil=True)
def _simulate_trace_rate_seeded(
    n_ph: int, kq: np.ndarray, t_step: float, tau0: float, random_seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(random_seed)
    dts = np.zeros(n_ph, dtype=np.float64)
    phs = np.zeros(n_ph, dtype=np.uint8)
    n_frames = kq.shape[0]
    if n_frames == 0:
        return dts, phs
    shift_nbrs = np.random.randint(0, n_frames if n_frames > 1 else 1, n_ph)

    for i in range(n_ph):
        dts[i], phs[i] = _photon_rate_walk(kq, t_step, tau0, shift_nbrs[i], n_frames)
    return dts, phs


def simulate_photon_trace_rate(
    n_ph: int,
    kQ: np.ndarray,
    t_step: float = 0.01,
    tau0: float = 0.25,
    random_seed: int | None = None,
):
    """Monte-Carlo photon trace against a per-frame quenching rate.

    Passing ``random_seed`` makes the trace reproducible, at the cost of
    running single-threaded; see `_simulate_trace_rate_seeded`.
    """
    kq = np.asarray(kQ, dtype=np.float32)
    if random_seed is None:
        # Draw the thread seeds from the OS so consecutive unseeded runs are
        # genuinely independent samples, not the same one four times.
        _seed_worker_threads(int.from_bytes(os.urandom(4), "little"))
        return _simulate_trace_rate(n_ph, kq, t_step, tau0)
    return _simulate_trace_rate_seeded(n_ph, kq, t_step, tau0, int(random_seed))


@njit(cache=True, parallel=True)
def _simulate_decay_quench_core(
    n_curves: int,
    decay: np.ndarray,
    dt_tac: float,
    k_quench: np.ndarray,
    t_step: float,
    tau0: float,
) -> None:
    """Approximate trajectory-based decay curve by integrating quenching rates.

    This function mirrors the conceptual behaviour of the original
    ``chisurf.fluorescence.simulation.photon.simulate_decay_quench`` Cython
    implementation:

    - For each of ``n_curves`` virtual trajectories, pick a random starting
      frame along ``k_quench``.
    - Propagate the excited-state population in discrete time steps ``t_step``
      using a total decay rate ``1/tau0 + k_quench[frame]``.
    - Accumulate the emitted intensity into TAC bins of width ``dt_tac``.

    The result is a smooth decay curve in ``decay`` without generating
    individual photons.
    """

    n_frames = k_quench.shape[0]
    n_bins = decay.shape[0]
    if n_frames == 0 or n_bins == 0 or n_curves <= 0:
        return

    intrinsic_rate = 1.0 / tau0 if tau0 > 0.0 else 0.0
    max_time = dt_tac * n_bins

    # Random starting frame indices (mimic legacy shift behaviour)
    shifts = np.random.randint(0, n_frames // 2 if n_frames > 1 else 1, n_curves)

    for i in prange(n_curves):
        shift = shifts[i]
        intensity = 1.0
        t = 0.0
        frame = 0

        # Propagate until the population is negligible or TAC window is covered.
        while t < max_time and intensity > 1e-6 and frame < n_frames:
            idx = (shift + frame) % n_frames
            total_rate = intrinsic_rate + float(k_quench[idx])
            dt = t_step
            if t + dt > max_time:
                dt = max_time - t

            if dt <= 0.0:
                break

            emitted = intensity * total_rate * dt
            bin_idx = int(t / dt_tac)
            if 0 <= bin_idx < n_bins:
                decay[bin_idx] += emitted

            # Exponential decay of the remaining population
            intensity *= np.exp(-total_rate * dt)
            t += dt
            frame += 1


def simulate_decay_quench(
    n_curves: int,
    decay: np.ndarray,
    dt_tac: float,
    k_quench: np.ndarray,
    t_step: float,
    tau0: float,
) -> None:
    """Public wrapper for trajectory-based decay-curve simulation.

    Parameters
    ----------
    n_curves:
        Number of virtual decay curves (samples) to average over.
    decay:
        1D array that will be filled in-place with the decay curve.
    dt_tac:
        TAC bin width in ns.
    k_quench:
        1D array of instantaneous quenching rates per trajectory frame.
    t_step:
        Time step of the diffusion simulation in ns.
    tau0:
        Intrinsic excited-state lifetime in ns.
    """

    _simulate_decay_quench_core(
        int(n_curves),
        np.asarray(decay, dtype=np.float64),
        float(dt_tac),
        np.asarray(k_quench, dtype=np.float32),
        float(t_step),
        float(tau0),
    )
