"""Pure Python/Numba implementations of photon trace simulation.

This replaces the former Cython module ``photon.pyx`` but keeps the same
public API: ``simulate_photon_trace``, ``simulate_photon_trace_kQ`` and
``simulate_photon_trace_rate``.
"""
from __future__ import annotations

from typing import Tuple

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


@njit(cache=True, parallel=True)
def _simulate_trace(
    n_ph: int,
    collided: np.ndarray,
    quenching_prob: float,
    t_step: float,
    tau0: float,
) -> Tuple[np.ndarray, np.ndarray]:
    dts = np.zeros(n_ph, dtype=np.float64)
    phs = np.zeros(n_ph, dtype=np.uint8)
    n_frames = collided.shape[0]
    shift_nbrs = np.random.randint(0, n_frames // 2 if n_frames > 1 else 1, n_ph)

    if quenching_prob == 0.0:
        for i in prange(n_ph):
            dts[i] = _log_exp_wait(tau0)
            phs[i] = 1
        return dts, phs

    if quenching_prob >= 1.0:
        for i in prange(n_ph):
            dts[i] = _log_exp_wait(tau0)
            phs[i] = 1
            n_step = int(dts[i] / t_step)
            shift_nbr = shift_nbrs[i]
            for j in range(shift_nbr, shift_nbr + n_step):
                if j >= collided.shape[0]:
                    break
                if collided[j]:
                    dts[i] = 0.0
                    phs[i] = 0
                    break
        return dts, phs

    for i in prange(n_ph):
        dts[i] = _log_exp_wait(tau0)
        phs[i] = 1
        n_step = int(dts[i] / t_step)
        shift_nbr = shift_nbrs[i]
        for j in range(shift_nbr, shift_nbr + n_step):
            if j >= collided.shape[0]:
                break
            if collided[j] and _ranf() < quenching_prob:
                dts[i] = 0.0
                phs[i] = 0
                break
    return dts, phs


def simulate_photon_trace(
    n_ph: int,
    collided: np.ndarray,
    quenching_prob: float = 0.5,
    t_step: float = 0.01,
    tau0: float = 0.25,
):
    return _simulate_trace(n_ph, np.asarray(collided, dtype=np.uint8), quenching_prob, t_step, tau0)


@njit(cache=True, parallel=True)
def _simulate_trace_kq(
    n_ph: int, collided: np.ndarray, kq: float, t_step: float, tau0: float
) -> Tuple[np.ndarray, np.ndarray]:
    dts = np.zeros(n_ph, dtype=np.float64)
    phs = np.zeros(n_ph, dtype=np.uint8)
    n_frames = collided.shape[0]
    shift_nbrs = np.random.randint(0, n_frames // 2 if n_frames > 1 else 1, n_ph)
    p_quench = kq * t_step

    if kq == 0.0:
        for i in prange(n_ph):
            dts[i] = _log_exp_wait(tau0)
            phs[i] = 1
        return dts, phs

    for i in prange(n_ph):
        dts[i] = _log_exp_wait(tau0)
        phs[i] = 1
        n_step = int(dts[i] / t_step)
        shift_nbr = shift_nbrs[i]
        for j in range(shift_nbr, shift_nbr + n_step):
            if j >= collided.shape[0]:
                break
            if collided[j] and _ranf() < p_quench:
                dts[i] = 0.0
                phs[i] = 0
                break
    return dts, phs


def simulate_photon_trace_kQ(
    n_ph: int,
    collided: np.ndarray,
    kQ: float = 0.5,
    t_step: float = 0.01,
    tau0: float = 0.25,
):
    return _simulate_trace_kq(n_ph, np.asarray(collided, dtype=np.uint8), kQ, t_step, tau0)


@njit(cache=True, parallel=True)
def _simulate_trace_rate(
    n_ph: int, kq: np.ndarray, t_step: float, tau0: float
) -> Tuple[np.ndarray, np.ndarray]:
    dts = np.zeros(n_ph, dtype=np.float64)
    phs = np.zeros(n_ph, dtype=np.uint8)
    n_frames = kq.shape[0]
    shift_nbrs = np.random.randint(0, n_frames // 2 if n_frames > 1 else 1, n_ph)

    for i in prange(n_ph):
        dts[i] = _log_exp_wait(tau0)
        phs[i] = 1
        n_step = int(dts[i] / t_step)
        shift_nbr = shift_nbrs[i]
        for j in range(shift_nbr, shift_nbr + n_step):
            if j >= kq.shape[0]:
                break
            if _ranf() < kq[j] * t_step:
                dts[i] = 0.0
                phs[i] = 0
                break
    return dts, phs


def simulate_photon_trace_rate(
    n_ph: int,
    kQ: np.ndarray,
    t_step: float = 0.01,
    tau0: float = 0.25,
):
    return _simulate_trace_rate(n_ph, np.asarray(kQ, dtype=np.float32), t_step, tau0)


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

