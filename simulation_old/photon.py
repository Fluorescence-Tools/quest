from __future__ import annotations


import numpy as np

from quest.lib.tools.dye_diffusion import photon as quest_photon


def simulate_photon_trace_rate(
        n_ph,
        quench,
        t_step: float = 0.01,
        tau0: float = 4.10,
        n_traj: int = 4,
        **kwargs
):
    """Compatibility wrapper around QuEst's photon trace simulation.

    This function preserves the public API of the legacy chisurf
    ``simulate_photon_trace_rate`` but delegates the actual simulation
    to :mod:`quest.lib.tools.dye_diffusion.photon` so that the core
    implementation lives in a single place.
    """

    ravel_traj = kwargs.get('ravel_traj', True)

    n_ph = int(n_ph)
    n_traj = int(n_traj)
    quench_arr = np.asarray(quench, dtype=np.float32)

    if n_traj <= 1:
        dts, phs = quest_photon.simulate_photon_trace_rate(
            n_ph=n_ph,
            kQ=quench_arr,
            t_step=float(t_step),
            tau0=float(tau0),
        )
        return dts, phs

    # Split photons across virtual trajectories and optionally stack.
    n_ph_per = max(1, n_ph // n_traj)
    dts_all = []
    phs_all = []
    for _ in range(n_traj):
        dts_i, phs_i = quest_photon.simulate_photon_trace_rate(
            n_ph=n_ph_per,
            kQ=quench_arr,
            t_step=float(t_step),
            tau0=float(tau0),
        )
        dts_all.append(dts_i)
        phs_all.append(phs_i)

    dts = np.stack(dts_all, axis=0)
    phs = np.stack(phs_all, axis=0)
    if ravel_traj:
        dts = dts.ravel()
        phs = phs.ravel()
    return dts, phs


def simulate_decay_quench(
        n_curves: int,
        decay,
        dt_tac: float,
        k_quench,
        t_step,
        tau0: float,
        **kwargs
):
    """Compatibility wrapper for trajectory-based decay curve simulation.

    Delegates to :func:`quest.lib.tools.dye_diffusion.photon.simulate_decay_quench`
    so that the core implementation is maintained inside the QuEst package.
    """

    decay_arr = np.asarray(decay, dtype=np.float64)
    quest_photon.simulate_decay_quench(
        n_curves=int(n_curves),
        decay=decay_arr,
        dt_tac=float(dt_tac),
        k_quench=np.asarray(k_quench, dtype=np.float32),
        t_step=float(t_step),
        tau0=float(tau0),
    )
