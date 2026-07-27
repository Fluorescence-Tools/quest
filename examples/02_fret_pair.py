"""Simulate a FRET pair: donor, acceptor, and the transfer between them.

The donor diffuses through its accessible volume while the transfer rate is
evaluated *per frame*, so FRET responds to where the dye actually is rather
than to one averaged distance.

    python examples/02_fret_pair.py
"""

from __future__ import annotations

import numpy as np

from _common import example_project
from quest.api import simulate_project
from quest.project import DEFAULT_FORSTER_RADIUS


def fret_project(donor: int, acceptor: int, r0: float = DEFAULT_FORSTER_RADIUS):
    """A donor/acceptor project. `r0` is a Forster radius in **Angstrom**."""
    project = example_project(with_fret=True, residue=donor)
    project["fret"]["dyes"][1]["attachment"] = {
        "chain": "E", "residue": acceptor, "atom": "CB",
    }
    project["fret"]["R0_matrix"] = [[None, r0], [r0, None]]
    return project


def main() -> int:
    donor, acceptor = 55, 132
    result = simulate_project(fret_project(donor, acceptor), save_outputs=False)

    print(f"donor site      E:{donor}")
    print(f"acceptor site   E:{acceptor}")
    print(f"Forster radius  {DEFAULT_FORSTER_RADIUS} A   (Angstrom — a value in nm switches FRET off)")

    print("\n--- Yields -------------------------------------------------------")
    print(f"  donor alone        {result.quantum_yield_donor:.4f}")
    print(f"  donor + acceptor   {result.quantum_yield_fret:.4f}")
    print(f"  transfer efficiency {result.fret_efficiency:.4f}")

    # E(t) = f_DA(t) / f_D0(t): the time-resolved transfer efficiency, which is
    # what a distance distribution is deconvolved from (notebook 03).
    t = result.time
    fd0 = result.donor_counts.astype(float)
    fda = result.fret_counts.astype(float)
    # Skip the first nanosecond: the histogram's leading bins carry the
    # excitation spike, where the ratio of two decays is meaningless.
    usable = (fd0 > 20.0) & (t > 1.0) & (t < 20.0)
    et = 1.0 - fda[usable] / fd0[usable]
    print("\n--- E(t) ---------------------------------------------------------")
    print(f"  early ({t[usable][0]:.1f} ns)  {et[0]:.3f}")
    print(f"  late  ({t[usable][-1]:.1f} ns)  {et[-1]:.3f}")

    # A smaller Forster radius must transfer less. Cheap, and it catches a unit
    # mistake immediately.
    weak = simulate_project(fret_project(donor, acceptor, r0=35.0), save_outputs=False)
    print("\n--- Sanity: efficiency follows R0 --------------------------------")
    print(f"  R0 = 52 A -> E = {result.fret_efficiency:.3f}")
    print(f"  R0 = 35 A -> E = {weak.fret_efficiency:.3f}")
    assert weak.fret_efficiency < result.fret_efficiency
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
