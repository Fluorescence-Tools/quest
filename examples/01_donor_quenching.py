"""Simulate the fluorescence decay of a dye tethered to a protein.

What QuEst does, in one run: build the dye's accessible volume, diffuse it there
under Brownian dynamics, and let photo-induced electron transfer from nearby
amino acids quench it. The output is a decay curve you could compare with a
TCSPC measurement, plus the numbers that summarise it.

    python examples/01_donor_quenching.py
"""

from __future__ import annotations

import numpy as np

from _common import example_project
from quest.api import simulate_project, top_quenchers


def main() -> int:
    # Residue 100 of T4 lysozyme sits next to a tryptophan, so the quenching is
    # unmistakable. Try 55 for a site the dye barely quenches at.
    project = example_project(chain="E", residue=100)
    print(f"Structure : {project['pdb']}")
    site = project["attachment"]
    print(f"Labeling  : chain {site['chain']}, residue {site['residue']}, atom {site['atom']}")

    # save_outputs=False keeps the run in memory; set it True to get a
    # jobs/<uuid>/ directory with the decay, the volumes and the point clouds.
    result = simulate_project(project, save_outputs=False)

    print("\n--- Decay -------------------------------------------------------")
    print(f"  bins                {result.time.size} over {result.time[-1]:.1f} ns")
    print(f"  photons collected   {int(result.donor_counts.sum()):,}")

    print("\n--- Derived quantities ------------------------------------------")
    print(f"  quantum yield       {result.quantum_yield_donor:.4f}")
    print(f"  mean lifetime       {result.lifetime_donor:.3f} ns  (tau0 = {project['tau0']} ns)")
    print(f"  contact fraction    {result.collisions_fraction * 100:.2f} %")
    print(f"  accessible volume   {result.av_volume:.0f} A^3")
    print(f"  contact volume      {result.contact_volume:.0f} A^3")

    # Quenching is resolved per residue type: this is what makes the answer
    # interpretable rather than just a number.
    print("\n--- What quenches the dye ---------------------------------------")
    ranked = top_quenchers(result.aa_residence, limit=5)
    if not ranked:
        print("  nothing: the dye never reaches a quenching residue from here")
    # top_quenchers is the summary; result.aa_residence has the full table,
    # including how long the dye spent near each residue type.
    residence = {row["residue_name"]: row for row in (result.aa_residence or [])}
    for entry in ranked:
        row = residence.get(entry["residue_name"], {})
        print(
            f"  {entry['residue_name']:>4s}"
            f"  {entry['quench_rate_fraction'] * 100:5.1f} % of the quenching"
            f"   in contact {row.get('fraction_near', 0.0) * 100:6.3f} % of frames"
            f"   ({row.get('event_count', 0)} visits)"
        )

    # The decay is multi-exponential even though the dye has one unquenched
    # lifetime: the dye samples environments that quench it differently.
    counts = result.donor_counts
    peak = int(np.argmax(counts))
    tail = counts[peak:]
    positive = tail > 0
    if positive.sum() > 10:
        slope = np.polyfit(result.time[peak:][positive], np.log(tail[positive]), 1)[0]
        print(f"\n  single-exponential fit of the tail: {-1 / slope:.2f} ns")
        print("  (compare with the mean lifetime above; they differ because the")
        print("   decay is not single-exponential)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
