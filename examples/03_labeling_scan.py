"""Choose a labeling site: simulate several, compare what quenches them.

This is the question QuEst exists to answer before an experiment — which
position gives a dye that is bright, or deliberately quenched.

    python examples/03_labeling_scan.py
"""

from __future__ import annotations

from _common import example_project
from quest.scan import scan_positions


def main() -> int:
    # Candidate sites on chain E of T4 lysozyme.
    candidates = {
        f"E:{residue}": {
            "chain_identifier": "E",
            "residue_seq_number": residue,
            "atom_name": "CB",
        }
        for residue in (15, 55, 100, 132)
    }

    project = example_project()
    print(f"scanning {len(candidates)} sites on {project['pdb']}\n")

    header = f"{'site':>8s}  {'QY':>6s}  {'tau [ns]':>8s}  {'contact':>8s}  quenched by"
    print(header)
    print("-" * len(header))

    # scan_positions never raises: a buried or unlabelable site is reported as
    # data, so one bad candidate cannot abort the scan.
    for record in scan_positions(project, candidates):
        if record["status"] != "success":
            print(f"{record['name']:>8s}  failed: {record['error'][:48]}")
            continue
        quencher = record["dominant_quencher"] or "-"
        share = record["dominant_quencher_fraction"]
        share_text = f"{share * 100:.0f}%" if share else ""
        print(
            f"{record['name']:>8s}  {record['quantum_yield']:6.3f}  "
            f"{record['lifetime']:8.2f}  {record['contact_fraction'] * 100:7.1f}%  "
            f"{quencher} {share_text}"
        )

    print("\nA bright site (high QY) reports the dye faithfully; a quenched one")
    print("is sensitive to its environment. Which you want depends on the experiment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
