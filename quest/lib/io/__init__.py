from __future__ import annotations

import numpy as np
import quest.lib.io.pdb


def write_xyz(
        filename: str,
        points: np.ndarray,
        verbose: bool = False
):
    """
    Writes the points as xyz-format file. The xyz-format file can be opened and displayed for instance
    in PyMol
    :param filename: string
        Filename the cartesian coordinates in points are written to as xyz-format file
    :param points:
    :param verbose:
    """
    if verbose:
        print("\nwrite_xyz")
        print("---------")
        print("Filename: %s" % filename)
    fp = open(filename, 'w')
    npoints = len(points)
    fp.write('%i\n' % npoints)
    fp.write('Name\n')
    for p in points:
        fp.write('D %.3f %.3f %.3f\n' % (p[0], p[1], p[2]))
    fp.close()
    if verbose:
        print("-------------------")