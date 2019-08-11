import numpy as np
import _vector


def cartesian(arrays, out=None):
    """Generate a cartesian product of input arrays.

    :param arrays: list of arrays
        1-D arrays to form the cartesian product of.
    :param out: 2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.
    :return: 2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------

    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out


def angle(a, b, c):
    """
    The angle between three vectors

    :param a: numpy array
    :param b: numpy array
    :param c: numpy array
    :return: angle between three vectors/points in space
    """
    return _vector.angle(a, b, c)


def dihedral(v1, v2, v3, v4):
    """
    Dihedral angle between four-vectors

    :param v1:
    :param v2:
    :param v3:
    :param v4:

    :return: dihedral angle between four vectors
    """
    return _vector.dihedral(v1, v2, v3, v4)


def sub(a, b):
    """
    Subtract two 3D-vectors
    :param a:
    :param b:
    :return:
    """
    return _vector.sub3(a, b)


def add(a, b):
    """
    Add two 3D-vectors
    :param a:
    :param b:
    :return:
    """
    return _vector.add3(a, b)


def dot(a, b):
    """
    Vector product of two 3D-vectors.

    :param a: 1D numpy-array
    :param b: 1D numpy-array
    :return: dot-product of the two numpy arrays
    """
    return _vector.dot3(a, b)


def dist(u, v):
    """
    Distance between two vectors

    :param u:
    :param v:
    :return:
    """
    return _vector.dist(u, v)


def dist2(u, v):
    """
    Squared distance between two vectors
    :param u:
    :param v:
    :return:
    """
    return _vector.sq_dist(u, v)


def norm(v):
    """
    Euclidean  norm of a vector

    :param v: 1D numpy-array
    :return: normalized numpy-array
    """
    return _vector.norm3(v)


def cross(a, b):
    """
    Cross-product of two vectors

    :param a: numpy array of length 3
    :param b: numpy array of length 3
    :return: cross-product of a and b
    """
    return _vector.cross(a, b)
