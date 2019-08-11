from lib.math.linalg.vector import *
import numpy as np
import scipy.optimize


def solve_nnls(A, b, reg=1e-5, x_max=20):
    right = np.dot(A, b)
    left = np.dot(A, A.T) + reg * np.diag(np.ones_like(right)) * A.T.shape[0] / A.T.shape[1]
    x = scipy.optimize.nnls(left, right)[0]
    x[x > x_max] = 0
    return x
