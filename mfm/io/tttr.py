from . import _tttrlib
import numpy as np


def becker_merged(b):
    """Get the macro-time, micro-time and the routing channel number of a BH132-file contained in a
    binary numpy-array of 8-bit chars.

    :param b: numpy-array
        a numpy array of chars containing the binary information of a
        BH132-file
    :return: list
        a list containing the number of photons, numpy-array of macro-time (64-bit unsigned integers),
        numpy-array of TAC-values (32-bit unsigned integers), numpy-array of channel numbers (8-bit unsigned integers)
    """
    length = (b.shape[0] - 4) / 4
    event = np.zeros(length, dtype=np.uint64)
    mt = np.zeros(length, dtype=np.uint64)
    tac = np.zeros(length, dtype=np.uint32)
    can = np.zeros(length, dtype=np.uint8)
    g = _tttrlib.beckerMerged(b, can, tac, mt, event, length)
    return g, mt, 4095 - tac, can


def ht3(b):
    """Get the macro-time, micro-time and the routing channel number of a PQ-HT3-file (version 1) contained in a
    binary numpy-array of 8-bit chars.

    :param b: numpy-array
        a numpy array of chars containing the binary information of a
        BH132-file
    :return: list
        a list containing the number of photons, numpy-array of macro-time (64-bit unsigned integers),
        numpy-array of TAC-values (32-bit unsigned integers), numpy-array of channel numbers (8-bit unsigned integers)
    """
    length = (b.shape[0]) / 4
    event = np.zeros(length, dtype=np.uint64)
    mt = np.zeros(length, dtype=np.uint64)
    tac = np.zeros(length, dtype=np.uint32)
    can = np.zeros(length, dtype=np.uint8)
    number_of_photon = _tttrlib.ht3(b, can, tac, mt, event, length)
    return number_of_photon, mt, tac, can


def iss(b):
    # TODO: documentation, better, organization
    return _tttrlib.iss(b)