import numpy as np

from mfm.fluorescence.fcs import _fcs


weightCalculations = ['Koppel', 'none']
correlationMethods = ['tp']


def surenWeights(t, g, dur, cr):
    """
    :param t: correlation times [ms]
    :param g: correlation amplitude
    :param dur: measurement duration [s]
    :param cr: count-rate [kHz]
    """
    dt = np.diff(t)
    dt = np.hstack([dt, dt[-1]])
    ns = dur * 1000.0 / dt
    na = dt * cr
    syn = (t < 10) + (t >= 10) * 10 ** (-np.log(t + 1e-12) / np.log(10) + 1)
    b = np.mean(g[1:5]) - 1
    imaxhalf = np.min(np.nonzero(g < b / 2 + 1))
    tmaxhalf = t[imaxhalf]
    A = np.exp(-2 * dt / tmaxhalf)
    B = np.exp(-2 * t / tmaxhalf)
    m = t / dt
    S = (b * b / ns * ((1 + A) * (1 + B) + 2 * m * (1 - A) * B) / (1 - A) + 2 * b / ns / na * (1 + B) + (1 + b * np.sqrt(B)) / (ns * na * na)) * syn
    S = np.abs(S)
    return 1./ np.sqrt(S)


def normalize(np1, np2, dt1, dt2, tau, corr, B):
    cr1 = float(np1) / float(dt1)
    cr2 = float(np2) / float(dt2)
    for j in range(corr.shape[0]):
        pw = 2.0**int(j / B)
        tCor = dt1 if dt1 < dt2 - tau[j] else dt2 - tau[j]
        corr[j] /= (tCor * float(pw))
        corr[j] /= (cr1 * cr2)
        tau[j] = tau[j] // pw * pw
    return float(min(cr1, cr2))


def tp(mt, tac, rout, cr_filter, w1, w2, B, nc, fine, nTAC):
    return _fcs.correlate_tp(mt, tac, rout, cr_filter, w1, w2, B, nc, fine, nTAC)


def crFilter(mt, nPh, timeWindow, timeChunk, tolerance):
    return _fcs.count_rate_filter(mt, nPh, timeWindow, timeChunk, tolerance)


def getWeights(rout, tac, wt, nPh):
    return _fcs.get_weights(rout, tac, wt, nPh)

