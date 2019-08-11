from . import _tcspc


def fconv_per_cs(fit, x, lamp, start, stop, n_points, period, dt, conv_stop):
    """
    Fast convolution at high repetition rate with stop. Originally developed for
    Paris.

    :param fit: array of doubles
        Here the convolved fit is stored
    :param x: array of doubles
        Lifetime-spectrum of the form (amplitude, lifetime, amplitude, lifetime, ...)
    :param lamp: array-doubles
        The instrument response function
    :param start: int
        Start channel of convolution (position in array of IRF)
    :param stop: int
        Stop channel of convolution (position in array of IRF)
    :param n_points: int
        Number of points in fit and lamp
    :param period: double
        Period of repetition in nano-seconds
    :param dt: double
        Channel-width in nano-seconds
    :param conv_stop: int
        Stopping channel of convolution

     Remarks
     -------

     Fit and lamp have to have the same length.
     Seems to have problems with rising components (acceptor rise):_fconv_per works

    """
    stop = min(stop, n_points - 1)
    start = max(start, 0)
    return _tcspc.fconv_per_cs(fit, x, lamp, start, stop, n_points, period, dt, conv_stop)


def rescale_w_bg(fit, decay, w_res, bg, start, stop):
    return _tcspc.rescale_w_bg(fit, decay, w_res, bg, start, stop)


def fconv(y, x, irf, stop, dt):
    """
    :param y: numpy-array
        the content of this array is overwritten by the y-values after convolution
    :param x: vector of amplitdes and lifetimes in form: amplitude, lifetime
    :param irf:
    :param stop:
    :param dt:
    :return:
    """
    return _tcspc.fconv(y, x, irf, stop, dt)


def pddem(decayA, decayB, k, px, pm, pAB):
    """
    Electronic Energy Transfer within Asymmetric
    Pairs of Fluorophores: Partial Donor-Donor
    Energy Migration (PDDEM)
    Stanislav Kalinin
    http://www.diva-portal.org/smash/get/diva2:143149/FULLTEXT01


    Kalinin, S.V., Molotkovsky, J.G., and Johansson, L.B.
    Partial Donor-Donor Energy Migration (PDDEM) as a Fluorescence
    Spectroscopic Tool for Measuring Distances in Biomacromolecules.
    Spectrochim. Acta A, 58 (2002) 1087-1097.

    -> same results as Stas pddem code (pddem_t.c)

    :param decayA: decay A in form of [ampl lifetime, apml, lifetime...]
    :param decayB: decay B in form of [ampl lifetime, apml, lifetime...]
    :param k: rates of energy transfer [kAB, kBA]
    :param px: probabilities of excitation (pxA, pxB)
    :param pm: probabilities of emission (pmA, pmB)
    :param pAB: pure AB [0., 0]
    :return:
    """
    return _tcspc.pddem(decayA, decayB, k, px, pm, pAB)


def pile_up(data, model, rep_rate, dead_time, measurement_time, verbose=False):
    """
    Add pile up effect to model function.
    Attention: This changes the scaling of the model function.

    :param rep_rate: float
        The repetition-rate in MHz
    :param dead_time: float
        The dead-time of the system in nanoseconds
    :param measurement_time: float
        The measurement time in seconds
    :param data: numpy-array
        The array containing the experimental decay
    :param model: numpy-array
        The array containing the model function

    References
    ----------

    .. [1]  Coates, P.B., A fluorimetric attachment for an
            atomic-absorption spectrophotometer
            1968 J. Phys. E: Sci. Instrum. 1 878

    .. [2]  Walker, J.G., Iterative correction for pile-up in
            single-photon lifetime measurement
            2002 Optics Comm. 201 271-277

    """
    rep_rate *= 1e6
    dead_time *= 1e-9
    n_pulse_detected = data.sum()
    total_dead_time = n_pulse_detected * dead_time
    live_time = measurement_time - total_dead_time
    n_excitation_pulses = live_time * rep_rate
    if verbose:
        print "------------------"
        print "rep. rate [Hz]: %s" % rep_rate
        print "live time [s]: %s" % live_time
        print "dead time per pulse [s]: %s" % dead_time
        print "n_pulse: %s" % n_excitation_pulses
        print "dead [s]: %s" % total_dead_time
    return _tcspc.pile_up(n_excitation_pulses, data, model)