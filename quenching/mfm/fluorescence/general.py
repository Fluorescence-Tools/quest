import numexpr as ne
import numpy as np
import numpy.linalg as linalg
import scipy.optimize


rate2lifetime = lambda rate, lifetime: ne.evaluate("1. / (1. / lifetime + rate)")
et = lambda fd0, fda: fda / fd0


def fretrate2distance(fretrate, R0, tau0, kappa2=2./3.):
    """Calculate the distance given a FRET-rate

    :param fretrate: FRET-rate
    :param R0: Forster radius
    :param tau0: lifetime of the donor
    :param kappa2: orientation factor
    :return:
    """
    return R0 * (fretrate * tau0/kappa2 * 2./3.)**(-1./6)


def tcspc_weights(y):
    """
    Calculated the weights used in TCSPC. Poissonian noise (sqrt of counts)
    :param y: numpy-array
        the photon counts
    :return: numpy-array
        an weighting array for fittin
    """
    if min(y) >= 0:
        w = np.array(y, dtype=np.float64)
        w[w == 0.0] = 1e30
        w = 1. / np.sqrt(w)
    else:
        w = np.ones_like(y)
    return w


def tcspc_fitrange(y, threshold=5.0, area=0.999):
    """
    Determines the fitting range based on the total number of photons to be fitted (fitting area).

    :param y: numpy-array
        a numpy array containing the photon counts
    :param threshold: float
        a threshold value. Lowest index of the fitting range is the first encountered bin with a photon count
        higher than the threshold.
    :param area: float
        The area which should be considered for fitting. Here 1.0 corresponds to all measured photons. 0.9 to 90%
        of the total measured photons.
    :return:
    """
    try:
        xmin = np.where(y > threshold)[0][0]
        cumsum = np.cumsum(y, dtype=np.float64)
        s = np.sum(y, dtype=np.float64)
        xmax = np.where(cumsum >= s * area)[0][0]
        return xmin, xmax
    except IndexError:
        return None, None


def interleaved_to_two_columns(ls):
    """
    Converts an interleaved spectrum to two column-data
    :param ls: numpy array
        The interleaved spectrum (amplitude, lifetime)
    :return:
    """
    x, t = ls.reshape((ls.shape[0]/2, 2)).T
    return x, t


def two_column_to_interleaved(x, t):
    """
    Converts two column lifetime spectra to interleaved lifetime spectra
    :param ls: The
    :return:
    """
    c = np.vstack((x, t)).ravel([-1])
    return c


def species_averaged_lifetime(lifetime_spectrum, normalize=True):
    """
    Calculates the species averaged lifetime given a lifetime spectrum

    :param lifetime_spectrum: inter-leaved lifetime-spectrum
    :return:
    """
    x, t = interleaved_to_two_columns(lifetime_spectrum)
    if normalize:
        x /= x.sum()
    tau_x = np.dot(x, t)
    return tau_x


def fluorescence_averaged_lifetime(lifetime_spectrum, taux=None, normalize=True):
    """

    :param lifetime_spectrum: interleaved lifetime spectrum
    :param taux: float
        The species averaged lifetime. If this value is not provided it is calculated based
        on th lifetime spectrum
    :return:
    """
    taux = species_averaged_lifetime(lifetime_spectrum) if taux is None else taux
    x, t = interleaved_to_two_columns(lifetime_spectrum)
    if normalize:
        x /= x.sum()
    tau_f = np.dot(x, t**2) / taux
    return tau_f


def phasor_giw(f, n, omega, times):
    """Phasor plot gi(w)
    The phasor approach to fluorescence lifetime page 236

    :param f: array of the fluorescence intensity at the provided times
    :param n: the nth harmonics
    :param omega: the angular frequency (2*pi*frequency)
    :param times: the times of the fluorescence intensities
    :return:
    """
    y = f * np.cos(n * omega * times)
    x = times
    return np.trapz(y, x) / np.trapz(f, x)


def phasor_siw(f, n, omega, times):
    """Phasor plot gi(w)
    The phasor approach to fluorescence lifetime page 236

    :param f: array of the fluorescence intensity at the provided times
    :param n: the nth harmonics
    :param omega: the angular frequency (2*pi*frequency)
    :param times: the times of the fluorescence intensities
    :return:
    """
    y = f * np.sin(n * omega * times)
    x = times
    return np.trapz(y, x) / np.trapz(f, x)


def fdfa2transfer_efficency(fdfa, phiD, phiA):
    """This function converts the donor-acceptor intensity ration FD/FA to the transfer-efficency E

    :param fdfa: float
        donor acceptor intensity ratio
    :param phiD: float
        donor quantum yield
    :param phiA: float
        acceptor quantum yield
    :return: float, the FRET transfer efficency
    """
    r = 1 + fdfa * phiA/phiD
    trans = 1./r
    return trans


def transfer_efficency2fdfa(E, phiD, phiA):
    """This function converts the transfer-efficency E to the donor-acceptor intensity ration FD/FA to

    :param E: float
        The transfer-efficency
    :param phiD: float
        donor quantum yield
    :param phiA: float
        acceptor quantum yield
    :return: float, the FRET transfer efficency
    """
    fdfa = phiA/phiD*(1./E-1)
    return fdfa


def distance2fretrate(r, R0, tau0, kappa2=2./3.):
    """ Converts the DA-distance to a FRET-rate

    :param r: donor-acceptor distance
    :param R0: Forster-radius
    :param tau0: lifetime
    :param kappa2: orientation factor
    :return:
    """
    return 3./2. * kappa2 * 1./tau0*(R0/r)**6.0


def distance2transfer(distance, R0):
    """

    .. math::

        E = 1.0 / (1.0 + (R_{DA} / R_0)^6)

    :param distance: DA-distance
    :param R0: Forster-radius
    :return:
    """
    return 1.0 / (1.0 + (distance / R0) ** 6)


def lifetime2transfer(tau, tau0):
    """

    .. math::

        E = 1 - tau / tau_0

    :param tau:
    :param tau0:
    :return:
    """
    return 1 - tau / tau0


def transfer2distance(E, R0):
    """
    Converts the transfer-efficency to a distance

    .. math::

        R_{DA} = R_0 (1 / E - 1)^{1/6}

    :param E: Transfer-efficency
    :param R0: Forster-radius
    :return:
    """
    return (1 / E - 1) ** (1.0 / 6.0) * R0


def transfer2lifetime(E, tau0):
    """

    .. math::

        tau_{DA} = (1 - E) * tau_0

    :param E:
    :param tau0:
    :return:
    """
    return (1 - E) * tau0


def distance2rate(distance, kappa2, tau0, R0):
    """
    Converts the DA-distance to a FRET-rate

    :param distance: DA-distance
    :param kappa2: orientation factor
    :param tau0: radiative lifetime
    :param R0: Forster-radius
    :return:
    """
    factor = np.array(3./2. * kappa2 / tau0 * R0**6, dtype=distance.dtype)
    r = ne.evaluate("factor * (1. / distance) ** 6")
    return r.astype(dtype=distance.dtype)


def transfer_space(e_min, e_max, n_steps, R0=52.0):
    """
    Generates distances with equally spaced transfer efficiencies

    :param e_min: float
        minimum transfer efficency
    :param e_max: float
        maximum transfer efficency
    :param n_steps: int
        number of distances
    :param R0: float
        Forster-radius
    :return:
    """
    es = np.linspace(e_min, e_max, n_steps)
    rdas = transfer2distance(es, R0)
    return rdas


def kappasq(delta, sD2, sA2, beta1=None, beta2=None):
    """
    Calculates the kappa2 distribution given the order parameter sD2 and sA2

    :param delta:
    :param sD2: order parameter of donor s2D = - sqrt(r_inf_D/r0)
    :param sA2: order parameter of acceptor s2A = sqrt(r_inf_A/r0)
    :param beta1:
    :param beta2:
    """
    if beta1 is None or beta2 is None:
        beta1 = 0
        beta2 = delta

    s2delta = (3.0 * np.cos(delta) * np.cos(delta) - 1.0) / 2.0
    s2beta1 = (3.0 * np.cos(beta1) * np.cos(beta1) - 1.0) / 2.0
    s2beta2 = (3.0 * np.cos(beta2) * np.cos(beta2) - 1.0) / 2.0
    k2 = 2.0 / 3.0 * (1 + sD2 * s2beta1 + sA2 * s2beta2 +
                  sD2 * sA2 * (s2delta +
                                 6 * s2beta1 * s2beta2 +
                                 1 + 2 * s2beta1 +
                                 2 * s2beta2 -
                                 9 * np.cos(beta1) *
                                 np.cos(beta2) * np.cos(delta)))
    return k2


def calc_transfer_matrix(t, rDA_min=1.0, rDA_max=200.0, n_steps=200.0, kappa2=0.667, space='lin', **kwargs):
    """
    Calculates a matrix converting a distance distribution to an E(t)-decay


    :param t:
    :param rDA_min:
    :param rDA_max:
    :param n_steps:
    :param kappa2:
    :param log_space:
    :param kwargs:
        tau0 - lifetime,
        R0 - Forster-radius,
        n_donor_bins: int (default 10)
            the number of bins with rate 0.0 (Donor-only). The longest distances are
            replaced by zero-rates

    Examples
    --------

    >>> t = np.arange(0, 20, 0.0141)
    >>> m, r_da = calc_transfer_matrix(t)

    .. plot:: plots/e_transfer_matrix.py

    """
    R0 = kwargs.get('R0', 52.0)
    tau0 = kwargs.get('tau0', 4.0)
    r_DA = kwargs.get('r_DA', None)
    n_donor_bins = kwargs.get('n_donor_bins', 10)

    if r_DA is None:
        if space == 'lin':
            r_DA = np.linspace(rDA_min, rDA_max, n_steps)
        elif space == 'log':
            lmin = np.log10(rDA_min)
            lmax = np.log10(rDA_max)
            r_DA = np.logspace(lmin, lmax, n_steps)
        elif space == 'trans':
            e_min = distance2transfer(rDA_min, R0)
            e_max = distance2transfer(rDA_max, R0)
            r_DA = transfer_space(e_min, e_max, n_steps, R0)

    rates = distance2rate(r_DA, kappa2, tau0, R0)
    # Use the last bins for D-Only
    rates[-1:-n_donor_bins] = 0.0
    m = np.outer(rates, t)
    M = np.nan_to_num(np.exp(-m))
    return M, r_DA


def calc_decay_matrix(t, tau_min=0.01, tau_max=200.0, n_steps=200.0, space='lin'):
    """
    Calculates a fluorescence decay matrix converting probabilities of lifetimes to a time-resolved
    fluorescence intensity

    :param t:
    :param tau_min:
    :param tau_max:
    :param n_steps:
    :param space:

    Examples
    --------

    >>> t = np.arange(0, 20, 0.0141)
    >>> m, r_da = calc_decay_matrix(t)

    Now plot the decay matrix

    >>> import pylab as p
    >>> p.imshow(m)
    >>> p.show()
    """
    if space == 'lin':
        taus = np.linspace(tau_min, tau_max, n_steps)
    elif space == 'log':
        lmin = np.log10(tau_min)
        lmax = np.log10(tau_max)
        taus = np.logspace(lmin, lmax, n_steps)
    rates = 1. / taus
    m = np.outer(rates, t)
    M = np.nan_to_num(np.exp(-m))
    return M, taus


def et2pRDA(ts, et, t_matrix=None, r_DA=None, **kwargs):
    """Calculates the distance distribution given an E(t) decay
    Here the amplitudes of E(t) are passed as well as the time-axis. If no transfer-matrix is provided it will
    be calculated in a range from 5 Ang to 200 Ang assuming a lifetime of 4 ns with a Forster-radius of 52 Ang.
    These parameters can be provided by *kwargs*

    :param t:
    :param et:
    :param t_matrix:
    :param r_DA:
    :param kwargs: tau0 donor-lifetime in absence of quenching, R0 - Forster-Radius
    :return: a distance and probability array

    Examples
    --------

    First calculate an E(t)-decay

    >>> rda_mean = [45.1, 65.0]
    >>> rda_sigma = [8.0, 8.0]
    >>> amplitudes = [0.6, 0.4]
    >>> rates = gaussian2rates(rda_mean, rda_sigma, amplitudes, interleaved=False)
    >>> a = rates[:,0]
    >>> kFRET = rates[:,1]
    >>> ts = np.logspace(0.1, 3, 18000)
    >>> et = np.array([np.dot(a, np.exp(-kFRET * t)) for t in ts])

    """
    if t_matrix is None or r_DA is None:
        t_matrix, r_DA = calc_transfer_matrix(ts, 5, 200, 200, **kwargs)

    p_rDA = scipy.optimize.nnls(t_matrix.T, et)[0]
    return r_DA, p_rDA


def kappasqAllDelta(delta, sD2, sA2, step=0.25, n_bins=31):
    """
    :param delta:
    :param sD2:
    :param sA2:
    :param step: step in degree
    :return:
    """
    #beta angles
    beta1 = np.arange(0.001, np.pi/2, step*np.pi/180.0)
    phi = np.arange(0.001, 2*np.pi, step*np.pi/180.0)
    n = beta1.shape[0]
    m = phi.shape[0]
    R = np.array([1, 0, 0])

    # kappa-square values for allowed betas
    k2 = np.zeros((n, m))
    k2hist = np.zeros(n_bins - 1)
    k2scale = np.linspace(0, 4, n_bins) # histogram bin edges

    for i in range(n):
        d1 = np.array([np.cos(beta1[i]),  0, np.sin(beta1[i])])
        n1 = np.array([-np.sin(beta1[i]), 0, np.cos(beta1[i])])
        n2 = np.array([0, 1, 0])
        for j in range(m):
            d2 = (n1*np.cos(phi[j])+n2*np.sin(phi[j]))*np.sin(delta)+d1*np.cos(delta)
            beta2 = np.arccos(abs(np.dot(d2, R)))
            k2[i, j] = kappasq(delta, sD2, sA2, beta1[i], beta2)
        y, x = np.histogram(k2[i, :], bins=k2scale)
        k2hist += y*np.sin(beta1[i])
    return k2scale, k2hist, k2


def kappasq_all(sD2, sA2, n=100, m=100):
    k2 = np.zeros((n, m))
    k2scale = np.arange(0, 4, 0.05)
    k2hist = np.zeros(len(k2scale) - 1)
    for i in range(n):
        d1 = np.random.random((m, 3))
        d2 = np.random.random((m, 3))
        for j in range(m):
            delta = np.arccos(np.dot(d1[j, :], d2[j, :]) / linalg.norm(d1[j, :])/linalg.norm(d2[j, :]))
            beta1 = np.arccos(d1[j, 0]/linalg.norm(d1[j, :]))
            beta2 = np.arccos(d2[j, 0]/linalg.norm(d2[j, :]))
            k2[i, j] = kappasq(delta, sD2, sA2, beta1, beta2)
        y, x = np.histogram(k2[i, :], bins=k2scale)
        k2hist += y
    return k2scale, k2hist, k2


def stack_lifetime_spectra(lifetime_spectra, fractions, normalize_fractions=True):
    """
    Takes an array of lifetime spectra and an array of fractions and returns an mixed array of lifetimes
    whereas the amplitudes are multiplied by the fractions. `normalize_fractions` is True the fractions
    are normalized to one.

    :return: numpy-array

    """
    fn = np.array(fractions, dtype=np.float64) / sum(fractions) if normalize_fractions else fractions
    re = []
    for i, ls in enumerate(lifetime_spectra):
        ls = np.copy(ls)
        ls[::2] = ls[::2] * fn[i]
        re.append(ls)
    return np.hstack(re)


def s2delta(r_0, s2donor, s2acceptor, r_inf_AD):
    """calculate delta given residual anisotropies

    :param r_0:
    :param s2donor: -np.sqrt(self.r_Dinf/self.r_0)
    :param s2acceptor: np.sqrt(self.r_Ainf/self.r_0)
    :param r_inf_DA:

    Accurate Distance Determination of Nucleic Acids via Foerster Resonance Energy Transfer:
    Implications of Dye Linker Length and Rigidity
    http://pubs.acs.org/doi/full/10.1021/ja105725e
    """
    delta = r_inf_AD/(r_0*s2donor*s2acceptor)
    return delta


def distribution2rates(distribution, tau0, kappa2, R0, remove_negative=False):
    """
    gets distribution in form: (1,2,3)
    0: number of distribution
    1: amplitude
    2: distance

    returns:
    0: number of dist
    1: amplitude
    2: rate

    :param distribution:
    :param tau0:
    :param kappa2:
    :param R0:
    """

    n_dist, n_ampl, n_points = distribution.shape
    rate_dist = np.copy(distribution)
    if remove_negative:
        np.putmask(rate_dist, rate_dist < 0, 0.0)
    for i in range(n_dist):
        rate_dist[i, 1] = distance2rate(rate_dist[i, 1], kappa2, tau0, R0)
    return rate_dist


def gaussian2rates(means, sigmas, amplitudes, tau0=4.0, kappa2=0.667, R0=52.0,
                   n_points=64, m_sigma=1.5, interleaved=True):
    """
    Calculate distribution of FRET-rates given a list of normal/Gaussian distributed
    distances.

    :param means: array
    :param sigmas: array
    :param amplitudes: array
    :param tau0: float
    :param kappa2: float
    :param R0: float
    :param n_points: int
    :param m_sigma: float

    :return: either an interleaved rate-spectrum, or a 2D-array stack

    Examples
    --------

    This generates an interleaved array here it follows the *rule*:
    amplitude, rate, amplitude, rate, ...

    >>> gaussian2rates([50], [8.0], [1.0], interleaved=True, n_points=8)
    array([ 0.06060222,  1.64238732,  0.10514622,  0.97808684,  0.1518205 ,
            0.60699836,  0.18243106,  0.39018018,  0.18243106,  0.25853181,
            0.1518205 ,  0.17589017,  0.10514622,  0.12247918,  0.06060222,
            0.08706168])

    If *interleaved* is False a 2D-numpy array is returned. The first dimension corresponds
    to the amplitudes the second to the rates.

    >>> gaussian2rates([50], [8.0], [1.0], interleaved=False, n_points=8)
    array([[ 0.06060222,  1.64238732],
           [ 0.10514622,  0.97808684],
           [ 0.1518205 ,  0.60699836],
           [ 0.18243106,  0.39018018],
           [ 0.18243106,  0.25853181],
           [ 0.1518205 ,  0.17589017],
           [ 0.10514622,  0.12247918],
           [ 0.06060222,  0.08706168]])
    """
    means = np.array(means, dtype=np.float64)
    sigmas = np.array(sigmas, dtype=np.float64)
    amplitudes = np.array(amplitudes, dtype=np.float64)
    n_gauss = means.shape[0]

    rates = np.empty((n_gauss, n_points), dtype=np.float64)
    p = np.empty_like(rates)

    for i in range(n_gauss):
        g_min = max(1e-9, means[i] - m_sigma * sigmas[i])
        g_max = means[i] + m_sigma * sigmas[i]
        bins = np.linspace(g_min, g_max, n_points)
        p[i] = np.exp(-(bins - means[i]) ** 2 / (2 * sigmas[i] ** 2))
        p[i] /= np.sum(p[i])
        p[i] *= amplitudes[i]
        rates[i] = distance2rate(bins, kappa2, tau0, R0)

    ls = rates.ravel()
    ps = p.ravel()

    if interleaved:
        return np.dstack((ps, ls)).ravel()
    else:
        return np.dstack((ps, ls))[0]


def rates2lifetimes(rates, donors, x_donly=0.0):
    """
    Converts an interleaved rate spectrum to an interleaved lifetime spectrum
    given an interleaved donor spectrum and the fraction of donor-only

    :param rates: numpy-array
    :param donors: numpy-array
    :param x_donly: float

    """
    n_donors = donors.shape[0]/2
    n_rates = rates.shape[0]/2

    pd, ld = donors.reshape((n_donors, 2)).T
    pr, r = rates.reshape((n_rates, 2)).T

    # allocated memory
    ls = np.empty((n_donors, n_rates), dtype=np.float64)
    ps = np.empty_like(ls)

    x_fret = (1.0 - x_donly)
    ## Quench donor ##
    for i in range(n_donors):
        ls[i] = rate2lifetime(r, ld[i])
        ps[i] = pd[i] * pr * x_fret
    ls = ls.ravel()
    ps = ps.ravel()

    gl = np.dstack((ps, ls)).ravel()
    donor = np.dstack((pd * x_donly, ld)).ravel()
    ds = np.hstack([gl, donor])
    return ds


def elte2(e1, e2):
    """
    Takes two interleaved spectrum of lifetimes (a11, l11, a12, l12,...) and (a21, l21, a22, l22,...)
    and return a new spectrum of lifetimes of the form (a11*a21, 1/(1/l11+1/l21), a12*a22, 1/(1/l22+1/l22), ...)

    :param e1: array-like
        Lifetime spectrum 1
    :param e2: array-like
        Lifetime spectrum 2
    :return: array-like
        Lifetime-spectrum

    Examples
    --------

    >>> import numpy as np
    >>> e1 = np.array([1,2,3,4])
    >>> e2 = np.array([5,6,7,8])
    >>> elte2(e1, e2)
    array([  5.        ,   1.5       ,   7.        ,   1.6       ,
        15.        ,   2.4       ,  21.        ,   2.66666667])
    """
    n1 = e1.shape[0]/2
    a1, l1 = e1.reshape((n1, 2)).T

    n2 = e2.shape[0]/2
    a2, l2 = e2.reshape((n2, 2)).T

    a = np.outer(a1, a2).ravel()
    r = 1. / np.add.outer(1. / l1, 1. / l2).ravel()
    return two_column_to_interleaved(a, r)


def e1tn(e1, n):
    """
    Multiply aplitudes of rate spectrum by float

    :param e1: array-like
        Rate spectrum
    :param n: float

    Examples
    --------

    >>> e1 = np.array([1,2,3,4])
    >>> e1tn(e1, 2.0)
    array([2, 2, 6, 4])
    """
    e1[::2] *= n
    return e1