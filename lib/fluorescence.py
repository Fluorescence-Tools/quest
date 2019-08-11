import numexpr as ne
import numpy as np
import numpy.linalg as linalg
import scipy.optimize


lifetime2transfer = lambda tau, tau0: 1-tau/tau0
transfer2distance = lambda E, R0: (1/E-1)**(1.0/6.0)*R0
transfer2lifetime = lambda E, tau0: (1-E)*tau0
distance2transfer = lambda distance, R0: 1.0/(1.0+(distance/R0)**6)
distance2rate = lambda distance, kappa2, tau0, R0: ne.evaluate("3. / 2. * kappa2 / tau0 * (R0 / distance) ** 6")
rate2lifetime = lambda rate, lifetime: ne.evaluate("1. / (1. / lifetime + rate)")
fretrate2distance = lambda fretrate, R0, tau0: R0 * (fretrate*tau0)**(-1./6)
distance2fretrate = lambda r, R0, tau0: 1./tau0*(R0/r)**6.0
et = lambda fd0, fda: fda / fd0


def transfer_space(e_min, e_max, n_steps, R0=52.0):
    es = np.linspace(e_min, e_max, n_steps)
    rdas = transfer2distance(es, R0)
    return rdas

def kappasq(delta, sD2, sA2, beta1=None, beta2=None):
    """
    Calculate kappa2 given the order parameter sD2 and sA2

    TODO: where equation explain

    :param delta:
    :param sD2: order parameter of donor s2D = - sqrt(r_inf_D/r0)
    :param sA2: order parameter of acceptor s2A = sqrt(r_inf_A/r0)
    :param beta1:
    :param beta2:
    :return:
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


def calc_transfer_matrix(t, rDA_min=1.0, rDA_max=200.0, n_steps=200.0, kappa2=0.667, tau0=4.0, R0=52.0, space='lin'):
    """
    Calculates a matrix converting a distance distribution to an E(t)-decay
    :param t:
    :param rDA_min:
    :param rDA_max:
    :param n_steps:
    :param kappa2:
    :param tau0:
    :param R0:
    :param log_space:
    :return:

    Example
    -------
    >>> t = np.arange(0, 20, 0.0141)
    >>> m, r_da = calc_transfer_matrix(t)
    Now plot the transfer matrix
    >>> import pylab as p
    >>> p.imshow(m)
    >>> p.show()
    """
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
    rates[-1:-10] = 0.0
    m = np.outer(rates, t)
    M = np.nan_to_num(np.exp(-m))
    return M, r_DA


def et2pRDA(ts, et, t_matrix=None, r_DA=None):
    """

    :param t:
    :param et:
    :param t_matrix:
    :param r_DA:
    :return:

    Example
    -------
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
        t_matrix, r_DA = calc_transfer_matrix(ts, 5, 200, 200)

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


def s2delta(r_0, s2donor, s2acceptor, r_inf_AD):
    """
    calculate delta given residual anisotropies
    :param r_0:
    :param s2donor: -np.sqrt(self.r_Dinf/self.r_0)
    :param s2acceptor: np.sqrt(self.r_Ainf/self.r_0)
    :param r_inf_DA:

    Accurate Distance Determination of Nucleic Acids via Foerster Resonance Energy Transfer:
    Implications of Dye Linker Length and Rigidity
    http://pubs.acs.org/doi/full/10.1021/ja105725e
    """
    delta = r_inf_AD/(r_0*s2donor*s2acceptor)
    print("delta: %s" % delta)
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

    nGauss, nAmpl, nPoints = distribution.shape
    rate_dist = np.copy(distribution)
    if remove_negative:
        np.putmask(rate_dist, rate_dist < 0, 0.0)

    for i in range(nGauss):
        rate_dist[i, 1] = distance2rate(rate_dist[i, 1], kappa2, tau0, R0)
    return rate_dist


def gaussian2rates(means, sigmas, amplitudes, tau0=4.0, kappa2=0.667, R0=52.0, n_points=64, m_sigma=1.5, interleaved=True):
    """
    Calculate distribution of transfer-rates given a list of normal/Gaussian distributed
    distances.

    :param means: array
    :param sigmas: array
    :param amplitudes: array
    :param tau0: float
    :param kappa2: float
    :param R0: float
    :param n_points: int
    :param m_sigma: float

    """

    nGauss = len(means)
    rates = np.empty((nGauss, n_points), dtype=np.float64)
    p = np.empty_like(rates)
    for i in range(nGauss):
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
    lifetime spectrum: amplitude, lifetime, amplitude, lifetime, ...
    """
    pd, ld = donors[::2], donors[1::2]
    pr, r = rates[::2], rates[1::2]
    n_donors = len(pd)
    n_rates = len(pr)

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


"""
import numpy as np
import numexpr as ne
import scipy.optimize
import numpy.linalg


rda_mean = [50., 50.0]
rda_sigma = [8, 2.0]
amplitudes = [0.5, 0.5]
rates = gaussian2rates(rda_mean, rda_sigma, amplitudes, interleaved=False)
a = rates[:,0]
kFRET = rates[:,1]
#ts = np.linspace(0.01, 500, 2048)
ts = np.logspace(-3, 3, 2048)
et = np.array([np.dot(a, np.exp(-kFRET * t)) for t in ts])

t_matrix, r_DA = calc_transfer_matrix(ts, 0.1, 120, 256, space='lin')
p.imshow(t_matrix)
p.show()

y1  = numpy.linalg.lstsq(t_matrix.T, et, 5.0e-16)[0]
#y1  = numpy.linalg.solve(t_matrix, et)
p.plot(r_DA, y1)

#y2  = scipy.optimize.nnls(t_matrix.T, et)[0]
#p.plot(r_DA, y2)

p.show()
"""