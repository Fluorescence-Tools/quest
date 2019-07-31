__author__ = 'thomas'
import numpy as np

windowTypes = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']


def window(data, window_len, window='bartlett'):
    """
    smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    :param data: 1D numpy-array (data)
    :param window_len: the dimension of the smoothing window; should be an odd integer
    :param window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'\
                   flat window will produce a moving average smoothing.
    :return: 1D numpy-array (smoothed data)

    Examples
    --------


    """

    if data.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if data.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        return data

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[2 * data[0] - data[window_len:1:-1], data, 2 * data[-1] - data[-1:-window_len:-1]]

    if window == 'flat': # moving average
        w = np.ones(window_len,'d')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len-1:-window_len+1]


def autocorr(x, axis=0, fast=False):
    """
    Estimate the autocorrelation function of a time series using the FFT.

    :param x:
        The time series. If multidimensional, set the time axis using the
        ``axis`` keyword argument and the function will be computed for every
        other axis.

    :param axis: (optional)
        The time axis of ``x``. Assumed to be the first axis if not specified.

    :param fast: (optional)
        If ``True``, only use the largest ``2^n`` entries for efficiency.
        (default: False)

    """
    x = np.atleast_1d(x)
    m = [slice(None), ] * len(x.shape)

    # For computational efficiency, crop the chain to the largest power of
    # two if requested.
    if fast:
        n = int(2**np.floor(np.log2(x.shape[axis])))
        m[axis] = slice(0, n)
        x = x
    else:
        n = x.shape[axis]

    # Compute the FFT and then (from that) the auto-correlation function.
    f = np.fft.fft(x-np.mean(x, axis=axis), n=2*n, axis=axis)
    m[axis] = slice(0, n)
    acf = np.fft.ifft(f * np.conjugate(f), axis=axis)[m].real
    m[axis] = 0

    return acf / acf[m]