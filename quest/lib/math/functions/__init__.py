import numpy as np

from . import datatools



def xcorr_fft(
        in_1: np.ndarray,
        in_2: np.ndarray,
        axis: int = 0,
        normalize: bool = True
) -> np.ndarray:
    """Computes the cross-correlation function of two arrays using fast fourier transforms.

    If the ccf could not be computed a numpy array filled with ones is returned.

    :param in_1: a numpy array that is cross-correlated with signal_b
    :param in_2: a numpy array that is cross-correlated with signal_a
    :param normalize: if normalize is True a normalized cross correlation function is returned
    :return: a cross-correlation of the two input signals
    """
    if len(in_1) > 0 and len(in_2) > 0:
        c = in_1
        d = in_2
        n = c.shape[axis]
        m = [slice(None), ] * len(c.shape)

        # Compute the FFT and then (from that) the auto-correlation function.
        f1 = np.fft.fft(c-np.mean(c, axis=axis), n=2*n, axis=axis)
        f2 = np.fft.fft(d-np.mean(d, axis=axis), n=2*n, axis=axis)

        m[axis] = slice(0, n)
        acf = np.fft.ifft(f1 * np.conjugate(f2), axis=axis)[m[axis]].real
        m[axis] = 0
        if normalize:
            return acf / acf[m[axis]]
        else:
            return acf
    else:
        return np.array([], dtype=in_1.dtype)




def autocorr(
        x: np.ndarray,
        axis: int = 0,
        normalize: bool = True
) -> np.array:
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
    if len(x) > 0:
        return xcorr_fft(
            in_1=x,
            in_2=x,
            axis=axis,
            normalize=normalize
        )
    else:
        return np.array([], dtype=x.dtype)

