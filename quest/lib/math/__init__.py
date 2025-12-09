import numpy as np


def xcorr_fft(
    in_1: np.ndarray,
    in_2: np.ndarray,
    axis: int = 0,
    normalize: bool = True,
) -> np.ndarray:
    """Cross-correlation via FFT."""
    if len(in_1) == 0 or len(in_2) == 0:
        return np.array([], dtype=in_1.dtype)

    c = in_1
    d = in_2
    n = c.shape[axis]
    m = [slice(None)] * len(c.shape)

    f1 = np.fft.fft(c - np.mean(c, axis=axis), n=2 * n, axis=axis)
    f2 = np.fft.fft(d - np.mean(d, axis=axis), n=2 * n, axis=axis)

    m[axis] = slice(0, n)
    acf = np.fft.ifft(f1 * np.conjugate(f2), axis=axis)[m[axis]].real
    m[axis] = 0
    return acf / acf[m[axis]] if normalize else acf


def autocorr(
    x: np.ndarray,
    axis: int = 0,
    normalize: bool = True,
) -> np.ndarray:
    """Autocorrelation of a time series using FFT."""
    if len(x) == 0:
        return np.array([], dtype=x.dtype)
    return xcorr_fft(in_1=x, in_2=x, axis=axis, normalize=normalize)


