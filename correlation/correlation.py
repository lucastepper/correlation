import numpy as np
import mkl_fft


def mkl_res_conj_inplace(x):
    """ The resulting array of the intel_mkl.fft inplace function
    (overwrite_x=True) writes the results in the form
    Real(Nyquist freq), half the spectrum, Real(zero freq).
    This is possible due to the hermetian spectrum of real data.
    To complex conjugate this we want to conj all but the
    first and last element.
    Arguments:
        real valued np.ndarray, output of mkl.rfft
    Returns:
        pointer to input array """

    x[1:-1] = np.conj(x[1:-1].view(np.complex128)).view(np.float64)
    return x


def mkl_res_mult_inplace(x, y):
    """ The resulting array of the intel_mkl.fft inplace function
    (overwrite_x=True) writes the results in the form
    Real(Nyquist freq), half the spectrum, Real(zero freq).
    This is possible due to the hermetian spectrum of real data.
    To multiply two arrays, we need to do complex multiplication
    for spectrum and pointwise for first and last elem.
    Arguments:
        (x) real valued np.ndarray, output of mkl.rfft
        (y) real valued np.ndarray, output of mkl.rfft
    Returns:
        (x) real valued np.ndarray, output of mkl.rfft,
            written in input x """

    x[1:-1] = (x[1:-1].view(np.complex128) * y[1:-1].view(np.complex128)).view(np.float64)
    x[0] *= y[0]
    x[-1] *= y[-1]
    return x


def correlation(a, b, trunc=None, overwrite=False):
    """ Calculates correlation via FFT. Only tested for 1d data.
    Zero pads the data. Zero padding twice makes no difference,
    so we do not check if it has been done already. We check
    if we compute ACF by id(a) == id(b) and only single fft
    in that case. If trunc not provided,returns len(a) points,
    else trunc. Only works for real data, as it uses mkl.rfft.
    Arguments:
        a (np.ndarray, real): input 1
        b (np.ndarray, real): input 2
        trunc (int): number of out elements to keep
        overwrite (bool): Overwrite input
    Returns (np.ndarray): correlation """

    assert isinstance(a, np.ndarray)
    assert isinstance(b, np.ndarray)
    assert a.ndim == 1
    assert a.shape == b.shape
    assert np.isreal(a.all())
    assert np.isreal(b.all())
    assert isinstance(trunc, (int, type(None)))

    if not trunc:
        trunc = len(a)
    trunc = min(trunc, len(a))
    len_a = len(a)

    a = np.append(a, np.zeros(len_a))
    a = mkl_fft.rfft(a, n=None, axis=-1, overwrite_x=overwrite)
    if b is None:
        a1 = mkl_res_conj_inplace(np.copy(a))
        a = mkl_res_mult_inplace(a, a1)
        del a1
    else:
        b = np.append(b, np.zeros(len_a))
        b = mkl_fft.rfft(b, n=None, axis=-1, overwrite_x=overwrite)
        a = mkl_res_conj_inplace(a)
        a = mkl_res_mult_inplace(a, b)
    a = mkl_fft.irfft(a, n=None, axis=-1, overwrite_x=overwrite)[:trunc]
    a /= np.linspace(len_a, len_a - trunc + 1, trunc)
    return a
