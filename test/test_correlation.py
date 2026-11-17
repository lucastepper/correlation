import pytest
import numpy as np
import torch
from numba import njit
from correlation import correlation, correlation_torch


@njit
def correlation_ref(a, b, trunc):
    """Calculates correlation directly."""
    if len(a) != len(b):
        raise ValueError("a and b must have the same length")
    output = np.zeros(trunc)
    counter = np.zeros(trunc)
    for tcorr in range(trunc):
        for iframe in range(len(a) - tcorr):
            output[tcorr] += a[iframe] * b[iframe + tcorr]
            counter[tcorr] += 1
    output /= counter
    return output


def correlation_ref_fft(a, b, trunc, bconj=False):
    """Calculates correlation via FFT."""
    len_a = len(a)
    a = np.concatenate([a, np.zeros(len_a)])
    b = np.concatenate([b, np.zeros(len_a)])
    a_fft = np.fft.fft(a)
    b_fft = np.fft.fft(b)
    # which one to conjugate should not matter if data is converged
    if bconj:
        corr = np.fft.ifft(a_fft * np.conj(b_fft))
    else:
        corr = np.fft.ifft(b_fft * np.conj(a_fft))
    corr = corr[:trunc].real
    corr /= np.linspace(len_a, len_a - trunc + 1, trunc)
    return corr


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
correlation_devices = [
    (False, "cpu"),
    (True, "cpu"),
    pytest.param(True, "cuda", marks=cuda_only),
]


@pytest.mark.parametrize("test_torch, device", correlation_devices)
def test_autocorr_1d(test_torch, device):
    """Test computation of autocorrelation."""
    size = int(1e3 + 1)
    np.random.seed(42)
    time = np.arange(size) * 0.02
    data = np.sin(time) + np.random.normal(scale=0.1, size=size)
    data_copy = np.copy(data)
    if test_torch:
        corr_test = correlation_torch(data, trunc=size, device=device)
    else:
        corr_test = correlation(data, trunc=size)
    np.testing.assert_allclose(corr_test, correlation_ref(data, data, size))
    np.testing.assert_allclose(corr_test, correlation_ref_fft(data, data, size))
    # somehow, setting overwrite in correlation does not corrupt input data
    np.testing.assert_allclose(data, data_copy)


@pytest.mark.parametrize("test_torch, device", correlation_devices)
def test_corr_1d(test_torch, device):
    """Test computation of correlation."""
    size = int(1e4 + 1)
    np.random.seed(42)
    time = np.arange(size) * 0.02
    data1 = np.sin(time) + np.random.normal(scale=0.01, size=size)
    data1_copy = np.copy(data1)
    data2 = np.sin(time) + np.random.normal(scale=0.01, size=size)
    data2_copy = np.copy(data2)
    if test_torch:
        corr_test = correlation_torch(data1, data2, trunc=size, device=device)
    else:
        corr_test = correlation(data1, data2, trunc=size)
    np.testing.assert_allclose(corr_test, correlation_ref(data1, data2, size))
    np.testing.assert_allclose(corr_test, correlation_ref_fft(data1, data2, size))
    # looks like conjugating a is the way to go, numerically
    assert not np.allclose(corr_test, correlation_ref_fft(data1, data2, size, bconj=True))
    np.testing.assert_allclose(data1, data1_copy)
    np.testing.assert_allclose(data2, data2_copy)


@pytest.mark.parametrize("test_torch, device", correlation_devices)
def test_autocorr_2d(test_torch, device):
    """Test computation of correlation in 2d."""
    size = (3, int(1e3 + 11))
    np.random.seed(42)
    time = np.arange(size[1]) * 0.02
    data = np.sin(time) + np.random.normal(scale=0.1, size=size)
    data_copy = np.copy(data)
    ref = np.zeros_like(data)
    for i, dat in enumerate(data):
        if test_torch:
            ref[i] = correlation_torch(dat, trunc=size[1], device=device)
        else:
            ref[i] = correlation_ref(dat, dat, data.shape[1])
    np.testing.assert_allclose(correlation(data, data), ref)
    np.testing.assert_allclose(data, data_copy)


@pytest.mark.parametrize("test_torch, device", correlation_devices)
def test_corr_2d(test_torch, device):
    """Test computation of correlation in 2d."""
    size = (3, int(1e3 + 11))
    np.random.seed(42)
    time = np.arange(size[1]) * 0.02
    data1 = np.sin(time) + np.random.normal(scale=0.1, size=size)
    data1_copy = np.copy(data1)
    data2 = np.sin(time) + np.random.normal(scale=0.1, size=size)
    data2_copy = np.copy(data2)
    ref = np.zeros_like(data1)
    for i, (dat1, dat2) in enumerate(zip(data1, data2)):
        if test_torch:
            ref[i] = correlation_torch(dat1, dat2, trunc=size[1], device=device)
        else:
            ref[i] = correlation_ref(dat1, dat2, data1.shape[1])
    np.testing.assert_allclose(correlation(data1, data2), ref)
    np.testing.assert_allclose(data1, data1_copy)
    np.testing.assert_allclose(data2, data2_copy)
