# CAREFUL, right now cpu and gpu versions are different when data is not all inside range
import math
from numba import njit
import numpy as np
import pytest
import correlation


@njit
def get_index(x, x_min, dx):
    """Get the index of the bin where x belongs to"""
    return math.floor((x - x_min) / dx)


@njit
def corr(a, b, cond, ncorr, cond_start, cond_end, x_min, dx, nbins):
    """Compute a conditional correlation between a and b."""
    if cond_start and cond_end:
        raise ValueError("cond_start and cond_end cannot be both True")
    cond_sum = np.zeros((ncorr, nbins))
    cond_count = np.zeros((ncorr, nbins))
    for tcorr in range(0, ncorr):
        for i in range(0, len(a) - tcorr):
            if cond_start:
                idx = get_index(cond[i], x_min, dx)
            else:
                idx = get_index(cond[i + tcorr], x_min, dx)
            if idx >= 0 and idx < nbins:
                cond_sum[tcorr, idx] += a[i] * b[i + tcorr]
                cond_count[tcorr, idx] += 1.0
    return cond_sum / cond_count


@pytest.mark.parametrize(
    "threads, cond_start, cond_end, device, fft",
    [
        (1, True, False, "cpu", False),
        (10, True, False, "cpu", False),
        (1, False, True, "cpu", False),
        (10, False, True, "cpu", False),
        (1, True, False, "cuda", False),
        (1, True, False, "cpu", True),
        (1, False, True, "cpu", True),
    ],
)
def test_cond_corr(threads, cond_start, cond_end, device, fft):
    """Test conditional correlation"""
    np.random.seed(43)
    n = 10000
    ncorr = 20
    nbins = 4
    range_bins = [-1.4, 1.4]
    dx = (range_bins[1] - range_bins[0]) / nbins
    a = np.random.randn(n)
    b = np.random.randn(n)
    # compute ref
    corr_ref = corr(a, b, a, ncorr, cond_start, cond_end, range_bins[0], dx, nbins)
    # compute test
    if device == "cuda":
        a = a.astype(np.float32)
        b = b.astype(np.float32)
    if fft:
        test_obj = correlation.ConditionalCorrelationFFT.from_range(
            ncorr=ncorr, traj_range=tuple(range_bins), nbins=nbins, cond_start=cond_start
        )
        test_obj.add_data(a, b, a)
    else:
        test_obj = correlation.ConditionalCorrelation(
            ncorr, nbins, range_bins, cond_start, cond_end, device
        )
        test_obj.add_data(a, b, a, threads)
    atol = 8e-8
    if fft:
        # fft is a bit less precise when dataset so small
        atol = 1e-4
    # compare
    np.testing.assert_allclose(test_obj.get_result(), corr_ref, atol=atol)


if __name__ == "__main__":
    test_cond_corr(threads=1, cond_start=True, cond_end=False, device="cuda", fft=False)
