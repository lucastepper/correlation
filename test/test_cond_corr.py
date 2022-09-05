# CAREFUL, right now cpu and gpu versions are different when data is not all inside range
import math
from numba import njit
import numpy as np
import correlation
import pytest
np.random.seed(43)


@njit
def get_index(x, x_min, dx):
    return math.floor((x - x_min) / dx)


@njit
def corr(a, b, cond, ncorr, cond_start, cond_end, x_min, dx, nbins):
    if cond_start and cond_end:
        raise ValueError("cond_start and cond_end cannot be both True")
    cond_sum = np.zeros((nbins, ncorr))
    cond_count = np.zeros((nbins, ncorr))
    for tcorr in range(0, ncorr):
        for i in range(0, len(a) - tcorr):
            if cond_start:
                idx = get_index(cond[i], x_min, dx)
            else:
                idx = get_index(cond[i + tcorr], x_min, dx)
            if idx >= 0 and idx < nbins:
                cond_sum[idx, tcorr] += a[i] * b[i + tcorr]
                cond_count[idx, tcorr] += 1.0
    return cond_sum / cond_count


@pytest.mark.parametrize("threads, cond_start, cond_end, device", [
    (1, True, False, "cpu"),
    (10, True, False, "cpu"),
    (1, False, True, "cpu"),
    (10, False, True, "cpu"),
    (1, True, False, "cuda"),
])
def test_cond_corr(threads, cond_start, cond_end, device):
    # Generate data
    n = 1000
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
    test_obj = correlation.ConditionalCorrelation(ncorr, nbins, range_bins, cond_start, cond_end, device)
    test_obj.add_data(a, b, a, threads)
    # compare
    np.testing.assert_allclose(test_obj.get_result(), corr_ref, atol=8e-8)

if __name__ == "__main__":
    test_cond_corr(threads=1, cond_start=True, cond_end=False, device="cuda")
