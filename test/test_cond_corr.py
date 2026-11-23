# CAREFUL, right now cpu and gpu versions are different when data is not all inside range
import math
import importlib.util
from numba import njit
import numpy as np
import pytest
import correlation

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


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
        (1, True, False, "cuda", True),
        (1, False, True, "cuda", True),
    ],
)
def test_cond_corr(threads, cond_start, cond_end, device, fft):
    """Test conditional correlation"""
    use_cpu_fallback = device == "cuda" and fft and not TORCH_AVAILABLE
    if device == "cuda" and not use_cpu_fallback:
        if not TORCH_AVAILABLE:
            pytest.skip("torch is not available")
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available")
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
            trunc=ncorr, traj_range=tuple(range_bins), nbins=nbins, cond_start=cond_start
        )
        if use_cpu_fallback:
            with pytest.warns(UserWarning, match="Falling back to CPU"):
                test_obj.add_data_cuda(a, b, a, dtype="float64")
        elif device == "cuda":
            test_obj.add_data_cuda(a, b, a, dtype="float64")
        elif device == "cpu":
            test_obj.add_data(a, b, a)
        else:
            raise ValueError("device must be cpu or cuda")
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


@pytest.mark.parametrize("cond_start, cond_end", [(True, False), (False, True)])
def test_cond_corr_fft_uses_lag_specific_pair_counts(cond_start, cond_end):
    """FFT conditional correlation normalizes by the actual pairs per lag/bin."""
    a = np.array([2.0, -1.0, 3.0, 4.0, -2.0, 5.0])
    b = np.array([1.5, 2.0, -3.0, 0.5, 4.0, -1.0])
    cond = np.array([0.2, 1.4, 0.4, 1.6, 0.6, 0.8])
    ncorr = 5
    nbins = 2
    range_bins = (0.0, 2.0)
    dx = (range_bins[1] - range_bins[0]) / nbins

    corr_ref = corr(a, b, cond, ncorr, cond_start, cond_end, range_bins[0], dx, nbins)
    test_obj = correlation.ConditionalCorrelationFFT.from_range(
        trunc=ncorr,
        traj_range=range_bins,
        nbins=nbins,
        cond_start=cond_start,
        verbose=False,
    )
    test_obj.add_data(a, b, cond)

    result = test_obj.get_result()
    result[np.isinf(result)] = np.nan

    np.testing.assert_allclose(test_obj.counts_sum, np.rint(test_obj.counts_sum))
    np.testing.assert_allclose(result, corr_ref, atol=1e-10)


if __name__ == "__main__":
    test_cond_corr(threads=1, cond_start=True, cond_end=False, device="cuda", fft=False)
