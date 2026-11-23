import math
import time
import warnings
from typing import Union
from matplotlib import pyplot as plt
import numpy as np
from numba import njit
import tqdm
from .correlation_mkl import correlation

try:
    import torch
except ImportError:
    torch = None

    def correlation_torch(*args, **kwargs):
        raise ImportError(
            "correlation_torch requires PyTorch. Install it with 'pip install torch'."
        )
else:
    from .correlation_torch import correlation_torch, string_to_torch_dtype


@njit
def get_index(x, x_min, dx):
    """Get index of the bin that x belongs to."""
    return math.floor((x - x_min) / dx)


@njit
def get_cond_data(data, cond, ibin, dx, x_min):
    """Zero all frames of data that are not in the ibin-th bin."""
    cond_data = np.zeros_like(data)
    indicator = np.zeros(len(data), dtype=np.float64)
    count = 0
    for i in range(len(data)):
        idx = get_index(cond[i], x_min, dx)
        if idx == ibin:
            cond_data[i] = data[i]
            indicator[i] = 1.0
            count += 1
    return cond_data, indicator, count


class ConditionalCorrelationFFT:
    """
    Iteratively add data and compute conditional correlation
    functions via FFT.
    Arguments:
        ----------
        trunc: int Number of correlation points to compute.
        intervals: np.ndarray Array of shape (nintervals) with the start and end of each interval.
        cond_start: bool, If True, the conditional correlation is conditioned on x[0],
            if False, it is conditioned on x[tcorr].
        verbose: bool, If True, print progress bar.
    Computes:
        ----------
        conditional correlation (trunc, np.ndarray(len(intervals - 1)))
    """

    def __init__(
        self,
        trunc: int,
        intervals: np.ndarray,
        cond_start: bool = True,
        verbose: bool = True,
    ) -> None:
        self.trunc = trunc
        self.intervals = intervals
        self.nintervals = intervals.shape[0]
        self.start = intervals[0]
        self.nbins = intervals.shape[0] - 1
        self.dx = intervals[1] - intervals[0]
        self.cond_start = cond_start
        self.verbose = verbose
        self.counts_sum = np.zeros((self.trunc, self.nbins))
        self.cond_fft_sum = np.zeros((self.trunc, self.nbins))

    @classmethod
    def from_dx(
        cls,
        dx: float,
        nbins: int,
        start: float,
        trunc: int,
        cond_start: bool = True,
        verbose: bool = True,
    ):
        """Build ConditionalCorrelationFFT from dx, start and nbins.
        Arguments:
            dx: float, Width of the intervals for the cond corr.
            nbins: int, Number of bins for cond_corr (Meaning nbins + 1 intervals).
            start: float, Left-most inverval edge.
            trunc: int, Number of correlation points to compute.
            cond_start: bool, If True, the conditional correlation is conditioned on x[0],
                if False, it is conditioned on x[tcorr].
            verbose: bool, If True, print progress bar.
        Returns:
            ConditionalCorrelationFFT
        """
        intervals = np.arange(nbins + 1) * dx + start
        return cls(trunc, intervals, cond_start=cond_start, verbose=verbose)

    @classmethod
    def from_range(
        cls,
        nbins: int,
        traj_range: tuple[Union[float, int]],
        trunc: int,
        cond_start: bool = True,
        verbose: bool = True,
    ):
        """Build ConditionalCorrelationFFT from dx, start and nbins.
        Arguments:
            nbins: int, Number of bins for cond_corr (Meaning nbins + 1 intervals).
            traj_range: [float; 2], Left-most and right-most inverval edge.
            trunc: int, Number of correlation points to compute.
            cond_start: bool, If True, the conditional correlation is conditioned on x[0],
                if False, it is conditioned on x[tcorr].
            verbose: bool, If True, print progress bar.
        Returns:
            ConditionalCorrelationFFT
        """
        intervals = np.linspace(*traj_range, nbins + 1)
        return cls(trunc, intervals, cond_start=cond_start, verbose=verbose)

    def check_traj_length(self, traj):
        """Check that is the trajectory is at least as long as self.trunc.
        If not, shorten self.trunc and self.cond_fft_sum.
        """
        if len(traj) < self.trunc:
            self.trunc = len(traj)
            self.counts_sum = self.counts_sum[: self.trunc, :]
            self.cond_fft_sum = self.cond_fft_sum[: self.trunc, :]

    def precheck_add_data(self, data1: np.ndarray, data2: np.ndarray, cond: np.ndarray):
        """Check that the dims of all trajs are 1, check that all trajs are longer than self.trunc
        or else shorten self.trunc finally check that all trajs have same length.
        """
        if data1.ndim != 1 or data2.ndim != 1 or cond.ndim != 1:
            raise ValueError(
                f"Data must be 1-dimensional, but found {data1.ndim=}, {data2.ndim=}, {cond.ndim=}"
            )
        # shorten self.trunc and self.cond_fft_sum if traj is too short
        for traj in [data1, data2, cond]:
            self.check_traj_length(traj)
        if len(data1) != len(data2) or len(data1) != len(cond):
            raise ValueError(
                f"Trajs need same length, but found {len(data1)=}, {len(data2)=}, {len(cond)=}"
            )

    def add_data(
        self, data1: np.ndarray, data2: np.ndarray, cond: np.ndarray, pbar=None
    ):
        """Compute conditional correlation using FFT, by multipling with indicator
        functions for each bin at time t(0). Assumes that input data is 1-dimensional.
        """
        self.precheck_add_data(data1, data2, cond)
        # compute conditional correlation
        cond_sums = np.zeros((self.trunc, self.nbins))
        counts = np.zeros((self.trunc, self.nbins))
        lag_counts = np.linspace(len(data1), len(data1) - self.trunc + 1, self.trunc)
        ones = np.ones(len(data1), dtype=np.float64)
        # check if we need to use pbar or it was provided
        if pbar is None:
            iterator = tqdm.tqdm(
                range(self.nbins), disable=not self.verbose, leave=False
            )
        else:
            iterator = range(self.nbins)
        # iterate over bins
        for ibin in iterator:
            if self.cond_start:
                cdata1, indicator, count = get_cond_data(
                    data1, cond, ibin, self.dx, self.start
                )
                cdata2 = data2
                count_corr = correlation(indicator, ones, trunc=self.trunc)
            else:
                cdata2, indicator, count = get_cond_data(
                    data2, cond, ibin, self.dx, self.start
                )
                cdata1 = data1
                count_corr = correlation(ones, indicator, trunc=self.trunc)
            # only update the bin if we have found any data
            if count > 0:
                cond_sums[:, ibin] = (
                    correlation(cdata1, cdata2, trunc=self.trunc) * lag_counts
                )
                counts[:, ibin] = np.rint(count_corr * lag_counts)
            if pbar is not None:
                pbar.update(1)
        self.counts_sum += counts
        self.cond_fft_sum += cond_sums

    def add_data_cuda(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        cond: np.ndarray,
        dtype="float32",
        pbar=None,
    ):
        """Compute conditional correlation using FFT on gpu with torch, by multipling with indicator
        functions for each bin at time t(0). Assumes that input data is 1-dimensional.
        Falls back to add_data if PyTorch is unavailable.
        """
        if torch is None:
            warnings.warn(
                "PyTorch is unavailable. Falling back to CPU conditional correlation.",
                stacklevel=2,
            )
            return self.add_data(data1, data2, cond, pbar=pbar)

        self.precheck_add_data(data1, data2, cond)
        dtype = string_to_torch_dtype(dtype)
        # compute conditional correlation
        data1 = torch.Tensor(data1).to(dtype=dtype).cuda()
        data2 = torch.Tensor(data2).to(dtype=dtype).cuda()
        cond = torch.Tensor(cond).to(dtype=dtype).cuda()
        idxs = torch.floor((cond - self.start) / self.dx).to(dtype=torch.long)
        temp_array = torch.zeros_like(data1)

        # output arrays
        cond_sums = torch.zeros((self.trunc, self.nbins), dtype=dtype, device="cuda")
        counts = torch.zeros((self.trunc, self.nbins), dtype=dtype, device="cuda")
        lag_counts = torch.linspace(
            len(data1),
            len(data1) - self.trunc + 1,
            self.trunc,
            dtype=dtype,
            device="cuda",
        )
        ones = torch.ones_like(data1)
        # check if we need to use pbar or it was provided
        if pbar is None:
            iterator = tqdm.tqdm(
                range(self.nbins), disable=not self.verbose, leave=False
            )
        else:
            iterator = range(self.nbins)
        # iterate over bins
        for ibin in iterator:
            temp_array.fill_(0)
            mask = idxs == ibin
            indicator = mask.to(dtype=dtype)
            if self.cond_start:
                temp_array[mask] = data1[mask]
                cdata1 = temp_array
                cdata2 = data2
                count_corr = correlation_torch(
                    indicator,
                    ones,
                    trunc=self.trunc,
                    dtype=dtype,
                    device="cuda",
                    device_out="cuda",
                )
            else:
                temp_array[mask] = data2[mask]
                cdata1 = data1
                cdata2 = temp_array
                count_corr = correlation_torch(
                    ones,
                    indicator,
                    trunc=self.trunc,
                    dtype=dtype,
                    device="cuda",
                    device_out="cuda",
                )
            # only update the bin if we have found any data
            if torch.any(mask):
                corr = correlation_torch(
                    cdata1,
                    cdata2,
                    trunc=self.trunc,
                    dtype=dtype,
                    device="cuda",
                    device_out="cuda",
                )
                cond_sums[:, ibin] = corr * lag_counts
                counts[:, ibin] = torch.round(count_corr * lag_counts)
            if pbar is not None:
                pbar.update(1)
        self.counts_sum += counts.cpu().numpy()
        self.cond_fft_sum += cond_sums.cpu().numpy()

    def get_result(self):
        """Gather the resulting conditional correlation function from all data added.
        Returns:
            cond_corr (np.ndarray(trunc, nbins)): The conditional correlation function.
        """
        return self.cond_fft_sum / self.counts_sum


def correlation_2d(
    data1: np.ndarray,
    data2: np.ndarray,
    cond_data: np.ndarray,
    intervals: np.ndarray,
    trunc: int,
    cond_start: bool = True,
    device: str = "cpu",
):
    """Compute 2d correlation function for a 1d data or a set of 1d datas.
    Arguments:
        data1 np.ndarray(ntrajs, nframes) or np.ndarray(nframes): The first data to correlate.
        data2 np.ndarray(ntrajs, nframes) or np.ndarray(nframes): The second data to correlate.
        cond_data np.ndarray(ntrajs, nframes) or np.ndarray(nframes): The data to condition on.
        intervals np.ndarray(nbins + 1): The intervals in which to compute the cond correlations.
        trunc int: The number of correlation points to compute.
        cond_start bool: If True, the conditional correlation is conditioned on data1, else data2
        device str: The device to use, either "cpu" or "cuda".
    Returns:
        cond_corr np.ndarray(trunc, nbins): The conditional correlation function.
    """
    # check shapes and cast to numpy array
    data1 = np.array(data1)
    if data1.ndim == 1:
        data1 = data1[np.newaxis, :]
    if data1.ndim != 2:
        raise ValueError(f"{data1=} must be 1 or 2-dimensional")
    data2 = np.array(data2)
    if data2.ndim == 1:
        data2 = data2[np.newaxis, :]
    if data2.ndim != 2:
        raise ValueError(f"{data2=} must be 1 or 2-dimensional")
    cond_data = np.array(cond_data)
    if cond_data.ndim == 1:
        cond_data = cond_data[np.newaxis, :]
    if cond_data.ndim != 2:
        raise ValueError(f"{cond_data=} must be 1 or 2-dimensional")
    intervals = np.array(intervals)
    if intervals.ndim != 1:
        raise ValueError(f"{intervals=} must be 1-dimensional")

    # build ConditionalCorrelationFFT object
    cond_corr_obj = ConditionalCorrelationFFT(
        trunc, intervals, cond_start=cond_start, verbose=False
    )
    # add data
    for dat1, dat2, cond in zip(data1, data2, cond_data):
        if device == "cpu":
            cond_corr_obj.add_data(dat1, dat2, cond, pbar=None)
        elif device == "cuda":
            cond_corr_obj.add_data_cuda(dat1, dat2, cond, pbar=None)

    return cond_corr_obj.get_result()
