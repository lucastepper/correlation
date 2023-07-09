import math
from typing import Union
import numpy as np
from numba import njit
import tqdm
import correlation


@njit
def get_index(x, x_min, dx):
    """Get index of the bin that x belongs to."""
    return math.floor((x - x_min) / dx)


@njit
def get_cond_data(data, cond, ibin, dx, x_min):
    """Zero all frames of data that are not in the ibin-th bin."""
    cond_data = np.zeros_like(data)
    count = 0
    for i in range(len(data)):
        idx = get_index(cond[i], x_min, dx)
        if idx == ibin:
            cond_data[i] = data[i]
            count += 1
    return cond_data, count


class ConditionalCorrelationFFT:
    """
    Iteratively add data and compute conditional correlation
    functions via FFT.
    Arguments:
        ----------
        ncorr: int Number of correlation points to compute.
        intervals: np.ndarray Array of shape (nintervals) with the start and end of each interval.
        cond_start: bool, If True, the conditional correlation is conditioned on x[0],
            if False, it is conditioned on x[tcorr].
        verbose: bool, If True, print progress bar.
    Computes:
        ----------
        conditional correlation (ncorr, np.ndarray(len(intervals - 1)))
    """

    def __init__(
        self, ncorr: int, intervals: np.ndarray, cond_start: bool = True, verbose: bool = True
    ) -> None:
        self.ncorr = ncorr
        self.intervals = intervals
        self.nintervals = intervals.shape[0]
        self.start = intervals[0]
        self.nbins = intervals.shape[0] - 1
        self.dx = intervals[1] - intervals[0]
        self.cond_start = cond_start
        self.verbose = verbose
        self.counts_sum = np.zeros((1, self.nbins))
        self.cond_fft_sum = np.zeros((self.ncorr, self.nbins))

    @classmethod
    def from_dx(
        cls,
        dx: float,
        nbins: int,
        start: float,
        ncorr: int,
        cond_start: bool = True,
        verbose: bool = True,
    ):
        """Build ConditionalCorrelationFFT from dx, start and nbins.
        Arguments:
            dx: float, Width of the intervals for the cond corr.
            nbins: int, Number of bins for cond_corr (Meaning nbins + 1 intervals).
            start: float, Left-most inverval edge.
            ncorr: int, Number of correlation points to compute.
            cond_start: bool, If True, the conditional correlation is conditioned on x[0],
                if False, it is conditioned on x[tcorr].
            verbose: bool, If True, print progress bar.
        Returns:
            ConditionalCorrelationFFT
        """
        intervals = np.arange(nbins + 1) * dx + start
        return cls(ncorr, intervals, cond_start=cond_start, verbose=verbose)

    @classmethod
    def from_range(
        cls,
        nbins: int,
        traj_range: tuple[Union[float, int]],
        ncorr: int,
        cond_start: bool = True,
        verbose: bool = True,
    ):
        """Build ConditionalCorrelationFFT from dx, start and nbins.
        Arguments:
            nbins: int, Number of bins for cond_corr (Meaning nbins + 1 intervals).
            traj_range: [float; 2], Left-most and right-most inverval edge.
            ncorr: int, Number of correlation points to compute.
            ncorr: int, Number of correlation points to compute.
            cond_start: bool, If True, the conditional correlation is conditioned on x[0],
                if False, it is conditioned on x[tcorr].
            verbose: bool, If True, print progress bar.
        Returns:
            ConditionalCorrelationFFT
        """
        intervals = np.linspace(*traj_range, nbins + 1)
        return cls(ncorr, intervals, cond_start=cond_start, verbose=verbose)

    def check_traj_lenght(self, traj):
        """Check that is the trajectory is at least as long as self.ncorr.
        If not, shorten self.ncorr and self.cond_fft_sum.
        """
        if len(traj) < self.ncorr:
            self.ncorr = len(traj)
            self.cond_fft_sum = self.cond_fft_sum[: self.ncorr, :]

    def add_data(self, data1: np.ndarray, data2: np.ndarray, cond: np.ndarray):
        """Compute conditional correlation using FFT, by multipling with indicator
        functions for each bin at time t(0). Assumes that input data is 1-dimensional.
        """
        if data1.ndim != 1 or data2.ndim != 1 or cond.ndim != 1:
            raise ValueError(
                f"Data must be 1-dimensional, but found {data1.ndim=}, {data2.ndim=}, {cond.ndim=}"
            )
        # shorten self.ncorr and self.cond_fft_sum if traj is too short
        for traj in [data1, data2, cond]:
            self.check_traj_lenght(traj)
        # compute conditional correlation
        cond_corr = np.zeros((self.ncorr, self.nbins))
        counts = np.zeros((1, self.nbins))
        for ibin in tqdm.tqdm(range(self.nbins), disable=not self.verbose, leave=False):
            if self.cond_start:
                cdata1, count = get_cond_data(data1, cond, ibin, self.dx, self.start)
                cdata2 = data2
            else:
                cdata2, count = get_cond_data(data2, cond, ibin, self.dx, self.start)
                cdata1 = data1
            if count > 0:
                norm = len(data1) / count
            else:
                norm = np.nan
            cond_corr[:, ibin] = norm * correlation.correlation(cdata1, cdata2, trunc=self.ncorr)
            counts[:, ibin] = count
        self.counts_sum += counts
        self.cond_fft_sum += counts * cond_corr

    def get_result(self):
        """Gather the resulting conditional correlation function from all data added.
        Returns:
            cond_corr (np.ndarray(ncorr, nbins)): The conditional correlation function.
        """
        return self.cond_fft_sum / self.counts_sum
