import math
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
        ncorr: int
            Number of correlation points to compute.
        intervals: np.ndarray
            Array of shape (nintervals, 1) with the start and end of each
            interval.
    Computes:
        ----------
        conditional correlation (np.ndarray(len(intervals - 1), ncorr))
        """

    def __init__(self, ncorr: int, intervals: np.ndarray, cond_start=True, verbose=True) -> None:
        self.ncorr = ncorr
        self.intervals = intervals
        self.nintervals = intervals.shape[0]
        self.start = intervals[0]
        self.nbins = intervals.shape[0] - 1
        self.dx = intervals[1] - intervals[0]
        self.corr = np.zeros((self.nintervals, self.ncorr))
        self.counts = np.zeros((self.nintervals, 1))
        self.cond_start = cond_start
        self.verbose = verbose
        self.counts_sum = np.zeros((self.nbins, 1))
        self.cond_fft_sum = np.zeros((self.nbins, self.ncorr))

    @classmethod
    def from_dx(cls, dx: float, nbins: int, start: float, ncorr: int):
        """ Build ConditionalCorrelationFFT from dx, start and nbins.
        Arguments:
            dx: float, Width of the intervals for the cond corr.
            nbins: int, Number of bins for cond_corr (Meaning nbins + 1 intervals).
            start: float, Left-most inverval edge.
            ncorr: int, Number of correlation points to compute.
        Returns:
            ConditionalCorrelationFFT
        """
        intervals = np.arange(nbins + 1) * dx + start
        return cls(ncorr, intervals)

    @classmethod
    def from_range(cls, nbins: int, traj_range: tuple, ncorr: int):
        """ Build ConditionalCorrelationFFT from dx, start and nbins.
        Arguments:
            nbins: int, Number of bins for cond_corr (Meaning nbins + 1 intervals).
            traj_range: [float; 2], Left-most and right-most inverval edge.
            ncorr: int, Number of correlation points to compute.
        Returns:
            ConditionalCorrelationFFT
        """
        intervals = np.linspace(*traj_range, nbins + 1)
        return cls(ncorr, intervals)


    def add_data(self, data1, data2, cond):
        """Compute conditional correlation using FFT, by multipling with indicator
        functions for each bin at time t(0).
        """
        cond_corr = np.zeros((self.nbins, self.ncorr))
        counts = np.zeros((self.nbins, 1))
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
            cond_corr[ibin, :] = norm * correlation.correlation(cdata1, cdata2, trunc=self.ncorr)
            counts[ibin] = count
        self.counts_sum += counts
        self.cond_fft_sum += counts * cond_corr

    def get_result(self):
        return self.cond_fft_sum / self.counts_sum
