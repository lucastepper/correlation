import numpy as np
from beartype import beartype
from beartype.typing import Callable, Union, Optional
import matplotlib.pyplot as plt
from correlation import correlation
import gle_sim


class IterativeCorrelation:
    """Module to iteratively add data and compute a time correlation function
    as the mean over all data added. Uses fft to compute the correlation.
    Optionally, pass conversion functions for argument 1 and 2 of correlation. TBD"""

    @beartype
    def __init__(
        self,
        conv_func_1: Optional[Callable] = None,
        conv_func_2: Optional[Callable] = None,
        trunc: Optional[int] = None,
        keep_corrs=False,
    ):
        """Constructor for IterativeCorrelation. Pass conversion functions for argument 1 and 2 of correlation.
        When they are None, the data passed to self.add_data is correlated unchanged."""

        self.trunc = trunc
        self.conv_func_1 = self.parse_conv_func(conv_func_1)
        self.conv_func_2 = self.parse_conv_func(conv_func_2)
        self.sum_corr = np.zeros(trunc)
        self.corrs = []
        self.keep_corrs = keep_corrs
        self.n_corr = 0

    @beartype
    def parse_conv_func(self, conv_func: Optional[Callable]):
        """Check if obs is a function returning a np.ndarray or None."""
        if conv_func is None:
            return lambda x: x
        # check if function that returns ndarray
        if not isinstance(conv_func(np.random.uniform(size=10)), np.ndarray):
            raise ValueError(f"conv_func: '{conv_func}' does not return a numpy array")
        return conv_func

    def add_data(self, data1: np.ndarray, data2: Optional[np.ndarray] = None):
        """Add data bit by bit."""
        # check shapes of data, allow 1d and 2d
        if data1.ndim == 1:
            data1 = data1.reshape(1, -1)
        if data2 is not None:
            if data2.ndim == 1:
                data2 = data2.reshape(1, -1)
            if data1.shape[1] != data2.shape[1]:
                # TODO add message
                raise ValueError()
        if data1.shape[1] < self.trunc:
            self.trunc = data1.shape[1]
            self.corrs = [corr[:self.trunc] for corr in self.corrs]
            self.sum_corr = self.sum_corr[:self.trunc]
        # iterate over data
        for i, dat1 in enumerate(data1):
            dat1 = self.conv_func_1(dat1)
            # compute autocorrelation if data2 is None
            # TODO if we have conversion function 2 defined, we should be able to only pass data1
            # and have dat1 be conv1(data1[i]) and dat2 be conv2(data1[i])
            if data2 is not None:
                dat2 = self.conv_func_2(data2[i])
            else:
                dat2 = dat1
            corr = correlation(dat1, dat2, self.trunc)
            self.sum_corr += corr
            if self.keep_corrs:
                self.corrs.append(corr)
            self.n_corr += 1

    def set_corr(self, corr):
        self.sum_corr = corr
        self.n_corr = 1
        self.corrs = []

    def get_result(self):
        return self.sum_corr / self.n_corr

    def plot(self, axes=None, dt=None, label=None, color=None, linestyle=None):
        """Plot results."""
        if axes is None:
            axes = plt.subplot()
        if dt is None:
            time = np.arange(self.trunc)
        else:
            time = np.arange(self.trunc) * dt
        axes.plot(time, self.get_results(), color=color, linestyle=linestyle, label=label)
        for corr in self.corrs:
            axes.plot(time, corr, alpha=0.5, color=color, linestyle=linestyle)
        return axes


class VVCorr(IterativeCorrelation):
    """Compute the time auto-correlation function of the velocity
    by computing the gradient of the data added. """

    @beartype
    def __init__(self, dt: Union[float, int], trunc: Optional[int] = None, keep_corrs=False):
        super().__init__(lambda x: np.gradient(x, dt), trunc=trunc, keep_corrs=keep_corrs)
        self.dt = dt

    @beartype
    def add_data(self, pos: np.ndarray):
        """ Add position data to VVCorr. Velocities are estimated via central differences.
        Arguments:
            positions (np.ndarray (ntrajs, nsteps,) or (nsteps,): position data.
        """
        super().add_data(pos, None)

    def plot(self, axes=None, color=None):
        axes = super().plot(axes, self.dt, r"$t$ [nm/ps]", r"$C^{vv}$ [nm$^2$/ps$^2$]", color)
        return axes


class XdUCorr(IterativeCorrelation):
    """Compute the time correlation function of the position and
    the gradient of the pmf. By default, computes dU with gle_sim:
    https://github.com/lucastepper/gle_sim """

    @beartype
    def __init__(self, config: dict, trunc: Optional[int] = None, dt: Union[float, int] = None, keep_corrs=False):
        super().__init__(lambda x: x, lambda x: gle_sim.du_dx(x, config), trunc, keep_corrs)
        self.dt = dt

    @beartype
    def add_data(self, data1: np.ndarray):
        super().add_data(data1, data1)

    def plot(self, axes=None, color=None):
        axes = super().plot(
            axes, self.dt, r"$t$ [nm/ps]", r"$C^{vv}$ [nm$^2$/ps$^2$]", color
        )
        return axes
