import time
import logging
from typing import Callable, Union, Optional
import psutil
import numpy as np
import h5py
import matplotlib.pyplot as plt
import gle_sim
from .correlation_mkl import correlation

MEMFACTOR = 17
LOGFILE = "/tmp/iterative_correlation.log"

logging.basicConfig(filename=LOGFILE, filemode="w", level=logging.INFO)


def get_nsplits(dset: h5py.Dataset, stride: int) -> int:
    """Get the amount of available memory. Assume that correlation
    computation requires MEMFACTOR (17) times the amount of memory as the data.
    """
    mem_avail = psutil.virtual_memory().available
    nbytes_data = dset.nbytes // stride
    fraction = mem_avail / (MEMFACTOR * nbytes_data)
    return int(np.ceil(1 / fraction))


class TimingContextManager:
    """Context manager for timing."""

    def __init__(self, execution_name):
        self.start_time = None
        self.execution_name = execution_name

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        execution_time = time.time() - self.start_time
        logging.info(f"Execution '{self.execution_name}' took {execution_time:.4e} seconds.")


class ComputeMean:
    """Accumulate data to compute mean over."""

    def __init__(self) -> None:
        self.sum_vals = 0.0
        self.counter = 0

    def add_data(self, vals: np.ndarray) -> None:
        """Accumulate data to compute mean over."""
        self.sum_vals += np.sum(vals)
        try:
            self.counter += np.sum(vals.shape)
        except AttributeError:
            self.counter += len(vals)

    def get_result(self) -> float:
        """Get mean over accumulated data."""
        return self.sum_vals / self.counter


def iter_chunks(dset: h5py.Dataset, stride: int = 1, nsplits: Optional[int] = None) -> np.ndarray:
    """Iterate over chunks of the dataset, allow for striding."""
    if nsplits is None:
        nsplits = get_nsplits(dset, stride)
    logging.info(f"Using {nsplits} splits for dataset")
    idxs_split = np.linspace(0, len(dset) + 1, nsplits + 1, dtype=int)
    while len(idxs_split) >= 2:
        # to allow for striding, we look at the start of the chunk and find the
        # lowest integer >= start that is divisible by stride
        # then we start from there and just stride through the chunk
        with TimingContextManager(f"loading_chunk_{idxs_split[0]}"):
            start, end = idxs_split[:2]
            idxs_split = idxs_split[1:]
            start = int(np.ceil(start / stride) * stride)
            chunk = dset[start : end : int(stride)]
            yield chunk


def get_traj_min_max(
    dset: h5py.Dataset, cache_traj_range: bool = True, nsplits: Optional[int] = None
) -> tuple[float, float]:
    """Find traj range of the dataset."""
    if (key := "traj_range") in dset.file:
        return tuple(dset.file[key][:])
    with TimingContextManager("get_traj_min_max"):
        print("Computing traj range...")
        min_traj, max_traj = np.inf, -np.inf
        for chunk in iter_chunks(dset, nsplits=nsplits):
            if (min_chunk := chunk.min()) < min_traj:
                min_traj = min_chunk
            if (max_chunk := chunk.max()) > max_traj:
                max_traj = max_chunk
        print(f"Found traj range: ({min_traj}, {max_traj})")
        if cache_traj_range and dset.file.mode == "r+":
            print("Caching traj range...")
            dset.file.create_dataset(key, data=np.array([min_traj, max_traj]))
    return min_traj, max_traj


def compute_fe(
    dset: h5py.Dataset,
    nbins: int,
    kbt: float,
    cache_fe: bool = True,
    nsplits: Optional[int] = None,
) -> np.ndarray:
    """Compute the free energy from the dataset."""
    if (key := f"fe_nbins_{nbins}") in dset.file:
        return dset.file[key][:]
    with TimingContextManager("compute_fe"):
        traj_range = get_traj_min_max(dset, nsplits=nsplits)
        print("Computing free energy...")
        hist = np.zeros(nbins)
        for chunk in iter_chunks(dset, nsplits=nsplits):
            temp, vals = np.histogram(chunk, bins=nbins, range=traj_range)
            hist += temp
        vals = (vals[1:] + vals[:-1]) / 2
        fe = -kbt * np.log(hist)
        fe -= fe.min()
        fe = np.stack((vals, fe), axis=0)
        if cache_fe and dset.file.mode == "r+":
            print("Caching free energy...")
            dset.file.create_dataset(key, data=fe)
    return fe


class IterativeCorrelation:
    """Module to iteratively add data and compute a time correlation function
    as the mean over all data added. Uses fft to compute the correlation.
    Optionally, pass conversion functions for argument 1 and 2 of correlation. TBD"""

    def __init__(
        self,
        # we need to use Optional[Callable] here instead of lambda x: x because
        # it allows us to know that we compute an autocorrelation function when
        # conv_func_1 == conv_func_2, which is tedious to check otherwise
        conv_func_1: Optional[Callable] = None,
        conv_func_2: Optional[Callable] = None,
        hook_data: Callable = lambda _x, _y: None,
        trunc: Optional[int] = None,
        stride: int = 1,
        keep_corrs=False,
    ):
        """Constructor for IterativeCorrelation. Pass conversion functions
        for argument 1 and 2 of correlation.
        Arguments:
            conv_func_1 (Optional[Callable]): conversion function for argument 1 of correlation.
                Defaults to None, no conversion.
            conv_func_2 (Optional[Callable]): conversion function for argument 2 of correlation.
                Defaults to None, no conversion.
            hook_data (Callable[dat1, dat2]): hook to call on data before adding it.
                Defaults to lambda _x, _y: None.
            trunc (Optional[int]): truncation of correlation. Defaults to None (no truncation).
            stride (int): stride for iterating over data. Defaults to 1.
            keep_corrs (bool): keep all correlations for plotting. Defaults to False.
        """
        self.trunc = trunc
        self.stride = stride
        self.conv_func_1 = conv_func_1
        self.conv_func_2 = conv_func_2
        self.hook_data = hook_data
        self.sum_corr = np.zeros(trunc)
        self.corrs = []
        self.keep_corrs = keep_corrs
        self.len_datas = []

    def convert_data_1(self, data1, stride):
        """Check if we defined a conversion function for data1, if so, apply it."""
        if self.conv_func_1 is None:
            return data1[::stride]
        out = self.conv_func_1(data1[::stride])
        if not isinstance(out, (type(None), np.ndarray)):
            raise ValueError(f"conv_func21 does not return a numpy array or None")
        return out

    def convert_data_2(self, data1, data2, stride):
        """If we defined a conversion function for data2, we apply it. As input, we use
        data2 if it is not None, otherwise data1. The four cases are:
            data2 is None, self.conv_func_2 is None -> data1
            data2 is None, self.conv_func_2 is set -> self.conv_func_2(data1)
            data2 is given, self.conv_func_2 is None -> data2
            data2 is given, self.conv_func_2 is set -> self.conv_func_2(data2)
        """
        if data2 is None:
            input_for_func = data1
        else:
            input_for_func = data2
        if self.conv_func_2 is None:
            return input_for_func[::stride]
        out = self.conv_func_2(input_for_func[::stride])
        if not isinstance(out, (type(None), np.ndarray)):
            raise ValueError(f"conv_func2 does not return a numpy array or None")
        return out

    def add_corr(self, corr, len_data):
        """Add a correlation that was computed. Check if trunc is correct,
        if not, adjust. Keep corr for plotting if keep_corrs is True.
        """
        if len(corr) < self.trunc:
            self.trunc = len(corr)
            self.corrs = [corr[: self.trunc] for corr in self.corrs]
        self.sum_corr = self.sum_corr[: self.trunc] + len_data * corr[: self.trunc]
        if self.keep_corrs:
            self.corrs.append(corr[: self.trunc])
        self.len_datas.append(len_data)

    def add_data(
        self, data1: np.ndarray, data2: Optional[np.ndarray] = None, stride: Optional[int] = None
    ):
        """Add data bit by bit and stride it. Since the iterative data chunks
        could be contiguous or not, we cannot make sure that the striding is
        happening correctly. Therefore, for contiguous data, the caller needs
        to pass len(data1) % stride == 0.
        Arguments:
            data1 (np.ndarray (ntrajs, nsteps,) or (nsteps,)): data to add.
            data2 (Optional[np.ndarray (ntrajs, nsteps,) or (nsteps,)]): data to add.
            stride (Optional[int]): Allow for overriding self.stride. Defaults to None.
                Overriding of self.stride is useful when adding data from h5py.Dataset,
                where we already stride while iterating in iter_chunks.
        """
        if stride is None:
            stride = self.stride
        # check shapes of data, allow 1d and 2d
        if data1.ndim == 1:
            data1 = data1.reshape(1, -1)
        if data2 is not None:
            if data2.ndim == 1:
                data2 = data2.reshape(1, -1)
            if data1.shape[1] != data2.shape[1]:
                raise ValueError(f"When adding data, {data1.shape[1]} != {data2.shape[1]}")
        else:
            data2 = [None] * len(data1)
        # iterate over data
        for dat1, dat2 in zip(data1, data2):
            conv_dat1 = self.convert_data_1(dat1, stride)
            conv_dat2 = self.convert_data_2(dat1, dat2, stride)
            self.hook_data(conv_dat1, conv_dat2)
            corr = correlation(conv_dat1, conv_dat2, self.trunc)
            # use len(conv_dat1) instead of len(dat1) to account for striding and possible
            # change of shape due to conversion
            self.add_corr(corr, len(conv_dat1))

    def add_data_h5(self, dset: h5py.Dataset, nsplits: Optional[int] = None):
        """Add data from h5py.Dataset bit by bit and stride it. For now,
        we do not implement getting data from multiple datasets.
        Arguments:
            dset (h5py.Dataset): dataset to add.
            nsplits (Optional[int]): number of splits to use for iterating over dataset.
                Defaults to None, which uses available memory to determine nsplits.
        """
        for chunk in iter_chunks(dset, self.stride, nsplits=nsplits):
            # we need to overwrite stride in add_data, since we already stride while iterating
            self.add_data(chunk, stride=1)

    def set_corr(self, corr):
        """Set the result to a given correlation."""
        self.sum_corr = corr
        self.len_datas = [1]
        self.corrs = []

    def get_result(self):
        """Get the result from self."""
        return self.sum_corr / sum(self.len_datas)

    def plot(self, axes: plt.Axes = None, dt: float = None, label: str = "", **kwargs):
        """Plot results.
        Arguments:
            axes (matplotlib.axes.Axes): axes to plot on.
                defaults to None, usesplt.subplot()
            dt (float): time step of data.
            label (str): label for plot. Defaults to "".
            kwargs: keyword arguments passed to plt.plot.
        Returns:
            axes (matplotlib.axes.Axes): axes plotted on.
        """
        if axes is None:
            axes = plt.subplot()
        if dt is None:
            time_arr = np.arange(self.trunc)
        else:
            time_arr = np.arange(self.trunc) * dt
        kwargs_result = kwargs.copy()
        kwargs_result["linestyle"] = "-"
        kwargs_result["ls"] = "-"
        axes.plot(time_arr, self.get_result(), label=label, **kwargs_result)
        for corr in self.corrs:
            axes.plot(time_arr, corr, alpha=0.5, **kwargs)
        return axes


class VVCorr(IterativeCorrelation):
    """Compute the time auto-correlation function of the velocity
    by computing the gradient of the data added."""

    def __init__(
        self,
        dt: Optional[Union[float, int]] = None,
        gradient_func: Optional[Callable] = None,
        trunc: Optional[int] = None,
        stride: int = 1,
        keep_corrs: bool = False,
    ):
        """Constructor for Iterative Velocity autocorrelation function. Pass conversion functions
        for argument 1 and 2 of correlation.
        Arguments:
            dt (Union[float, int]): time step of data. Time step is multiplied by stride
                to compute the effective time step, which is used to compute the velocities.
            gradient_func (Optional[Callable]): gradient function to compute velocities.
                Defaults to None, uses lambda x: np.diff(x) / self.dt.
            trunc (Optional[int]): truncation of correlation. Defaults to None (no truncation).
            stride (int): stride for iterating over data. Defaults to 1.
            keep_corrs (bool): keep all correlations for plotting. Defaults to False.
        """
        if dt is None and gradient_func is None:
            raise ValueError("Either dt or gradient_func must be set")
        if gradient_func is None:
            gradient_func = lambda x: np.diff(x) / self.dteff
        elif dt is None:
            self.dteff = 1.0
        else:
            self.dteff = dt * stride
        self.mean_vsqrd = ComputeMean()
        super().__init__(
            conv_func_1=gradient_func,
            # If we don't return None here, we compute the gradient twice
            # and then the two gradients have different id()s, so we do extra
            # work in correlation, which is not necessary
            conv_func_2=lambda _x: None,
            hook_data=lambda v, _v: self.mean_vsqrd.add_data(v**2),
            trunc=trunc,
            stride=stride,
            keep_corrs=keep_corrs,
        )

    def add_data(self, pos: np.ndarray, stride: Optional[int] = None):
        """Add position data to VVCorr. Velocities are estimated via central differences.
        Arguments:
            positions (np.ndarray (ntrajs, nsteps,) or (nsteps,): position data.
            stride (Optional[int]): Allow for overriding self.stride. Defaults to None.
                Overriding of self.stride is useful when adding data from h5py.Dataset,
                where we already stride while iterating in iter_chunks.
        """
        super().add_data(pos, None, stride=stride)

    def get_mean_vsqrd(self) -> float:
        """ Get mean of squared veloicty over accumulated data."""
        return self.mean_vsqrd.get_result()

    def plot(self, axes=None, color=None):
        axes = super().plot(axes, self.dteff, color=color)
        axes.set(xlabel=r"$t$ [nm/ps]", ylabel=r"$C^{vv}$ [nm$^2$/ps$^2$]")
        return axes


class XdUCorr(IterativeCorrelation):
    """Compute the time correlation function of the position and
    the gradient of the pmf. By default, computes dU with gle_sim:
    https://github.com/lucastepper/gle_sim"""

    def __init__(
        self,
        kbt: Optional[float] = None,
        nbins: Optional[int] = None,
        trunc: Optional[int] = None,
        dt: Union[float, int] = 1.0,
        keep_corrs: bool = False,
        stride: int = 1,
    ):
        self.dteff = dt * stride
        self._kbt = kbt
        self._nbins = nbins
        self.config = None
        self.mean_du = ComputeMean()
        super().__init__(
            conv_func_1=lambda x: x,
            conv_func_2=self.get_du,
            hook_data=lambda _x, du: self.mean_du.add_data(du),
            trunc=trunc,
            stride=stride,
            keep_corrs=keep_corrs,
        )

    @property
    def kbt(self) -> float:
        """Check that kbt is set when computing fe."""
        if self._kbt is None:
            raise ValueError("No kbt set for XdUCorr to compute fe")
        return self._kbt

    @property
    def nbins(self) -> int:
        """Check that nbins is set when computing fe."""
        if self._nbins is None:
            raise ValueError("No nbins set for XdUCorr to compute fe")
        return self._nbins

    def get_du(self, x: np.ndarray) -> np.ndarray:
        """Compute the gradient of the pmf with gle_sim."""
        if self.config is None:
            raise ValueError("No config set for XdUCorr to compute dU")
        return -gle_sim.get_force_pmf(self.config, x)

    def set_fe(self, fe: np.ndarray) -> None:
        """Set the free energy. Always use linear interpolation for fe to compute dU.
        Arguments:
            fe (np.ndarray): free energy (values, fe). Shape (2, nbins).
        """
        if not isinstance(fe, np.ndarray):
            raise ValueError(f"fe must be a numpy array, got {type(fe)}")
        if not fe.shape[0] == 2 or not fe.ndim == 2:
            raise ValueError(f"fe must have shape (2, nbins), got {fe.shape}")
        self.config = gle_sim.Config.default()
        self.config.pmf = gle_sim.LinearInterpolation(values=fe[0], fe=fe[1])

    def add_data_h5(self, dset: h5py.Dataset, nsplits: Optional[int] = None):
        """Add data from h5py.Dataset bit by bit and stride it. For now,
        we do not implement getting data from multiple datasets. If self.config
        is not set, compute the free energy from the dataset. If dset["fe_nbins_{nbins}"]
        is present, use that instead of computing the free energy.
        Arguments:
            dset (h5py.Dataset): dataset to add.
            nsplits (Optional[int]): number of splits to use for iterating over dataset.
                Defaults to None, which uses available memory to determine nsplits.
        """
        if self.config is None:
            self.set_fe(compute_fe(dset, self.nbins, self.kbt, nsplits=nsplits))
        super().add_data_h5(dset, nsplits=nsplits)

    def compute_fe(self, trajs: np.ndarray) -> None:
        """Compute fe from trajs. Always use linear interpolation for fe to compute dU.
        Arguments:
            trajs (np.ndarray or Iterable[np.ndarray]): trajectories to compute fe from.
        """
        if isinstance(trajs, np.ndarray) and trajs.ndim == 1:
            trajs = trajs.reshape(1, -1)
        else:
            if any(not isinstance(traj, np.ndarray) for traj in trajs):
                raise ValueError("trajs must be a numpy array or iterable of numpy arrays")
        hist = np.zeros(self.nbins)
        kbt = self.kbt
        # compute trajs range
        trajs_min, trajs_max = np.inf, -np.inf
        for traj in trajs:
            if (min_temp := traj.min()) < trajs_min:
                trajs_min = min_temp
            if (max_temp := traj.max()) > trajs_max:
                trajs_max = max_temp
        # compute fe
        for traj in trajs:
            hist_tmp, vals = np.histogram(traj, bins=self.nbins, range=(trajs_min, trajs_max))
            hist += hist_tmp
        vals = (vals[1:] + vals[:-1]) / 2
        # pylint: disable=invalid-unary-operand-type
        fe = -kbt * np.log(hist)
        fe = np.stack((vals, fe - fe.min()), axis=0)
        # set self.config for later calls to self.get_du
        self.config = gle_sim.Config.default()
        self.config.pmf = gle_sim.LinearInterpolation(values=fe[0], fe=fe[1])
        return fe

    def get_mean_du(self) -> float:
        """Get mean over the gradient of the force over all data added."""
        return self.mean_du.get_result()

    def plot(self, axes=None, color=None):
        axes = super().plot(axes, self.dteff, color=color)
        axes.set(xlabel=r"$t$ [nm/ps]", ylabel=r"$C^{x \nabla U}$ [kJ/mol]")
        return axes
