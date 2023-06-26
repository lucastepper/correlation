import pytest
import numpy as np
import h5py
from numba import njit
from correlation import IterativeCorrelation, VVCorr, XdUCorr

np.random.seed(42)
KBT = 2.494


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


def gen_test_data(steps, dt, return_du=False):
    """Generate a test trajectory in harmonic potential"""
    try:
        import gle_sim
    except ImportError:
        pytest.skip("Could not import GLE simulation module")
    config = gle_sim.Config.default()
    config.steps = steps
    config.dt = dt
    config.pmf = gle_sim.Harmonic(k=45.0)
    config.kernel = gle_sim.Exponential(ks=[400.0], gammas=[100.0])
    config.integrator = "RungeKutta4"
    config.kbt = KBT
    config.seed = np.random.randint(0, int(2 ** 32 - 1))
    traj = gle_sim.integrate(config)
    if return_du:
        du = -gle_sim.get_force_pmf(config, traj)
        return traj, du
    return traj


@pytest.mark.parametrize(
    "pass_vel, ntraj, stride",
    [(True, 1, 1), (True, 2, 1), (False, 1, 1), (False, 2, 1), (True, 2, 11), (False, 2, 11)],
)
def test_vv_corr(pass_vel, ntraj, stride):
    """Test the velocity-velocity correlation function from test sim data"""
    trunc, steps, dt = 10, int(1e4), 1e-3
    corr_ref = np.zeros(trunc)
    n_traj_tot = 0
    # obj for test
    if pass_vel:
        # we do not want so stride when we pass vel as input, as the grad changes with stride
        corr_obj = IterativeCorrelation(trunc=trunc, stride=1)
        add_data = lambda _x, v: corr_obj.add_data(v)
    else:
        corr_obj = VVCorr(dt, trunc=trunc, keep_corrs=True, stride=stride)
        add_data = lambda x, _v: corr_obj.add_data(x)
    # do computation
    for i in range(1, ntraj + 1):
        traj = gen_test_data(steps * i, dt)
        vel = np.diff(traj[::stride]) / (dt * stride)
        corr_ref += len(vel) * correlation_ref(vel, vel, trunc)
        n_traj_tot += len(vel)
        add_data(traj, vel)
    corr_ref /= n_traj_tot
    print(corr_obj.corrs)
    np.testing.assert_allclose(corr_obj.get_result(), corr_ref)


@pytest.mark.parametrize("stride, nsplits", [(1, 1), (11, 1), (2, 11)])
def test_vv_corr_h5(stride, nsplits):
    """Test the velocity-velocity correlation function from test sim data using h5 file in /tmp"""
    trunc, steps, dt = 10, int(1e5), 1e-3
    traj = gen_test_data(steps, dt)
    vel = np.diff(traj[::stride]) / (dt * stride)
    corr_ref = correlation_ref(vel, vel, trunc)

    with h5py.File("/tmp/test.h5", "w") as fh:
        _dset = fh.create_dataset("traj", data=traj)
    corr_obj = VVCorr(dt, trunc=trunc, stride=stride)
    with h5py.File("/tmp/test.h5", "r") as fh:
        corr_obj.add_data_h5(fh["traj"], nsplits=nsplits)
    # test
    if nsplits == 1:
        atol = 1e-16
    else:
        # We are using an unsplit trajectory for ref corr, so we need to increase atol
        atol = 1e-3
    np.testing.assert_allclose(corr_obj.get_result(), corr_ref, atol=atol)


@pytest.mark.parametrize(
    "pass_du, ntraj, stride",
    [(True, 1, 1), (False, 1, 1), (True, 2, 1), (True, 1, 11), (True, 2, 11), (False, 2, 11)],
)
def test_xdu_corr(pass_du, ntraj, stride):
    """Test the position-force correlation function from test sim data"""
    trunc, steps, dt = 10, int(1e5), 1e-3
    corr_ref = np.zeros(trunc)
    trajs = [gen_test_data(steps * i, dt, return_du=True) for i in range(1, ntraj + 1)]
    # obj for test
    if pass_du:
        corr_obj = IterativeCorrelation(trunc=trunc, stride=stride)
        add_data = corr_obj.add_data
    else:
        corr_obj = XdUCorr(trunc=trunc, keep_corrs=True, stride=stride, kbt=KBT)
        corr_obj.compute_fe([traj for traj, _du in trajs], nbins=200)
        # we need to overwrite du in trajs, as we have the correct, linear force there now
        trajs = [(traj, corr_obj.get_du(traj)) for traj, _ in trajs]
        add_data = lambda x, _du: corr_obj.add_data(x)
    # do computation
    for traj, du in trajs:
        add_data(traj, du)
        traj, du = traj[::stride], du[::stride]
        corr_ref += len(du) * correlation_ref(traj, du, trunc)
    corr_ref /= np.sum([len(traj[::stride]) for traj, _ in trajs])
    np.testing.assert_allclose(corr_obj.get_result(), corr_ref)


@pytest.mark.parametrize("stride, nsplits", [(1, 1), (11, 1), (2, 11)])
def test_xdu_corr_h5(stride, nsplits):
    """Test the velocity-velocity correlation function from test sim data using h5 file in /tmp"""
    trunc, steps, dt = 10, int(1e5), 1e-3
    traj = gen_test_data(steps, dt)
    corr_obj = XdUCorr(trunc=trunc, keep_corrs=True, stride=stride, kbt=KBT)
    # use corr obj to compute du, as we need linear interpolation for du
    corr_obj.compute_fe(traj, nbins=200)
    du = corr_obj.get_du(traj)
    corr_ref = correlation_ref(traj[::stride], du[::stride], trunc)
    # kick config again, so we have to recompute du from h5 file
    corr_obj.config = None

    with h5py.File("/tmp/test.h5", "w") as fh:
        _dset = fh.create_dataset("traj", data=traj)
    with h5py.File("/tmp/test.h5", "r") as fh:
        corr_obj.add_data_h5(fh["traj"], nbins=200, nsplits=nsplits)
    # test
    if nsplits == 1:
        atol, rtol = 1e-16, 1e-7
    else:
        # We are using an unsplit trajectory for ref corr, so we need to increase atol
        atol, rtol = 1e-2, 1e-2
    np.testing.assert_allclose(corr_obj.get_result(), corr_ref, atol=atol, rtol=rtol)
