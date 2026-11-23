# correlation

Tools for computing time correlation functions of real-valued trajectories, with FFT, PyTorch, conditional, and incremental HDF5 workflows.

## Installation

Install `correlation-fft` from [PyPI](https://pypi.org/project/correlation-fft/).

```bash
pip install correlation-fft
```

The Rust extension always builds with both CPU and CUDA support. Build a wheel for your target platform and Python version with Maturin:

```bash
maturin build --release
pip install target/wheels/correlation_fft-*.whl
```

The Rust extension loads the NVIDIA driver when it tries to use CUDA. CPU-only machines can use the same wheel. The Rust CUDA path uses the embedded PTX and requires a compatible NVIDIA driver. The separate PyTorch CUDA path requires a CUDA-enabled PyTorch installation.

## Features and usage

### Correlation functions

`correlation(a, b=None, trunc=None)` computes an autocorrelation when `b` is omitted, or a cross-correlation when `b` is supplied. It supports one- and two-dimensional NumPy arrays and uses `mkl_fft`.

```python
import numpy as np
from correlation import correlation

x = np.random.randn(10000)
acf = correlation(x)                 # autocorrelation
cross = correlation(x, np.roll(x, 1), trunc=1000)
```

`correlation_torch` provides the same operation through PyTorch and accepts a `device` such as `"cpu"` or `"cuda"`. Install PyTorch separately with `pip install torch` to use it.

```python
from correlation import correlation_torch

acf_gpu = correlation_torch(x, device="cuda")
```

### Conditional correlation

`ConditionalCorrelation` is a Rust-backed accumulator. Choose a trajectory range and number of bins, then add trajectory arrays and retrieve the result:

```python
from correlation import ConditionalCorrelation

cc = ConditionalCorrelation(100, 8, (-2.0, 2.0), cond_start=True)
cc.add_data(x, x, condition)
result = cc.get_result()
```

The default device is `"cpu"`. `device="auto"` selects CUDA when the driver can load the kernel and `cond_start=True, cond_end=False`. Otherwise it selects CPU. Explicit `device="cuda"` (or `"gpu"`) raises an error if the driver, kernel, or conditioning mode is unavailable. The `device` property reports the selected backend.

`add_data` accepts `float32` and `float64` NumPy arrays on both backends. The CPU computes in double precision. The CUDA kernel converts inputs to single precision. Use `device="cpu"` to keep double precision.

`ConditionalCorrelationFFT` accumulates conditional correlations using FFTs. Construct it with `from_range(nbins, traj_range, trunc)` or `from_dx(dx, nbins, start, trunc)`, add trajectories with `add_data(...)`, then call `get_result()`.
Without PyTorch, `add_data_cuda(...)` issues a warning and uses `add_data(...)` on the CPU.

### Incremental HDF5 correlation

`IterativeCorrelation` and its `VVCorr` (velocity autocorrelation function) and `XdUCorr` (position gradient-of-pmf correlation function) variants support correlation calculations over chunked HDF5 datasets, limiting memory use for large inputs. See their class docstrings for the dataset and observable arguments.
