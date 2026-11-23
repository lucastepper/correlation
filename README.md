# correlation

Tools for computing time correlation functions of real-valued trajectories, with FFT, PyTorch, conditional, and incremental HDF5 workflows.

## Installation

Install the package and its Python dependencies with pip:

```bash
pip install .
```

This builds the Rust extension with CPU support by default. To build CUDA support, build a wheel with Maturin and install it:

```bash
maturin build --release --features cuda
pip install target/wheels/correlation-*.whl
```

CUDA builds require a CUDA driver library and toolchain available to the linker. CUDA execution in the PyTorch API also requires a CUDA-enabled PyTorch installation.

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

The default device is CPU. CUDA device support is available only when the extension was built with the `cuda` Cargo feature.

`ConditionalCorrelationFFT` accumulates conditional correlations using FFTs. Construct it with `from_range(nbins, traj_range, trunc)` or `from_dx(dx, nbins, start, trunc)`, add trajectories with `add_data(...)`, then call `get_result()`.
Without PyTorch, `add_data_cuda(...)` issues a warning and uses `add_data(...)` on the CPU.

### Incremental HDF5 correlation

`IterativeCorrelation` and its `VVCorr` (velocity autocorrelation function) and `XdUCorr` (position gradient-of-pmf correlation function) variants support correlation calculations over chunked HDF5 datasets, limiting memory use for large inputs. See their class docstrings for the dataset and observable arguments.
