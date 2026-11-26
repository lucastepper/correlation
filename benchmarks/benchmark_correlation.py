#!/usr/bin/env python3
"""Compare package, NumPy, and SciPy FFT-based correlations.

Example:
    python benchmarks/benchmark_correlation.py --size 1000000 --repeats 20
"""

from __future__ import annotations

import argparse
import statistics
import threading
import time

import numpy as np
import psutil

try:
    from correlation import correlation as package_correlation
except ModuleNotFoundError as error:
    if error.name == "mkl_fft":
        raise SystemExit(
            "This benchmark requires the package dependency mkl_fft; "
            "install the project dependencies first."
        ) from error
    raise


def numpy_fft_correlation(a: np.ndarray, b: np.ndarray, trunc: int) -> np.ndarray:
    n = a.shape[-1]
    fa = np.fft.rfft(a, n=2 * n)
    fb = np.fft.rfft(b, n=2 * n)
    result = np.fft.irfft(np.conj(fa) * fb, n=2 * n)[:trunc]
    return result / np.arange(n, n - trunc, -1)


def scipy_fft_correlation(a: np.ndarray, b: np.ndarray, trunc: int) -> np.ndarray:
    from scipy import fft

    n = a.shape[-1]
    fa = fft.rfft(a, n=2 * n)
    fb = fft.rfft(b, n=2 * n)
    result = fft.irfft(np.conj(fa) * fb, n=2 * n)[:trunc]
    return result / np.arange(n, n - trunc, -1)


def run_measured(function, a, b, trunc):
    process = psutil.Process()
    start_rss = process.memory_info().rss
    peak_rss = start_rss
    running = True

    def sample_rss():
        nonlocal peak_rss
        while running:
            peak_rss = max(peak_rss, process.memory_info().rss)
            time.sleep(0.001)

    sampler = threading.Thread(target=sample_rss)
    sampler.start()
    start = time.perf_counter()
    output = function(a, b, trunc)
    elapsed = time.perf_counter() - start
    running = False
    sampler.join()
    peak_rss = max(peak_rss, process.memory_info().rss)
    return output, elapsed, peak_rss - start_rss


def benchmark(function, a, b, trunc, repeats):
    function(a, b, trunc)  # warm-up
    timings = []
    memory = []
    output = None
    for _ in range(repeats):
        output, elapsed, peak_bytes = run_measured(function, a, b, trunc)
        timings.append(elapsed)
        memory.append(peak_bytes)
    return output, statistics.median(timings), max(memory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=100_000)
    parser.add_argument("--trunc", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cross-correlation", action="store_true")
    args = parser.parse_args()
    if args.size < 1 or args.repeats < 1:
        parser.error("--size and --repeats must be positive")

    trunc = min(args.size, args.trunc or args.size)
    rng = np.random.default_rng(args.seed)
    a = rng.standard_normal(args.size)
    b = rng.standard_normal(args.size) if args.cross_correlation else a
    implementations = [("package (mkl_fft)", package_correlation), ("numpy.fft", numpy_fft_correlation)]
    try:
        from scipy import fft  # noqa: F401
        implementations.append(("scipy.fft", scipy_fft_correlation))
    except ImportError:
        print("scipy.fft unavailable; install scipy to include it.")

    reference = None
    print(f"n={args.size:,}, trunc={trunc:,}, repeats={args.repeats}")
    if args.size <= 10_000_000:
        time_scale, time_unit = 1e3, "ms"
        memory_scale, memory_unit = 1e6, "MB"
    else:
        time_scale, time_unit = 1.0, "s"
        memory_scale, memory_unit = 1e9, "GB"

    print(f"| Implementation | Median time ({time_unit}) | Peak RAM usage ({memory_unit}) | Max error |")
    print("|---|---:|---:|---:|")
    for name, function in implementations:
        output, median, peak_bytes = benchmark(function, a, b, trunc, args.repeats)
        if reference is None:
            reference = output
        error = np.max(np.abs(output - reference))
        print(f"| `{name}` | {median * time_scale:.3f} | {peak_bytes / memory_scale:.2f} | {error:.3e} |")


if __name__ == "__main__":
    main()
