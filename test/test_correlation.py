import os
os.chdir('..')
import numpy as np
from correlation import correlation


def correlation_ref(a, b, trunc):
    output = np.zeros(trunc)
    counter = np.zeros(trunc)
    for i in range(trunc):
        for j in range(trunc - i):
            output[j] += a[i] * b[i + j]
            counter[j] += 1
    output /= counter
    return output


def test_autocorr_1d():
    size = int(1e3 + 1)
    np.random.seed(42)
    time = np.arange(size) * 0.02
    data = np.sin(time) + np.random.normal(scale=0.1, size=size)
    data_copy = np.copy(data)
    np.testing.assert_allclose(correlation(data, data), correlation_ref(data, data, size))
    # somehow, setting overwrite in correlation does not corrupt input data
    np.testing.assert_allclose(data, data_copy)


def test_corr_1d():
    size = int(1e3 + 1)
    np.random.seed(42)
    time = np.arange(size) * 0.02
    data1 = np.sin(time) + np.random.normal(scale=0.1, size=size)
    data1_copy = np.copy(data1)
    data2 = np.sin(time) + np.random.normal(scale=0.1, size=size)
    data2_copy = np.copy(data2)
    np.testing.assert_allclose(correlation(data1, data2), correlation_ref(data1, data2, size))
    np.testing.assert_allclose(data1, data1_copy)
    np.testing.assert_allclose(data2, data2_copy)


def test_autocorr_2d():
    size = (3, int(1e3 + 11))
    np.random.seed(42)
    time = np.arange(size[1]) * 0.02
    data = np.sin(time) + np.random.normal(scale=0.1, size=size)
    data_copy = np.copy(data)
    ref = np.zeros_like(data)
    for i, dat in enumerate(data):
        ref[i] = correlation_ref(dat, dat, data.shape[1])
    np.testing.assert_allclose(correlation(data, data), ref)
    np.testing.assert_allclose(data, data_copy)


def test_corr_2d():
    size = (3, int(1e3 + 11))
    np.random.seed(42)
    time = np.arange(size[1]) * 0.02
    data1 = np.sin(time) + np.random.normal(scale=0.1, size=size)
    data1_copy = np.copy(data1)
    data2 = np.sin(time) + np.random.normal(scale=0.1, size=size)
    data2_copy = np.copy(data2)
    ref = np.zeros_like(data1)
    for i, (dat1, dat2) in enumerate(zip(data1, data2)):
        ref[i] = correlation_ref(dat1, dat2, data1.shape[1])
    np.testing.assert_allclose(correlation(data1, data2), ref)
    np.testing.assert_allclose(data1, data1_copy)
    np.testing.assert_allclose(data2, data2_copy)