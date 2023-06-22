from typing import Optional, Union
import numpy as np
# allow for optional torch import
try:
    import torch
except ImportError:
    pass


def correlation_torch(
    a: Union[np.ndarray, torch.Tensor], b: Optional[Union[np.ndarray, torch.Tensor]] = None, trunc: int = None, device: str = "cpu"
) -> Union[np.ndarray, torch.Tensor]:
    """Calculates correlation via FFT. We check
    if we compute ACF by id(a) == id(b) and only single fft
    in that case. If trunc not provided, returns len(a) points,
    else trunc. Only works for real data, as it uses mkl.rfft.
    Arguments:
        a (np.ndarray, torch.Tensor, real): input 1
        b (np.ndarray, torch.Tensor, real): input 2; if None, computes ACF; default None
        trunc (int): number of out elements to keep
        device (str): device to use for torch, default cpu
    Returns (np.ndarray, torch.Tensor): correlation. Output has the same number
        of dimensions as the input, returns same type as input.
    """
    assert isinstance(trunc, (int, type(None)))
    assert isinstance(device, str)
    if b is not None:
        assert a.shape == b.shape
        assert a.dtype == b.dtype

    # The code assumes 2d shape, reshape from 1d, maintain input var for later
    input_ndarray = isinstance(a, np.ndarray)
    if a.ndim == 1:
        reshaped = True
        a = a.reshape(1, -1)
        if b is not None:
            b.reshape(1, -1)
    elif a.ndim == 2:
        reshaped = False
    else:
        raise ValueError(f"Only 1d and 2d arrays supported, got {a.ndim}d array.")
    # convert to torch
    if isinstance(a, np.ndarray):
        a = torch.tensor(a, device=device)
    if isinstance(b, np.ndarray):
        b = torch.tensor(b, device=device)
    dtype = a.dtype
    if b is None:
        b = a

    len_a = a.shape[-1]
    # ndim, len_a = a.shape
    if not trunc:
        trunc = len_a
    trunc = min(trunc, len_a)

    a = torch.fft.rfft(a, n=2 * len_a, dim=-1)
    if id(a) == id(b):
        b = a
    else:
        b = torch.fft.rfft(b, n=2 * len_a, dim=-1)
    a = torch.conj(a) * b
    del b
    a = torch.real(torch.fft.irfft(a, n=None, dim=-1)[:, :trunc])
    a /= torch.linspace(len_a, len_a - trunc + 1, trunc, device=device, dtype=dtype).reshape(1, -1)
    # return numpy if input was numpy, return 1d if input was 1d
    a = a.cpu()
    if input_ndarray:
        a = a.numpy()
    if reshaped:
        return a.flatten()
    return a
