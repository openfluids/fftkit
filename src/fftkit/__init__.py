"""fftkit — one FFT API over many backends.

fftkit provides a unified interface to multiple FFT implementations:
scipy, numpy, Intel MKL, GPU backends (CuPy, PyTorch), and more.

Quick example:

    import numpy as np
    import fftkit

    x = np.random.randn(1024)
    func = fftkit.get_fft_func()  # Auto-selects best available backend
    X = func(x)  # Compute FFT

Environment variables:

- ``FFTKIT_BACKEND``: Force specific backend (e.g., 'mkl', 'scipy', 'cupy')
- ``PYMODAL_FFT_BACKEND``: Legacy fallback for modalpy users
"""

from __future__ import annotations

import importlib.metadata
from typing import Any

from numpy.typing import ArrayLike
from scipy.fft import fftfreq, rfftfreq

# Import from backends
from .backends import (
    BACKENDS,
    UNSET_AXIS,
    ArrayResult,
    AxisDefaultWarning,
    TransformFunc,
    accelerate_fft,
    benchmark_backends,
    cupy_fft,
    get_available_backends,
    get_backend_names,
    get_fft_func,
    get_optimal_backend,
    gpu_available,
    mkl_available,
    mkl_fft_transform,
    numpy_fft,
    register_mkl_scipy_backend,
    resolve_axis,
    scipy_fft,
)

# Import from config
from .config import DEFAULT_BACKEND, detect_backend

# Import from gpu
from .gpu import (
    GPUBatchFFT,
    GPUConfig,
    benchmark_cpu_vs_gpu,
    get_gpu_info,
    gpu_fft,
    gpu_rfft,
    should_use_gpu,
)

# Import from signals
from .signals import generate_complex_signal

# Import from spectral
from .spectral import (
    blackman_tukey_rfft,
    calculate_error,
    find_peaks,
    periodogram_rfft,
    welch_method,
)

try:
    # pyproject.toml is the single source of truth for the version; this
    # reads it back out of the installed package metadata rather than
    # duplicating the string here, so the two can never drift apart again.
    __version__ = importlib.metadata.version("fftkit")
except importlib.metadata.PackageNotFoundError:
    # Source tree without an install record (e.g. running straight out of a
    # git checkout with no editable install).
    __version__ = "0.0.0+unknown"


def _dispatch(transform: str, backend: str | None) -> TransformFunc:
    return BACKENDS[backend or DEFAULT_BACKEND].get(transform)


# axis defaults to UNSET_AXIS rather than -1 so the 1-D transforms can tell
# "caller omitted axis" from "caller asked for -1". Only the first case can
# silently differ from 0.1.x, and only for multi-dimensional input; see
# AxisDefaultWarning. The effective default is still -1.
def fft(x: ArrayLike, n: int | None = None, axis: Any = UNSET_AXIS, norm: str | None = None,
        backend: str | None = None) -> ArrayResult:
    return _dispatch("fft", backend)(x, n=n, axis=resolve_axis(x, axis, "fft"), norm=norm)


def ifft(x: ArrayLike, n: int | None = None, axis: Any = UNSET_AXIS, norm: str | None = None,
         backend: str | None = None) -> ArrayResult:
    return _dispatch("ifft", backend)(x, n=n, axis=resolve_axis(x, axis, "ifft"), norm=norm)


def rfft(x: ArrayLike, n: int | None = None, axis: Any = UNSET_AXIS, norm: str | None = None,
         backend: str | None = None) -> ArrayResult:
    return _dispatch("rfft", backend)(x, n=n, axis=resolve_axis(x, axis, "rfft"), norm=norm)


def irfft(x: ArrayLike, n: int | None = None, axis: Any = UNSET_AXIS, norm: str | None = None,
          backend: str | None = None) -> ArrayResult:
    return _dispatch("irfft", backend)(x, n=n, axis=resolve_axis(x, axis, "irfft"), norm=norm)


def fft2(x: ArrayLike, s: tuple[int, ...] | None = None, axes: tuple[int, ...] | None = None,
          norm: str | None = None, backend: str | None = None) -> ArrayResult:
    return _dispatch("fft2", backend)(x, s=s, axes=axes, norm=norm)


def ifft2(x: ArrayLike, s: tuple[int, ...] | None = None, axes: tuple[int, ...] | None = None,
           norm: str | None = None, backend: str | None = None) -> ArrayResult:
    return _dispatch("ifft2", backend)(x, s=s, axes=axes, norm=norm)


def fftn(x: ArrayLike, s: tuple[int, ...] | None = None, axes: tuple[int, ...] | None = None,
          norm: str | None = None, backend: str | None = None) -> ArrayResult:
    return _dispatch("fftn", backend)(x, s=s, axes=axes, norm=norm)


def ifftn(x: ArrayLike, s: tuple[int, ...] | None = None, axes: tuple[int, ...] | None = None,
           norm: str | None = None, backend: str | None = None) -> ArrayResult:
    return _dispatch("ifftn", backend)(x, s=s, axes=axes, norm=norm)


__all__ = [
    # Version
    "__version__",
    "AxisDefaultWarning",
    # Config
    "detect_backend",
    "DEFAULT_BACKEND",
    # Transform matrix
    "fft",
    "ifft",
    "rfft",
    "irfft",
    "fft2",
    "ifft2",
    "fftn",
    "ifftn",
    "fftfreq",
    "rfftfreq",
    # Backends
    "get_fft_func",
    "get_backend_names",
    "get_available_backends",
    "get_optimal_backend",
    "benchmark_backends",
    "gpu_available",
    "mkl_available",
    "register_mkl_scipy_backend",
    "scipy_fft",
    "numpy_fft",
    "mkl_fft_transform",
    "cupy_fft",
    "accelerate_fft",
    # GPU
    "GPUBatchFFT",
    "GPUConfig",
    "should_use_gpu",
    "get_gpu_info",
    "gpu_fft",
    "gpu_rfft",
    "benchmark_cpu_vs_gpu",
    # Spectral
    "periodogram_rfft",
    "blackman_tukey_rfft",
    "welch_method",
    "find_peaks",
    "calculate_error",
    # Signals
    "generate_complex_signal",
]
