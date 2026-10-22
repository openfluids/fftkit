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
    MklBackendWarning,
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
    next_fast_len,
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
# High-level entry point for physical signals. Deliberately imported at the top
# level: it is the first thing most users of simulation data should reach for,
# and burying it in a submodule would make the neutral low-level transforms the
# path of least resistance for people who want the opinionated defaults.
from .physical import (
    MethodChoice,
    ResamplingWarning,
    SamplingReport,
    SpectrumResult,
    UniformResampling,
    choose_method,
    compare_methods,
    describe_sampling,
    resample_uniform,
    spectrum,
    tonality,
)
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


# Shared by every transform's docstring below. `workers` is documented once
# here rather than duplicated with drift risk across eight near-identical
# paragraphs; see fftkit.backends module docstring for the full per-backend
# breakdown of who honours it.
_WORKERS_DOC = """
    workers: Number of threads to use, or None (default) to leave the
        backend's own default threading untouched -- a true no-op, nothing
        is passed to the backend at all. -1 means "all available cores"
        (scipy's own convention; the pyfftw adapter maps -1 to
        os.cpu_count()). Only the scipy and pyfftw backends honour this;
        every other backend silently ignores it rather than raising, because
        it is a performance hint that can only change how fast a result is
        computed, never the result itself -- code that passes workers=
        stays portable across backends that can't use it.

        WARNING for MPI/multi-process users (e.g. one process per rank
        during DNS/LES post-processing): workers=-1 inside N ranks on an
        N-core node oversubscribes the node by up to a factor of N. The
        default is None specifically so this call is safe by default inside
        an MPI-parallel run; opt into -1 or an explicit thread count only in
        single-process contexts.
"""


# axis defaults to UNSET_AXIS rather than -1 so the 1-D transforms can tell
# "caller omitted axis" from "caller asked for -1". Only the first case can
# silently differ from 0.1.x, and only for multi-dimensional input; see
# AxisDefaultWarning. The effective default is still -1.
def fft(x: ArrayLike, n: int | None = None, axis: Any = UNSET_AXIS, norm: str | None = None,
        backend: str | None = None, workers: int | None = None) -> ArrayResult:
    return _dispatch("fft", backend)(x, n=n, axis=resolve_axis(x, axis, "fft"), norm=norm, workers=workers)


def ifft(x: ArrayLike, n: int | None = None, axis: Any = UNSET_AXIS, norm: str | None = None,
         backend: str | None = None, workers: int | None = None) -> ArrayResult:
    return _dispatch("ifft", backend)(x, n=n, axis=resolve_axis(x, axis, "ifft"), norm=norm, workers=workers)


def rfft(x: ArrayLike, n: int | None = None, axis: Any = UNSET_AXIS, norm: str | None = None,
         backend: str | None = None, workers: int | None = None) -> ArrayResult:
    return _dispatch("rfft", backend)(x, n=n, axis=resolve_axis(x, axis, "rfft"), norm=norm, workers=workers)


def irfft(x: ArrayLike, n: int | None = None, axis: Any = UNSET_AXIS, norm: str | None = None,
          backend: str | None = None, workers: int | None = None) -> ArrayResult:
    return _dispatch("irfft", backend)(x, n=n, axis=resolve_axis(x, axis, "irfft"), norm=norm, workers=workers)


def fft2(x: ArrayLike, s: tuple[int, ...] | None = None, axes: tuple[int, ...] | None = None,
          norm: str | None = None, backend: str | None = None, workers: int | None = None) -> ArrayResult:
    return _dispatch("fft2", backend)(x, s=s, axes=axes, norm=norm, workers=workers)


def ifft2(x: ArrayLike, s: tuple[int, ...] | None = None, axes: tuple[int, ...] | None = None,
           norm: str | None = None, backend: str | None = None, workers: int | None = None) -> ArrayResult:
    return _dispatch("ifft2", backend)(x, s=s, axes=axes, norm=norm, workers=workers)


def fftn(x: ArrayLike, s: tuple[int, ...] | None = None, axes: tuple[int, ...] | None = None,
          norm: str | None = None, backend: str | None = None, workers: int | None = None) -> ArrayResult:
    return _dispatch("fftn", backend)(x, s=s, axes=axes, norm=norm, workers=workers)


def ifftn(x: ArrayLike, s: tuple[int, ...] | None = None, axes: tuple[int, ...] | None = None,
           norm: str | None = None, backend: str | None = None, workers: int | None = None) -> ArrayResult:
    return _dispatch("ifftn", backend)(x, s=s, axes=axes, norm=norm, workers=workers)


# Real (non-f-string) docstrings assigned by attribute rather than written
# inline: all eight transforms share the exact same `workers` semantics, and
# writing that paragraph out eight times risks the copies drifting apart.
# Assigning __doc__ here keeps help()/Sphinx introspection working (an
# f-string is not a literal, so Python does not treat it as a docstring at
# all) while keeping a single source of truth for the shared text.
_ONE_LINERS = {
    fft: "Forward complex FFT.",
    ifft: "Inverse complex FFT.",
    rfft: "Forward real-input FFT.",
    irfft: "Inverse real-input FFT.",
    fft2: "2-D forward FFT.",
    ifft2: "2-D inverse FFT.",
    fftn: "N-D forward FFT.",
    ifftn: "N-D inverse FFT.",
}
for _func, _summary in _ONE_LINERS.items():
    _func.__doc__ = f"{_summary} See the fftkit.backends module docstring for the full API.\n\nArgs:{_WORKERS_DOC}"
del _func, _summary


__all__ = [
    # Version
    "__version__",
    "AxisDefaultWarning",
    "MklBackendWarning",
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
    "next_fast_len",
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
    # Physical signals: the high-level entry point
    "spectrum",
    "compare_methods",
    "choose_method",
    "tonality",
    "resample_uniform",
    "describe_sampling",
    "SpectrumResult",
    "SamplingReport",
    "UniformResampling",
    "MethodChoice",
    "ResamplingWarning",
    # Spectral
    "periodogram_rfft",
    "blackman_tukey_rfft",
    "welch_method",
    "find_peaks",
    "calculate_error",
    # Signals
    "generate_complex_signal",
]
