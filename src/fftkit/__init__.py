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

__version__ = "0.1.0"

# Import from backends
from .backends import (
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

__all__ = [
    # Version
    "__version__",
    # Config
    "detect_backend",
    "DEFAULT_BACKEND",
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
