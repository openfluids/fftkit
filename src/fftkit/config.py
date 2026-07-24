"""FFT backend configuration and auto-detection for fftkit.

This module handles detection and configuration of the FFT backend to use.
"""

import os


def detect_backend():
    """Auto-detect best available FFT backend.

    Checks for available FFT libraries in the following precedence:
    1. FFTKIT_BACKEND environment variable (highest precedence)
    2. PYMODAL_FFT_BACKEND environment variable (legacy fallback for modalpy users)
    3. mkl_fft module (2-10x faster than scipy on Intel CPUs)
    4. scipy.fft (default, always available)

    Returns
    -------
    str
        Name of the detected backend ('mkl', 'scipy', or other available backend)

    Notes
    -----
    - FFTKIT_BACKEND takes precedence over PYMODAL_FFT_BACKEND for fftkit users
    - PYMODAL_FFT_BACKEND is supported for backward compatibility with modalpy
    - The function gracefully degrades if optional backends are unavailable
    """
    # 1. Check fftkit-specific environment variable (highest priority)
    env_backend = os.environ.get("FFTKIT_BACKEND")
    if env_backend:
        return env_backend.lower()

    # 2. Check legacy modalpy environment variable (documented fallback)
    env_backend = os.environ.get("PYMODAL_FFT_BACKEND")
    if env_backend:
        return env_backend.lower()

    # 3. Try MKL (2-10x faster than scipy on Intel CPUs)
    try:
        import mkl_fft  # noqa: F401
        return "mkl"
    except ImportError:
        pass

    # 4. Default to scipy (always available)
    return "scipy"


# Detect backend at module load time
DEFAULT_BACKEND = detect_backend()
