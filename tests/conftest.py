import numpy as np
import pytest

import fftkit
from fftkit.backends import BACKENDS


@pytest.fixture
def sine_wave_1024():
    """Generate a 1024-point sine wave at 50 Hz with amplitude 1.0, fs=1024."""
    fs = 1024
    duration = 1.0
    freq = 50
    amplitude = 1.0
    t = np.arange(0, duration, 1/fs)
    x = amplitude * np.sin(2 * np.pi * freq * t)
    return t, x, fs, freq, amplitude


@pytest.fixture
def available_backends():
    """Get list of available backends for this machine."""
    return fftkit.get_available_backends()


def all_backends_param(transform=None):
    """Build a pytest.mark.parametrize-ready list covering ALL registered
    backends (not just the ones importable on this machine).

    This is the fix for the collection-time trap of
    ``@pytest.mark.parametrize("backend", fftkit.get_available_backends())``:
    that form evaluates ``get_available_backends()`` once, at collection
    time, on whatever machine/CI runner happens to be running the suite. A
    backend that isn't importable there produces *no test case at all* --
    the suite silently shrinks and still reports green. It also means a
    backend that IS installed but lacks a given transform (e.g. tensorflow
    has no fftn) isn't distinguished from one that's simply not installed.

    Instead, this parametrizes over every name in ``fftkit.get_backend_names()``
    (the full static registry) and attaches a ``pytest.mark.skipif`` to each
    case: unavailable/unsupported backends show up as a clearly-named
    SKIPPED test (visible in ``-rs`` output), not an absence.

    Args:
        transform: if given, also skip backends whose registered Backend
            object does not implement this transform name (e.g. 'fftn').
            When None, only backend availability is checked.

    Returns:
        list[pytest.param] suitable for
        ``@pytest.mark.parametrize("backend", all_backends_param(...))``.
    """
    available = set(fftkit.get_available_backends())
    # Accept a single transform name or a sequence, so a test exercising more
    # than one (e.g. an ifft(fft(x)) round trip) skips unless the backend
    # implements every transform it actually calls. Parametrizing such a test
    # on "fft" alone let it run against accelerate, which has no ifft.
    if transform is None:
        required = ()
    elif isinstance(transform, str):
        required = (transform,)
    else:
        required = tuple(transform)

    params = []
    for name in fftkit.get_backend_names():
        reasons = []
        if name not in available:
            reasons.append(f"backend '{name}' not available on this machine")
        else:
            missing = [t for t in required if not BACKENDS[name].supports(t)]
            if missing:
                reasons.append(
                    f"backend '{name}' does not implement " + ", ".join(f"'{t}'" for t in missing)
                )
        params.append(
            pytest.param(
                name,
                marks=pytest.mark.skipif(bool(reasons), reason="; ".join(reasons)),
                id=name,
            )
        )
    return params


def partial_backends_param(transform):
    """Parametrize over backends that ARE available on this machine but do
    NOT implement ``transform`` (e.g. accelerate lacks rfft; tensorflow
    lacks fftn). Used to assert the documented NotImplementedError contract
    for partial backends. Skips (by name) any backend that either isn't
    available at all, or does support the transform -- there is nothing to
    assert in either case.
    """
    available = set(fftkit.get_available_backends())
    params = []
    for name in fftkit.get_backend_names():
        supports = name in available and BACKENDS[name].supports(transform)
        is_available = name in available
        if not is_available:
            reason = f"backend '{name}' not available on this machine"
        elif supports:
            reason = f"backend '{name}' supports '{transform}' (nothing to assert here)"
        else:
            reason = ""
        params.append(
            pytest.param(
                name,
                marks=pytest.mark.skipif(bool(reason), reason=reason),
                id=name,
            )
        )
    return params


# ---------------------------------------------------------------------------
# Declared backend limits
# ---------------------------------------------------------------------------
# accelerate wraps Apple's vDSP through ctypes and is deliberately narrow: a
# 1-D complex forward FFT of power-of-two length, no n= padding/truncation,
# and no norm other than the default. Those limits are declared by the
# backend itself (ValueError / NotImplementedError with specific messages),
# so the matrix tests assert them rather than skipping past them.
#
# They went unnoticed until CI ran on macOS: accelerate is unavailable on
# Linux, so every one of these cases was a silent skip locally and in the
# Linux jobs. 18 tests failed the first time a macOS runner saw them.
POW2_ONLY_BACKENDS = {"accelerate"}
NO_N_ARGUMENT_BACKENDS = {"accelerate"}
DEFAULT_NORM_ONLY_BACKENDS = {"accelerate"}


def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0


def assert_declared_limit(backend, call, *, length=None, n=None, norm=None):
    """Assert a backend's documented refusal, if this call violates one.

    Returns True when a limit applied and the documented exception was raised
    (the caller should stop; there is no numeric result to compare), False
    when the call is within the backend's contract and should be run normally.
    """
    if backend in NO_N_ARGUMENT_BACKENDS and n is not None:
        with pytest.raises(NotImplementedError, match="n="):
            call()
        return True
    if backend in DEFAULT_NORM_ONLY_BACKENDS and norm not in (None, "backward"):
        with pytest.raises(NotImplementedError, match="norm="):
            call()
        return True
    if backend in POW2_ONLY_BACKENDS and length is not None and not is_power_of_two(length):
        with pytest.raises(ValueError, match="power-of-two"):
            call()
        return True
    return False


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "gpu: marks tests as GPU-only (skip if GPU unavailable)"
    )
