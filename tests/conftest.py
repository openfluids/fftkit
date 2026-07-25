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
    params = []
    for name in fftkit.get_backend_names():
        reasons = []
        if name not in available:
            reasons.append(f"backend '{name}' not available on this machine")
        elif transform is not None and not BACKENDS[name].supports(transform):
            reasons.append(f"backend '{name}' does not implement '{transform}'")
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


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "gpu: marks tests as GPU-only (skip if GPU unavailable)"
    )
