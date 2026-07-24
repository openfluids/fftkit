import numpy as np
import pytest

import fftkit


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


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "gpu: marks tests as GPU-only (skip if GPU unavailable)"
    )
