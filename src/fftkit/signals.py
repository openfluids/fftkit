from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def generate_complex_signal(
    t: ArrayLike,
    f1: float,
    f2: float,
    num_harmonics_f1: int = 5,
    num_harmonics_f2: int = 3,
    harmonic_decay_f1: float = 0.5,
    harmonic_decay_f2: float = 0.7,
    f1_amplitude: float = 1.0,
    f2_amplitude: float = 0.3,
    noise_level: float = 0.15,
    *,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """
    Generate a quasiperiodic signal with two uncorrelated frequencies and their harmonics.

    Parameters:
    t : array-like
        Time array
    f1 : float
        Main frequency
    f2 : float
        Secondary frequency (much smaller than f1)
    num_harmonics_f1 : int, optional
        Number of harmonics for f1 (default is 5)
    num_harmonics_f2 : int, optional
        Number of harmonics for f2 (default is 3)
    harmonic_decay_f1 : float, optional
        Decay factor for f1 harmonics (default is 0.5)
    harmonic_decay_f2 : float, optional
        Decay factor for f2 harmonics (default is 0.7)
    f1_amplitude : float, optional
        Amplitude of the main frequency component (default is 1.0)
    f2_amplitude : float, optional
        Amplitude of the secondary frequency component (default is 0.3)
    noise_level : float, optional
        Level of random noise to add (default is 0.05)
    seed : int, optional
        Random seed for reproducible noise generation. When None, uses global numpy.random state.
        When specified, creates a local random generator for noise only without affecting
        the global random state (default is None).

    Returns:
    array-like
        The generated quasiperiodic signal
    """
    t = np.asarray(t)

    # Main frequency component (f1) with harmonics
    signal = f1_amplitude * np.sin(2 * np.pi * f1 * t)
    for i in range(2, num_harmonics_f1 + 1):
        harmonic_amplitude = f1_amplitude * (harmonic_decay_f1 ** (i-1))
        signal += harmonic_amplitude * np.sin(2 * np.pi * i * f1 * t)

    # Secondary frequency component (f2) with harmonics
    signal += f2_amplitude * np.sin(2 * np.pi * f2 * t)
    for i in range(2, num_harmonics_f2 + 1):
        harmonic_amplitude = f2_amplitude * (harmonic_decay_f2 ** (i-1))
        signal += harmonic_amplitude * np.sin(2 * np.pi * i * f2 * t)

    # Add random noise
    if seed is not None:
        rng = np.random.default_rng(seed)
        noise = noise_level * rng.standard_normal(len(t))
    else:
        noise = noise_level * np.random.randn(len(t))
    final_signal = signal + noise
    final_signal -= np.mean(final_signal)  # Normalize the signal
    final_signal /= np.std(final_signal)
    return final_signal
