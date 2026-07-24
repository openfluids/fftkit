"""
Test mathematical invariants of FFT implementations.

These tests verify that all backends correctly implement:
1. Normalization (unnormalized FFT convention)
2. Parseval's theorem (energy conservation)
3. Inverse round-trip consistency
4. Cross-backend agreement

All tests assume unnormalized FFT: X[k] = sum(x[n] * exp(-2j*pi*k*n/N))
"""

import numpy as np
import pytest
from scipy.fft import ifft
from scipy.fft import rfft as scipy_rfft

import fftkit

# Configuration
FS = 1024
DURATION = 1.0
FREQ = 50
AMPLITUDE = 1.0

# Tolerances (all justified for float64 arithmetic)
# 1e-10: standard float64 machine epsilon ≈ 2.2e-16, allow ~1e-10 relative error
TOL_INVARIANTS = 1e-10  # For normalized comparisons
TOL_ROUND_TRIP = 1e-9   # IFFT round-trip: 1/N scaling + numerical accumulation
# Cross-backend agreement is asserted against the spectrum's own scale rather
# than element-wise; see TestCrossBackendAgreement for why.


def theoretical_fft_peak(amplitude, N):
    """Expected FFT peak for unnormalized sine wave.

    For a sine of amplitude A and length N, FFT peak = A*N/2.
    """
    return amplitude * N / 2


@pytest.fixture
def sine_wave():
    """Generate reference sine wave."""
    t = np.arange(0, DURATION, 1/FS)
    x = AMPLITUDE * np.sin(2 * np.pi * FREQ * t)
    return x, len(x)


class TestNormalization:
    """Test FFT amplitude normalization (unnormalized convention)."""

    @pytest.mark.parametrize("backend", fftkit.get_available_backends())
    def test_pure_sine_peak(self, sine_wave, backend):
        """Verify FFT peak matches theory for pure sine."""
        x, N = sine_wave
        fft_func = fftkit.get_fft_func(backend)
        X = fft_func(x)

        # Find peak near test frequency
        freqs = np.fft.fftfreq(N, 1/FS)
        idx = np.argmin(np.abs(freqs - FREQ))
        amp_measured = np.abs(X[idx])
        amp_theory = theoretical_fft_peak(AMPLITUDE, N)

        # Check with tight relative tolerance (float64 precision)
        assert np.isclose(amp_measured, amp_theory, rtol=1e-9, atol=1e-12), \
            f"{backend}: measured {amp_measured:.6f}, expected {amp_theory:.6f}"


class TestParseval:
    """Test Parseval's theorem (energy conservation)."""

    @pytest.mark.parametrize("backend", fftkit.get_available_backends())
    def test_parseval_complex_fft(self, sine_wave, backend):
        """Verify sum(|x|^2) = (1/N) * sum(|X|^2)."""
        x, N = sine_wave
        fft_func = fftkit.get_fft_func(backend)
        X = fft_func(x)

        E_time = np.sum(np.abs(x) ** 2)
        E_freq = np.sum(np.abs(X) ** 2) / N

        assert np.isclose(E_time, E_freq, rtol=TOL_INVARIANTS, atol=1e-15), \
            f"{backend}: time energy {E_time:.10f}, freq energy {E_freq:.10f}, diff {abs(E_time-E_freq):.2e}"

    def test_parseval_rfft(self, sine_wave):
        """Verify Parseval for real FFT with the one-sided doubling rule.

        Not parametrized over backends: fftkit's registry wraps forward
        complex FFTs only, so there is no per-backend rfft to exercise.
        Parametrizing here would run scipy's rfft N times and report it as
        N backends' worth of coverage.
        """
        x, N = sine_wave

        X = scipy_rfft(x)
        E_time = np.sum(np.abs(x) ** 2)

        # Parseval for rfft: DC and Nyquist not doubled, others doubled
        if N % 2 == 0:
            E_freq = (np.abs(X[0])**2 + np.abs(X[-1])**2 + 2*np.sum(np.abs(X[1:-1])**2)) / N
        else:
            E_freq = (np.abs(X[0])**2 + 2*np.sum(np.abs(X[1:])**2)) / N

        assert np.isclose(E_time, E_freq, rtol=TOL_INVARIANTS, atol=1e-15), \
            f"rfft: time {E_time:.10f}, freq {E_freq:.10f}, diff {abs(E_time-E_freq):.2e}"


class TestInverseRoundTrip:
    """Test ifft(fft(x)) recovery."""

    @pytest.mark.parametrize("backend", fftkit.get_available_backends())
    def test_ifft_consistency(self, sine_wave, backend):
        """Verify ifft(fft(x)) recovers original signal."""
        x, N = sine_wave
        fft_func = fftkit.get_fft_func(backend)
        X = fft_func(x)
        x_recovered = ifft(X).real  # Use scipy's ifft which handles normalization

        assert np.allclose(x, x_recovered, atol=TOL_ROUND_TRIP), \
            f"{backend}: max diff {np.max(np.abs(x - x_recovered)):.2e}"


class TestCrossBackendAgreement:
    """Test agreement across backends."""

    @pytest.mark.parametrize("backend", fftkit.get_available_backends())
    def test_agreement_with_numpy(self, sine_wave, backend):
        """Verify backend agrees with numpy FFT to float64 precision.

        Error is measured against the scale of the spectrum, not per element.
        A pure sine leaves most bins holding nothing but cancellation error
        (|X| ~ 1e-14 against a peak of ~5e2). An element-wise relative
        tolerance on those bins compares noise to noise and reports ratios
        near 1 regardless of how correct the backend is, so it cannot
        distinguish a working FFT from a broken one.

        The meaningful question is whether the two spectra differ by more
        than rounding at the magnitude the signal actually occupies.
        """
        x, N = sine_wave

        if backend == 'numpy':
            pytest.skip("Comparing numpy with itself")

        fft_func = fftkit.get_fft_func(backend)
        X_backend = fft_func(x)
        X_numpy = np.fft.fft(x)

        scale = np.max(np.abs(X_numpy))
        max_abs_err = np.max(np.abs(X_backend - X_numpy))
        rel_to_scale = max_abs_err / scale

        # Measured on scipy vs numpy: 1.7e-16, i.e. under one float64 eps
        # (2.2e-16). 1e-12 leaves four orders of headroom for slower
        # algorithms while still catching any genuine numerical divergence.
        assert rel_to_scale < 1e-12, (
            f"{backend}: max|diff| = {max_abs_err:.3e} against spectrum peak "
            f"{scale:.3e} -> {rel_to_scale:.3e} of scale (limit 1e-12)"
        )


class TestRandomSignalInvariants:
    """Test invariants on random signals (seeded for reproducibility)."""

    @pytest.fixture
    def random_signal(self):
        """Generate random signal with fixed seed."""
        np.random.seed(12345)  # Fixed seed
        N = 256
        x = np.random.randn(N) + 1j * np.random.randn(N)
        return x

    @pytest.mark.parametrize("backend", fftkit.get_available_backends())
    def test_parseval_random(self, random_signal, backend):
        """Parseval on random complex signal."""
        x = random_signal
        N = len(x)
        fft_func = fftkit.get_fft_func(backend)
        X = fft_func(x)

        E_time = np.sum(np.abs(x) ** 2)
        E_freq = np.sum(np.abs(X) ** 2) / N

        assert np.isclose(E_time, E_freq, rtol=TOL_INVARIANTS, atol=1e-12), \
            f"{backend} random: time {E_time:.10f}, freq {E_freq:.10f}"
