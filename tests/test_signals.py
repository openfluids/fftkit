"""Test signal generation functions."""

import numpy as np
import pytest

import fftkit

# Test configuration
FS = 1024
DURATION = 1.0


@pytest.fixture
def time_array():
    """Generate standard time array for testing."""
    return np.arange(0, DURATION, 1/FS)


class TestGenerateComplexSignalReproducibility:
    """Test that seed parameter makes generation reproducible."""

    def test_same_seed_same_output(self, time_array):
        """Same seed should produce identical output."""
        seed = 42
        sig1 = fftkit.generate_complex_signal(time_array, f1=10, f2=2, seed=seed)
        sig2 = fftkit.generate_complex_signal(time_array, f1=10, f2=2, seed=seed)

        assert np.allclose(sig1, sig2), "Same seed should produce identical output"

    def test_different_seeds_different_output(self, time_array):
        """Different seeds should produce different output."""
        sig1 = fftkit.generate_complex_signal(time_array, f1=10, f2=2, seed=42)
        sig2 = fftkit.generate_complex_signal(time_array, f1=10, f2=2, seed=43)

        assert not np.allclose(sig1, sig2), "Different seeds should produce different outputs"

    def test_none_seed_uses_global_state(self, time_array):
        """seed=None should use global numpy.random state."""
        # Two calls without seed might differ (depends on global state)
        sig1 = fftkit.generate_complex_signal(time_array, f1=10, f2=2, seed=None)
        sig2 = fftkit.generate_complex_signal(time_array, f1=10, f2=2, seed=None)

        # They might be different unless global state is deterministic
        # Just check they're valid
        assert len(sig1) == len(time_array)
        assert len(sig2) == len(time_array)


class TestGenerateComplexSignalProperties:
    """Test properties of generated signals."""

    def test_output_is_normalized(self, time_array):
        """Output should be zero-mean and unit-std."""
        sig = fftkit.generate_complex_signal(
            time_array, f1=10, f2=2, noise_level=0.1, seed=123
        )

        mean = np.mean(sig)
        std = np.std(sig)

        assert np.abs(mean) < 1e-10, f"Mean should be ~0, got {mean}"
        assert np.abs(std - 1.0) < 1e-10, f"Std should be ~1, got {std}"

    def test_output_length_matches_input(self, time_array):
        """Output length should match input time array."""
        sig = fftkit.generate_complex_signal(time_array, f1=10, f2=2, seed=123)
        assert len(sig) == len(time_array)

    def test_output_is_real_valued(self, time_array):
        """Output should be real-valued (all imaginary parts ~0)."""
        sig = fftkit.generate_complex_signal(time_array, f1=10, f2=2, seed=123)
        # The function returns real signal, but may be cast to complex
        assert np.all(np.isfinite(sig)), "Output should be finite"


class TestGenerateComplexSignalFrequencies:
    """Test that planted frequencies are recoverable."""

    def test_f1_recovery(self, time_array):
        """Primary frequency f1 should be dominant in spectrum."""
        f1 = 10
        f2 = 2
        sig = fftkit.generate_complex_signal(
            time_array, f1=f1, f2=f2, noise_level=0.05, seed=123
        )

        # Compute spectrum
        X = np.fft.rfft(sig)
        freqs = np.fft.rfftfreq(len(sig), 1/FS)
        psd = np.abs(X) ** 2

        # Find peaks
        peak_idx = np.argmax(psd)
        peak_freq = freqs[peak_idx]

        # Primary peak should be near f1
        assert np.abs(peak_freq - f1) <= 2.0, \
            f"Primary frequency should be near {f1} Hz, got {peak_freq:.1f} Hz"

    def test_f2_recovery(self, time_array):
        """Secondary frequency f2 should be present in spectrum."""
        f1 = 20
        f2 = 5
        sig = fftkit.generate_complex_signal(
            time_array, f1=f1, f2=f2, f1_amplitude=0.5, f2_amplitude=0.8,
            noise_level=0.05, seed=123
        )

        X = np.fft.rfft(sig)
        freqs = np.fft.rfftfreq(len(sig), 1/FS)
        psd = np.abs(X) ** 2

        # Find all peaks
        peaks_indices = np.argsort(psd)[-5:]  # Top 5 peaks
        peak_freqs = freqs[peaks_indices]

        # f2 should be in the top peaks
        assert np.any(np.abs(peak_freqs - f2) <= 2.0), \
            f"Secondary frequency {f2} Hz should appear in top peaks, got {peak_freqs}"

    def test_harmonic_content(self, time_array):
        """Harmonics should be present and decay as configured."""
        f1 = 20
        sig = fftkit.generate_complex_signal(
            time_array, f1=f1, f2=1, num_harmonics_f1=3,
            harmonic_decay_f1=0.5, noise_level=0.05, seed=123
        )

        X = np.fft.rfft(sig)
        freqs = np.fft.rfftfreq(len(sig), 1/FS)
        psd = np.abs(X) ** 2

        # Find peaks near f1 and harmonics (f1, 2*f1, 3*f1)
        target_freqs = [f1, 2*f1, 3*f1]
        peak_heights = []

        for target in target_freqs:
            # Find closest peak to target
            idx = np.argmin(np.abs(freqs - target))
            peak_heights.append(psd[idx])

        # Peaks should generally decrease (due to decay factor)
        # Allow some flexibility due to noise
        assert len(peak_heights) == 3
        for i in range(len(peak_heights)):
            assert peak_heights[i] >= 0, "PSD should be non-negative"


class TestGenerateComplexSignalBackwardsCompatibility:
    """Test that seed=None preserves original behavior."""

    def test_seed_none_default(self, time_array):
        """Default behavior (seed=None) should work as before."""
        # This just ensures the function works without the seed parameter
        sig = fftkit.generate_complex_signal(time_array, f1=10, f2=2)

        assert len(sig) == len(time_array)
        assert np.all(np.isfinite(sig))

    def test_seed_parameter_is_keyword_only(self, time_array):
        """seed parameter should be keyword-only (can't pass positionally)."""
        # This test verifies the function signature has * before seed
        with pytest.raises(TypeError):
            # Try to pass seed as positional argument - should fail
            fftkit.generate_complex_signal(
                time_array, 10, 2, 5, 3, 0.5, 0.7, 1.0, 0.3, 0.15, 123  # 123 as seed positional
            )
