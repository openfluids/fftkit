"""Test spectral analysis functions."""

import numpy as np
import pytest

import fftkit

# Test configuration
FS = 1024        # Sampling frequency
TEST_FREQ = 50   # Test signal frequency
DURATION = 2.0   # Signal duration for better frequency resolution
AMPLITUDE = 1.0


@pytest.fixture
def test_sine_signal():
    """Generate a pure sine wave with known frequency."""
    t = np.arange(0, DURATION, 1/FS)
    x = AMPLITUDE * np.sin(2 * np.pi * TEST_FREQ * t)
    return x, t


@pytest.fixture
def multi_peak_signal():
    """Generate signal with multiple known peaks for find_peaks testing."""
    t = np.arange(0, DURATION, 1/FS)
    # Two frequency components
    x = 0.7 * np.sin(2 * np.pi * 30 * t) + 0.5 * np.sin(2 * np.pi * 80 * t)
    return x, t


class TestPeriodogram:
    """Test periodogram_rfft function."""

    def test_periodogram_returns_freqs_and_psd(self, test_sine_signal):
        """Periodogram should return frequencies and PSD."""
        x, _ = test_sine_signal
        freqs, psd = fftkit.periodogram_rfft(x, FS)

        assert len(freqs) == len(psd), "Frequency and PSD arrays should match"
        assert len(freqs) > 0, "Should return non-empty arrays"
        assert np.all(psd >= 0), "PSD should be non-negative"

    def test_periodogram_peak_location(self, test_sine_signal):
        """Peak in PSD should be near test frequency."""
        x, _ = test_sine_signal
        freqs, psd = fftkit.periodogram_rfft(x, FS)

        # Find peak
        peak_idx = np.argmax(psd)
        peak_freq = freqs[peak_idx]

        # Peak should be within 1 Hz of test frequency (coarse resolution)
        assert np.abs(peak_freq - TEST_FREQ) <= 1.0, \
            f"Peak at {peak_freq:.1f} Hz, expected near {TEST_FREQ} Hz"


class TestWelchMethod:
    """Test Welch's method for PSD estimation."""

    def test_welch_returns_freqs_and_psd(self, test_sine_signal):
        """Welch should return frequencies and PSD."""
        x, _ = test_sine_signal
        freqs, psd = fftkit.welch_method(x, FS)

        assert len(freqs) == len(psd), "Frequency and PSD arrays should match"
        assert len(freqs) > 0, "Should return non-empty arrays"
        assert np.all(psd >= 0), "PSD should be non-negative"

    def test_welch_peak_location(self, test_sine_signal):
        """Peak in PSD should be near test frequency."""
        x, _ = test_sine_signal
        freqs, psd = fftkit.welch_method(x, FS)

        peak_idx = np.argmax(psd)
        peak_freq = freqs[peak_idx]

        assert np.abs(peak_freq - TEST_FREQ) <= 2.0, \
            f"Peak at {peak_freq:.1f} Hz, expected near {TEST_FREQ} Hz"

    def test_welch_with_custom_nperseg(self, test_sine_signal):
        """Welch should accept custom segment length."""
        x, _ = test_sine_signal
        nperseg = 256
        freqs, psd = fftkit.welch_method(x, FS, nperseg=nperseg)

        assert len(freqs) > 0
        assert len(psd) == len(freqs)


class TestBlackmanTukey:
    """Test Blackman-Tukey PSD estimation."""

    def test_blackman_tukey_returns_freqs_and_psd(self, test_sine_signal):
        """Blackman-Tukey should return frequencies and PSD."""
        x, _ = test_sine_signal
        freqs, psd = fftkit.blackman_tukey_rfft(x, FS)

        assert len(freqs) == len(psd), "Frequency and PSD arrays should match"
        assert len(freqs) > 0, "Should return non-empty arrays"
        assert np.all(np.isfinite(psd)), "PSD should be finite"

    def test_blackman_tukey_peak_location(self, test_sine_signal):
        """Peak should be near test frequency."""
        x, _ = test_sine_signal
        freqs, psd = fftkit.blackman_tukey_rfft(x, FS)

        peak_idx = np.argmax(psd)
        peak_freq = freqs[peak_idx]

        # BT method may be smoother, so allow slightly larger tolerance
        assert np.abs(peak_freq - TEST_FREQ) <= 3.0, \
            f"Peak at {peak_freq:.1f} Hz, expected near {TEST_FREQ} Hz"


class TestFindPeaks:
    """Test peak detection."""

    def test_find_peaks_detects_peaks(self, multi_peak_signal):
        """find_peaks should detect planted peaks."""
        x, _ = multi_peak_signal
        freqs, psd = fftkit.periodogram_rfft(x, FS)

        detected_freqs, detected_psd = fftkit.find_peaks(freqs, psd, threshold=0.01)

        assert len(detected_freqs) > 0, "Should detect at least one peak"
        assert len(detected_psd) == len(detected_freqs)

    def test_find_peaks_identifies_two_components(self, multi_peak_signal):
        """Should identify both frequency components."""
        x, _ = multi_peak_signal
        freqs, psd = fftkit.periodogram_rfft(x, FS)

        detected_freqs, _ = fftkit.find_peaks(freqs, psd, threshold=0.01)

        # Should have at least 2 peaks
        assert len(detected_freqs) >= 2, f"Expected 2+ peaks, got {len(detected_freqs)}"

        # Peaks should be near 30 and 80 Hz
        expected = [30, 80]
        for exp_freq in expected:
            closest = min(detected_freqs, key=lambda f: abs(f - exp_freq))
            assert np.abs(closest - exp_freq) <= 2.0, \
                f"Expected peak near {exp_freq} Hz, closest is {closest:.1f} Hz"


class TestCalculateError:
    """Test error calculation between detected and true peaks."""

    def test_calculate_error_exact_match(self):
        """Error should be ~0 for exact matches."""
        true_peaks = np.array([30.0, 80.0])
        detected_peaks = np.array([30.0, 80.0])

        error = fftkit.calculate_error(detected_peaks, true_peaks)
        assert error < 0.01, f"Exact match should have ~0 error, got {error}"

    def test_calculate_error_with_shift(self):
        """Error should be positive for shifted peaks."""
        true_peaks = np.array([30.0, 80.0])
        detected_peaks = np.array([31.0, 81.0])  # Shifted by 1 Hz

        error = fftkit.calculate_error(detected_peaks, true_peaks)
        assert 0.5 < error < 2.0, f"Shifted peaks should have error ~1, got {error}"

    def test_calculate_error_larger_shift(self):
        """Larger shifts should produce larger errors."""
        true_peaks = np.array([30.0])
        detected_small = np.array([30.5])
        detected_large = np.array([35.0])

        error_small = fftkit.calculate_error(detected_small, true_peaks)
        error_large = fftkit.calculate_error(detected_large, true_peaks)

        assert error_large > error_small, "Larger shift should produce larger error"

    def test_calculate_error_empty_detected(self):
        """Should handle case with no detected peaks."""
        true_peaks = np.array([30.0, 80.0])
        detected_peaks = np.array([])

        error = fftkit.calculate_error(detected_peaks, true_peaks)
        # Error should be dominated by the true peak values
        assert error > 0, "Should return non-zero error for missing detections"
