"""Test spectral analysis functions.

The old version of this suite only checked that peaks landed within a few
Hz of the true frequency and that PSD output was finite. That is loose
enough that a factor-of-2 scaling error and a wrong (periodic instead of
symmetric) lag window both shipped in a released version without the suite
catching either. These tests assert the actual invariants a PSD estimator
must satisfy instead: total power under the PSD curve equals the signal's
sample variance (the scaling constant, not "some finite number"), and
non-negativity is checked per-window against its real mathematical
precondition rather than blanket-asserted for every window.
"""

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


def _total_power(freqs, psd):
    """Integrate a one-sided 'density'-scaled PSD back to total signal
    power. For a density-scaled PSD sampled on a uniform grid of spacing
    df, sum(psd) * df is the exact discrete counterpart of Parseval's
    theorem (not np.trapezoid: the trapezoidal rule halves the endpoint
    bins' contribution, which is systematically wrong here -- DC and
    Nyquist are still full bins in the underlying discrete sum, not
    half-bins as trapz's edge treatment assumes). Verified analytically
    exact (ratio == 1.0 to float64 precision) for periodogram_rfft, whose
    density scaling is derived directly from Parseval's theorem.
    """
    df = freqs[1] - freqs[0]
    return np.sum(psd) * df


class TestPSDScalingInvariant:
    """np.sum(psd)*df / np.var(x) must be 1.0 -- this is the assertion that
    would have caught the released factor-of-2 scaling bug. Tested at both
    even and odd N since odd N has no distinct Nyquist bin and is a
    separate branch in blackman_tukey_rfft (`if N % 2 == 0: psd[-1] /= 2`).
    """

    # periodogram_rfft's scaling is derived directly from Parseval's
    # theorem (scipy.signal.periodogram, scaling='density'), so this ratio
    # is exact to float64 precision, not merely "close".
    @pytest.mark.parametrize("N", [4096, 4097], ids=["even_N", "odd_N"])
    def test_periodogram_scaling_exact(self, N):
        rng = np.random.default_rng(11)
        x = rng.standard_normal(N)
        freqs, psd = fftkit.periodogram_rfft(x, FS)
        ratio = _total_power(freqs, psd) / np.var(x)
        assert ratio == pytest.approx(1.0, rel=1e-9), (
            f"N={N}: periodogram scaling ratio {ratio:.6f}, expected 1.0 "
            "(a factor-of-2 scaling bug would show up here as ~0.5 or ~2.0)"
        )

    @pytest.mark.parametrize("N", [4096, 4097], ids=["even_N", "odd_N"])
    def test_blackman_tukey_scaling(self, N):
        rng = np.random.default_rng(11)
        x = rng.standard_normal(N)
        freqs, psd = fftkit.blackman_tukey_rfft(x, FS)
        ratio = _total_power(freqs, psd) / np.var(x)
        # 1% tolerance: BT trades some scaling exactness for reduced
        # variance via lag windowing; measured deviation across a 30-seed
        # sweep at this N was < 0.1%, so 1% leaves ample margin while still
        # catching a whole-factor scaling bug (which shows up as ~50-100%).
        assert ratio == pytest.approx(1.0, rel=1e-2), (
            f"N={N}: Blackman-Tukey scaling ratio {ratio:.6f}, expected ~1.0"
        )

    @pytest.mark.parametrize("N", [16384, 16385], ids=["even_N", "odd_N"])
    def test_welch_scaling(self, N):
        # Fixed seed=42 with a large N: Welch's segment-averaged estimate
        # of *total* power still has finite-sample statistical scatter
        # (it is an average over overlapping, windowed segments, not an
        # exact Parseval identity like periodogram_rfft). Measured worst
        # deviation across a 30-seed x {16384,16385} sweep was ~1.0%; 2%
        # leaves headroom for that real estimator variance while still
        # catching a whole-factor (50%+) scaling bug.
        rng = np.random.default_rng(42)
        x = rng.standard_normal(N)
        freqs, psd = fftkit.welch_method(x, FS)
        ratio = _total_power(freqs, psd) / np.var(x)
        assert ratio == pytest.approx(1.0, rel=2e-2), (
            f"N={N}: Welch scaling ratio {ratio:.6f}, expected ~1.0"
        )


class TestBlackmanTukeyWhiteNoiseMean:
    """For white noise, PSD should be flat with mean ~= variance / Nyquist
    (the total power var(x), spread evenly across [0, Nyquist])."""

    def test_bt_mean_psd_matches_variance_over_nyquist(self):
        rng = np.random.default_rng(7)
        N, fs = 2048, 1000
        x = rng.standard_normal(N)
        freqs, psd = fftkit.blackman_tukey_rfft(x, fs)
        nyquist = fs / 2
        expected_mean = np.var(x) / nyquist

        # 5% tolerance: mean of a single-realization estimate over a finite
        # band still has sampling scatter; measured deviation here was <1%,
        # 5% catches gross errors (wrong window normalization, wrong scale
        # factor) without being sensitive to noise-realization luck.
        assert psd.mean() == pytest.approx(expected_mean, rel=0.05)


class TestBlackmanTukeyNonNegativity:
    """A Blackman-Tukey PSD estimate is guaranteed non-negative ONLY when
    the lag window's own Fourier transform is non-negative everywhere:
    the estimate is the convolution of the (always non-negative) true
    periodogram with the window's FT, and a convolution of a non-negative
    function with a signed one can go negative.

    Only the triangular family qualifies. bartlett's FT is the Fejer
    kernel, |sinc|^2, non-negative by construction; parzen's likewise.
    blackman, hann and hamming are raised-cosine windows whose FTs all
    have negative sidelobes, so all three can legitimately produce
    slightly negative PSD values. Measured min(DFT of lag window)/max for
    each, which is what decides membership:

        bartlett  +1.9e-08   -> guaranteed
        parzen    +7.1e-09   -> guaranteed
        blackman  -1.0e-03   -> NOT guaranteed
        hamming   -6.5e-03   -> NOT guaranteed
        hann      -2.6e-02   -> NOT guaranteed

    blackman's -1.0e-03 is a real sidelobe, not roundoff: it is eight
    orders of magnitude above float64 eps and does not shrink with the
    signal scale. Grouping it with bartlett on the strength of one
    measurement on a pure sine (where it happens to come out positive)
    was wrong.

    The tests below assert non-negativity only for the guaranteed windows
    and document the others, rather than asserting a property the
    raised-cosine windows do not have -- which would pressure someone into
    clamping the output and destroying a real diagnostic signal.
    """

    @pytest.mark.parametrize("window", ["bartlett", "parzen"])
    def test_guaranteed_nonnegative_windows_stay_nonnegative(self, window, multi_peak_signal):
        x, _ = multi_peak_signal
        freqs, psd = fftkit.blackman_tukey_rfft(x, FS, window=window)
        # These windows' FTs are non-negative in exact arithmetic, so the
        # only slack needed is float64 rfft roundoff near the FT's zero
        # crossings -- ~1e-12 relative, not the 1e-4 a signed-FT window
        # would need. Keeping this tight is the point: it is what makes
        # the test able to tell a guaranteed window from a signed one.
        assert psd.min() >= -1e-12 * psd.max(), (
            f"{window}: min(psd)={psd.min():.3e}, max(psd)={psd.max():.3e} -- "
            f"{window}'s lag-window FT is non-negative by construction, so "
            "the estimate must not dip below zero beyond float64 roundoff"
        )

    @pytest.mark.parametrize("window", ["hann", "hamming"])
    def test_negative_lobe_windows_can_go_negative(self, window, multi_peak_signal):
        """Documents (does not fix) that hann/hamming legitimately produce
        negative PSD values on some signals, via a concrete reproduction:
        a two-tone signal drives this window's own negative sidelobes hard
        enough to show up as psd < 0 in the resulting estimate.
        """
        x, _ = multi_peak_signal
        freqs, psd = fftkit.blackman_tukey_rfft(x, FS, window=window)
        assert psd.min() < 0, (
            f"{window}: expected a legitimately negative PSD value on this "
            f"two-tone signal (its lag window has negative frequency-domain "
            f"sidelobes), got min(psd)={psd.min():.3e} -- if this now passes "
            f"with min(psd) >= 0, the window's implementation likely changed "
            f"and this characterization should be revisited, NOT 'fixed' by "
            f"clamping output to zero."
        )


class TestPeriodogram:
    """Basic shape/sanity checks for periodogram_rfft (scaling covered by
    TestPSDScalingInvariant above)."""

    def test_periodogram_returns_freqs_and_psd(self, test_sine_signal):
        """Periodogram should return frequencies and PSD."""
        x, _ = test_sine_signal
        freqs, psd = fftkit.periodogram_rfft(x, FS)

        assert len(freqs) == len(psd), "Frequency and PSD arrays should match"
        assert len(freqs) > 0, "Should return non-empty arrays"
        assert np.all(psd >= 0), "periodogram's PSD is |X|^2-based and must be exactly non-negative"

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

    def test_periodogram_peak_locations_two_tone(self, multi_peak_signal):
        """Both planted tones (30, 80 Hz) must be resolvable as the two
        largest PSD peaks (not just 'a' peak near one frequency)."""
        x, _ = multi_peak_signal
        freqs, psd = fftkit.periodogram_rfft(x, FS)
        top2_idx = np.argsort(psd)[-2:]
        top2_freqs = sorted(freqs[top2_idx])
        assert np.abs(top2_freqs[0] - 30) <= 1.0
        assert np.abs(top2_freqs[1] - 80) <= 1.0


class TestWelchMethod:
    """Basic shape/sanity checks for welch_method (scaling covered by
    TestPSDScalingInvariant above)."""

    def test_welch_returns_freqs_and_psd(self, test_sine_signal):
        """Welch should return frequencies and PSD."""
        x, _ = test_sine_signal
        freqs, psd = fftkit.welch_method(x, FS)

        assert len(freqs) == len(psd), "Frequency and PSD arrays should match"
        assert len(freqs) > 0, "Should return non-empty arrays"
        assert np.all(psd >= 0), "Welch's PSD is |X|^2-based and must be exactly non-negative"

    def test_welch_peak_location(self, test_sine_signal):
        """Peak in PSD should be near test frequency."""
        x, _ = test_sine_signal
        freqs, psd = fftkit.welch_method(x, FS)

        peak_idx = np.argmax(psd)
        peak_freq = freqs[peak_idx]

        assert np.abs(peak_freq - TEST_FREQ) <= 2.0, \
            f"Peak at {peak_freq:.1f} Hz, expected near {TEST_FREQ} Hz"

    def test_welch_peak_locations_two_tone(self, multi_peak_signal):
        """Both tones must be resolvable, but only with adequate frequency
        resolution: the default nperseg=256 on this 2048-sample signal
        gives df ~ 4 Hz, too coarse to separate 30 Hz from its own
        sidelobes -- nperseg=1024 (df ~1 Hz) is the minimum that actually
        resolves both planted tones, verified by direct inspection.
        """
        x, _ = multi_peak_signal
        freqs, psd = fftkit.welch_method(x, FS, nperseg=1024)
        top2_idx = np.argsort(psd)[-2:]
        top2_freqs = sorted(freqs[top2_idx])
        assert np.abs(top2_freqs[0] - 30) <= 2.0
        assert np.abs(top2_freqs[1] - 80) <= 2.0

    def test_welch_with_custom_nperseg(self, test_sine_signal):
        """Welch should accept custom segment length."""
        x, _ = test_sine_signal
        nperseg = 256
        freqs, psd = fftkit.welch_method(x, FS, nperseg=nperseg)

        assert len(freqs) > 0
        assert len(psd) == len(freqs)


class TestBlackmanTukey:
    """Basic shape/sanity checks for blackman_tukey_rfft (scaling and
    non-negativity covered by the dedicated classes above)."""

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

    def test_blackman_tukey_peak_locations_two_tone(self, multi_peak_signal):
        """Both tones must be visible as local maxima near the true
        frequencies. BT's default nlags=N//4 heavily smooths the estimate
        (that is the whole point of the method), so unlike periodogram's
        tight 1 Hz tolerance, this uses find_peaks with a low threshold and
        a wider frequency window consistent with BT's reduced resolution.
        """
        x, _ = multi_peak_signal
        freqs, psd = fftkit.blackman_tukey_rfft(x, FS)
        detected_freqs, _ = fftkit.find_peaks(freqs, psd, threshold=0.05)
        for target in (30, 80):
            closest = min(detected_freqs, key=lambda f: abs(f - target))
            assert np.abs(closest - target) <= 5.0, (
                f"expected a detected peak near {target} Hz, closest was {closest:.1f} Hz"
            )


class TestFindPeaks:
    """Test peak detection."""

    def test_find_peaks_keyword_is_freqs_not_st(self, multi_peak_signal):
        """0.2.0 renamed find_peaks' first parameter from 'st' to 'freqs'.
        Calling it by the new keyword must work; calling by the removed old
        keyword must raise TypeError -- pins the rename so it can't silently
        regress back.
        """
        x, _ = multi_peak_signal
        freqs, psd = fftkit.periodogram_rfft(x, FS)

        # New keyword works.
        fftkit.find_peaks(freqs=freqs, psd=psd, threshold=0.01)

        # Old keyword ('st') no longer exists.
        with pytest.raises(TypeError):
            fftkit.find_peaks(st=freqs, psd=psd, threshold=0.01)

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

    def test_calculate_error_empty_detected_equals_mean_true_peaks(self):
        """Characterization test, NOT a spec: when detected_peaks is empty,
        the current implementation's per-true-peak fallback is
        `errors.append(true_peak)`, so the returned value is literally
        `mean(true_peaks)` -- a frequency magnitude, not an actual distance
        error. This is a real asymmetry in the function and is pinned here
        deliberately so a future change to this behavior is a conscious
        decision, not an accidental regression.
        """
        true_peaks = np.array([30.0, 80.0])
        detected_peaks = np.array([])

        error = fftkit.calculate_error(detected_peaks, true_peaks)
        assert error == pytest.approx(np.mean(true_peaks))

    def test_calculate_error_ignores_spurious_detections(self):
        """calculate_error only ever measures, for each true peak, the
        distance to its *nearest* detected peak. Extra detected peaks that
        don't correspond to any true peak (spurious/false-positive
        detections) are never penalized -- adding a wildly-wrong extra
        detected peak must leave the error unchanged from the exact-match
        case. This is a real limitation of the current metric (it is not a
        precision/recall score), documented here rather than fixed, since
        src/ is out of scope for this test-only change.
        """
        true_peaks = np.array([30.0, 80.0])
        detected_exact = np.array([30.0, 80.0])
        detected_with_spurious = np.array([30.0, 80.0, 500.0])  # 500 Hz is bogus

        error_exact = fftkit.calculate_error(detected_exact, true_peaks)
        error_with_spurious = fftkit.calculate_error(detected_with_spurious, true_peaks)

        assert error_exact == pytest.approx(0.0, abs=1e-9)
        assert error_with_spurious == pytest.approx(0.0, abs=1e-9), (
            "a spurious extra detected peak must not change the error, "
            "since calculate_error only matches true peaks to their nearest "
            "detection and never penalizes unmatched detections"
        )

    def test_calculate_error_nearest_match_not_first_match(self):
        """Each true peak is matched independently to whichever detected
        peak is nearest -- not to detected peaks in index order and not
        exclusively (two true peaks may both match the same detected peak).
        """
        true_peaks = np.array([30.0, 32.0])
        detected_peaks = np.array([31.0])  # equidistant-ish single detection

        error = fftkit.calculate_error(detected_peaks, true_peaks)
        # Both true peaks are 1 Hz from the single detected peak.
        assert error == pytest.approx(1.0, abs=1e-9)


class TestCalculateErrorSymmetric:
    """The symmetric=True metric penalises spurious detections.

    The default (legacy) metric measures only true-to-detected distance, so
    extra detections cost nothing. That is not a hypothetical: on a real
    benchmark signal, Blackman-Tukey scored better than the periodogram while
    reporting peaks at unrelated frequencies, because over-detecting can only
    help. These tests pin the corrected direction.
    """

    TRUE = np.array([0.10, 0.20, 0.30])

    def test_exact_match_is_zero_both_ways(self):
        detected = np.array([0.10, 0.20, 0.30])
        assert fftkit.calculate_error(detected, self.TRUE, symmetric=True) == pytest.approx(0.0)

    def test_uniform_shift_is_reported_as_the_shift(self):
        """A rigid 0.01 shift should read as 0.01 under either direction, since
        every nearest-neighbour distance is exactly the shift."""
        detected = self.TRUE + 0.01
        assert fftkit.calculate_error(detected, self.TRUE, symmetric=True) == pytest.approx(0.01)

    def test_spurious_peak_is_penalised_only_when_symmetric(self):
        """The whole point: a perfect detection plus one bogus extra peak."""
        detected = np.array([0.10, 0.20, 0.30, 0.90])
        legacy = fftkit.calculate_error(detected, self.TRUE)
        symmetric = fftkit.calculate_error(detected, self.TRUE, symmetric=True)
        assert legacy == pytest.approx(0.0), "legacy metric is blind to spurious peaks"
        assert symmetric > 0.0, "symmetric metric must charge for the spurious peak"

    def test_detecting_everything_does_not_score_perfectly(self):
        """Reporting a peak in every bin trivially minimises the legacy metric.
        Any metric usable for ranking estimators must reject this."""
        detected = np.arange(0.0, 1.0, 0.001)
        assert fftkit.calculate_error(detected, self.TRUE) == pytest.approx(0.0)
        shifted = fftkit.calculate_error(self.TRUE + 0.01, self.TRUE, symmetric=True)
        everything = fftkit.calculate_error(detected, self.TRUE, symmetric=True)
        assert everything > shifted, (
            "detecting every bin must score WORSE than a small honest shift; "
            "under the legacy metric it scores better, which is the defect"
        )

    def test_empty_detection_is_infinite_not_middling(self):
        """Legacy returns mean(true_peaks) -- a frequency posing as a distance,
        which lands in the middle of the plausible error range. inf is the
        honest answer for 'found nothing'."""
        empty = np.array([])
        assert fftkit.calculate_error(empty, self.TRUE, symmetric=True) == np.inf
        assert fftkit.calculate_error(empty, self.TRUE) == pytest.approx(np.mean(self.TRUE))

    def test_symmetric_is_order_independent(self):
        """A mean of both directions must not depend on argument order for
        equal-size sets, or the metric is not the symmetric thing it claims."""
        a = np.array([0.10, 0.21, 0.32])
        b = np.array([0.11, 0.20, 0.30])
        assert fftkit.calculate_error(a, b, symmetric=True) == pytest.approx(
            fftkit.calculate_error(b, a, symmetric=True)
        )
