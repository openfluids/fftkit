"""Tests for fftkit.physical: sampling reports, resampling, tonality, and spectra."""

from __future__ import annotations

import numpy as np
import pytest

from fftkit.backends import next_fast_len
from fftkit.physical import (
    MethodChoice,
    ResamplingWarning,
    SamplingReport,
    SpectrumResult,
    UniformResampling,
    choose_method,
    compare_methods,
    describe_sampling,
    resample_uniform,
    spectrum,
    tonality,
)

FS = 1000.0
N = 8192


# ---------------------------------------------------------------------------
# Shared signal fixtures
# ---------------------------------------------------------------------------


def _turbulent(npts, slope, fs=FS, seed=1):
    """Synthetic power-law field: PSD ~ f**slope, random phase."""
    r = np.random.default_rng(seed)
    f = np.fft.rfftfreq(npts, d=1 / fs)
    amp = np.zeros_like(f)
    amp[1:] = f[1:] ** (slope / 2.0)
    phases = r.uniform(0, 2 * np.pi, f.size)
    return np.fft.irfft(amp * np.exp(1j * phases), n=npts)


@pytest.fixture
def t_uniform():
    return np.arange(N) / FS


@pytest.fixture
def tone_signal(t_uniform):
    return np.sin(2 * np.pi * 50 * t_uniform)


@pytest.fixture
def tone_plus_small_noise(t_uniform):
    rng = np.random.default_rng(0)
    return np.sin(2 * np.pi * 50 * t_uniform) + 0.01 * rng.standard_normal(N)


@pytest.fixture
def tone_plus_large_noise(t_uniform):
    rng = np.random.default_rng(0)
    return np.sin(2 * np.pi * 50 * t_uniform) + 0.5 * rng.standard_normal(N)


@pytest.fixture
def three_tones(t_uniform):
    return (
        np.sin(2 * np.pi * 50 * t_uniform)
        + np.sin(2 * np.pi * 120 * t_uniform)
        + np.sin(2 * np.pi * 213 * t_uniform)
    )


@pytest.fixture
def white_noise():
    rng = np.random.default_rng(0)
    return rng.standard_normal(N)


@pytest.fixture
def turbulent_53():
    return _turbulent(N, -5.0 / 3.0)


@pytest.fixture
def turbulent_3():
    return _turbulent(N, -3.0)


@pytest.fixture
def turbulent_plus_tone(t_uniform):
    x = _turbulent(N, -5.0 / 3.0)
    x = x / np.std(x)
    return x + 3 * np.sin(2 * np.pi * 50 * t_uniform)


@pytest.fixture
def variable_dt_record():
    """Non-uniform time base with a documented rng consumption order."""
    rng = np.random.default_rng(0)
    dt = (1 / FS) * (1 + 0.4 * rng.uniform(-1, 1, N))
    tv = np.cumsum(dt)
    xv = np.sin(2 * np.pi * 50 * tv) + 0.1 * np.sin(2 * np.pi * 213 * tv)
    return tv, xv


# ---------------------------------------------------------------------------
# describe_sampling
# ---------------------------------------------------------------------------


class TestDescribeSampling:
    def test_uniform_grid_is_flagged_uniform_with_near_zero_jitter(self, t_uniform):
        report = describe_sampling(t_uniform)
        assert isinstance(report, SamplingReport)
        assert report.is_uniform is True
        assert report.jitter < 1e-9

    def test_variable_dt_is_flagged_non_uniform_with_expected_jitter_band(
        self, variable_dt_record
    ):
        tv, _ = variable_dt_record
        report = describe_sampling(tv)
        assert report.is_uniform is False
        assert 0.2 <= report.jitter <= 0.3

    def test_f_max_defensible_is_strictly_below_nominal_nyquist_for_jittered_record(
        self, variable_dt_record
    ):
        tv, _ = variable_dt_record
        report = describe_sampling(tv)
        assert report.f_max_defensible < report.f_nyquist_nominal

    def test_rejects_non_increasing_time_base(self):
        t = np.array([0.0, 0.1, 0.05, 0.2])
        with pytest.raises(ValueError, match="strictly increasing"):
            describe_sampling(t)

    def test_rejects_fewer_than_two_samples(self):
        with pytest.raises(ValueError, match="at least two samples"):
            describe_sampling(np.array([0.0]))

    def test_summary_returns_nonempty_string(self, t_uniform):
        report = describe_sampling(t_uniform)
        text = report.summary()
        assert isinstance(text, str)
        assert len(text) > 0


# ---------------------------------------------------------------------------
# resample_uniform
# ---------------------------------------------------------------------------


class TestResampleUniform:
    def test_variable_dt_resample_reports_fast_length(self, variable_dt_record):
        tv, xv = variable_dt_record
        result = resample_uniform(tv, xv)
        assert isinstance(result, UniformResampling)
        assert result.fast_length is True
        assert next_fast_len(result.x.size) == result.x.size

    def test_no_extrapolation_grid_stays_inside_source_range(self, variable_dt_record):
        tv, xv = variable_dt_record
        result = resample_uniform(tv, xv)
        assert result.t[0] >= tv[0]
        assert result.t[-1] <= tv[-1]

    def test_default_does_not_antialias_when_not_downsampling(self, variable_dt_record):
        tv, xv = variable_dt_record
        result = resample_uniform(tv, xv)
        assert result.antialiased is False

    def test_fast_length_false_yields_non_fast_length_when_nominal_is_not_fast(self):
        # Pick a duration/fs combination whose nominal point count is not a
        # fast FFT length, so the flag has something to disprove.
        t = np.linspace(0.0, 1.0, 1009)
        x = np.sin(2 * np.pi * 10 * t)
        nominal = int(round((t[-1] - t[0]) * (1.0 / np.median(np.diff(t))))) + 1
        assert next_fast_len(nominal) != nominal, "test setup must pick a non-fast nominal length"
        result = resample_uniform(t, x, fast_length=False)
        assert result.fast_length is False

    def test_unknown_interpolation_raises(self, variable_dt_record):
        tv, xv = variable_dt_record
        with pytest.raises(ValueError, match="unknown interpolation"):
            resample_uniform(tv, xv, interpolation="quadratic")

    def test_nonpositive_fs_raises(self, variable_dt_record):
        tv, xv = variable_dt_record
        with pytest.raises(ValueError, match="must be positive"):
            resample_uniform(tv, xv, fs=0.0)
        with pytest.raises(ValueError, match="must be positive"):
            resample_uniform(tv, xv, fs=-10.0)

    def test_mismatched_lengths_raises(self, variable_dt_record):
        tv, xv = variable_dt_record
        with pytest.raises(ValueError, match="same length"):
            resample_uniform(tv, xv[:-1])

    def test_fs_far_above_source_rate_warns(self, variable_dt_record):
        tv, xv = variable_dt_record
        with pytest.warns(ResamplingWarning, match="exceeds"):
            resample_uniform(tv, xv, fs=100_000.0)

    def test_interpolation_accuracy_ordering_cubic_beats_linear_by_wide_margin(
        self, variable_dt_record
    ):
        """Band-power error (40-60 Hz) against the uniform reference signal,
        ranked cubic < akima < pchip < linear. Cubic must beat linear by at
        least 50x, which is the evidence for defaulting to cubic."""
        tv, xv = variable_dt_record

        # Dense uniform reference sampled directly from the known analytic
        # signal (not from resampling), so the errors below measure the
        # interpolators' fidelity and nothing else.
        fs_ref = 1.0 / np.median(np.diff(tv))
        n_ref = next_fast_len(max(2, int(round((tv[-1] - tv[0]) * fs_ref)) + 1))
        t_ref = np.linspace(tv[0], tv[-1], n_ref)
        x_ref = np.sin(2 * np.pi * 50 * t_ref) + 0.1 * np.sin(2 * np.pi * 213 * t_ref)
        ref_result = spectrum(x_ref, fs=fs_ref, method="periodogram", detrend=None)
        ref_band = ref_result.band_power(40, 60)

        errors = {}
        for interp in ("linear", "cubic", "akima", "pchip", "nearest"):
            resampled = resample_uniform(tv, xv, interpolation=interp, antialias=False)
            result = spectrum(resampled.x, fs=resampled.fs, method="periodogram", detrend=None)
            errors[interp] = abs(result.band_power(40, 60) - ref_band)

        assert errors["cubic"] < errors["akima"] < errors["pchip"] < errors["linear"]
        assert errors["cubic"] * 50 < errors["linear"]


class TestAntiAlias:
    """The anti-alias-on-downsample path: THE critical test for corrupted slopes."""

    @pytest.fixture
    def alias_prone_signal(self):
        t = np.arange(N) / FS
        x = np.sin(2 * np.pi * 20 * t) + np.sin(2 * np.pi * 400 * t)
        return t, x

    def test_antialias_true_suppresses_spurious_power_below_1e_minus_3(
        self, alias_prone_signal
    ):
        t, x = alias_prone_signal
        result = resample_uniform(t, x, fs=200.0, antialias=True)
        assert result.antialiased is True
        spec = spectrum(result.x, fs=result.fs, method="periodogram", detrend=None)
        total = spec.total_power()
        band = spec.band_power(18, 22)
        spurious = total - band
        assert spurious < 1e-3

    def test_antialias_false_warns_and_leaves_large_spurious_power(
        self, alias_prone_signal
    ):
        t, x = alias_prone_signal
        with pytest.warns(ResamplingWarning, match="antialias=False"):
            result = resample_uniform(t, x, fs=200.0, antialias=False)
        assert result.antialiased is False
        spec = spectrum(result.x, fs=result.fs, method="periodogram", detrend=None)
        total = spec.total_power()
        band = spec.band_power(18, 22)
        spurious = total - band
        assert spurious > 0.1


# ---------------------------------------------------------------------------
# tonality
# ---------------------------------------------------------------------------


class TestTonality:
    def test_pure_tone_scores_near_one(self, tone_signal):
        _, psd = _periodogram(tone_signal, FS)
        assert tonality(psd[1:]) == pytest.approx(1.0, abs=1e-6)

    def test_tone_with_small_noise_scores_near_one(self, tone_plus_small_noise):
        _, psd = _periodogram(tone_plus_small_noise, FS)
        assert tonality(psd[1:]) == pytest.approx(1.0, abs=1e-3)

    def test_tone_with_large_noise_scores_around_point_seven(self, tone_plus_large_noise):
        _, psd = _periodogram(tone_plus_large_noise, FS)
        score = tonality(psd[1:])
        assert 0.5 < score < 0.85

    def test_three_tones_scores_near_one(self, three_tones):
        _, psd = _periodogram(three_tones, FS)
        assert tonality(psd[1:]) == pytest.approx(1.0, abs=1e-6)

    def test_white_noise_scores_near_zero(self, white_noise):
        _, psd = _periodogram(white_noise, FS)
        assert tonality(psd[1:]) < 0.05

    def test_turbulent_minus_5_3_scores_near_zero(self, turbulent_53):
        _, psd = _periodogram(turbulent_53, FS)
        assert tonality(psd[1:]) < 0.1

    def test_turbulent_minus_3_scores_near_zero(self, turbulent_3):
        _, psd = _periodogram(turbulent_3, FS)
        assert tonality(psd[1:]) < 0.1

    def test_turbulent_plus_tone_scores_near_point_nine_six(self, turbulent_plus_tone):
        _, psd = _periodogram(turbulent_plus_tone, FS)
        score = tonality(psd[1:])
        assert score > 0.85

    @pytest.mark.parametrize("slope", [-1.0, -5.0 / 3.0, -2.0, -3.0, -4.0])
    def test_power_law_slope_sweep_all_score_broadband(self, slope):
        """Regression guard: earlier buggy versions scored 0.53 for -3 and
        0.68 for -4, routing steep power laws to the wrong estimator."""
        x = _turbulent(N, slope)
        _, psd = _periodogram(x, FS)
        score = tonality(psd[1:])
        assert score < 0.1, f"slope {slope} scored {score:.3f}, expected < 0.1"

    def test_fewer_than_eight_bins_returns_zero(self):
        assert tonality(np.array([1.0, 2.0, 3.0])) == 0.0

    def test_all_zero_input_returns_zero(self):
        assert tonality(np.zeros(64)) == 0.0

    def test_zero_block_does_not_produce_inf_or_nan(self):
        psd = np.ones(256)
        psd[64:96] = 0.0  # a quiet band, e.g. above a filter cutoff
        score = tonality(psd)
        assert np.isfinite(score)
        assert 0.0 <= score <= 1.0


def _periodogram(x, fs, window="hann"):
    from scipy import signal as _signal

    return _signal.periodogram(x, fs=fs, window=window, detrend="linear", scaling="density")


# ---------------------------------------------------------------------------
# choose_method
# ---------------------------------------------------------------------------


class TestChooseMethod:
    @pytest.mark.parametrize(
        "fixture_name, expected_method",
        [
            ("tone_signal", "periodogram"),
            ("tone_plus_small_noise", "periodogram"),
            ("tone_plus_large_noise", "periodogram"),
            ("three_tones", "periodogram"),
            ("white_noise", "blackman_tukey"),
            ("turbulent_53", "blackman_tukey"),
            ("turbulent_3", "blackman_tukey"),
            ("turbulent_plus_tone", "periodogram"),
        ],
    )
    def test_method_selection_matches_measured_ground_truth(
        self, request, fixture_name, expected_method
    ):
        x = request.getfixturevalue(fixture_name)
        choice = choose_method(x, FS)
        assert isinstance(choice, MethodChoice)
        assert choice.method == expected_method

    def test_turbulent_53_and_3_score_below_point_one_tonality(
        self, turbulent_53, turbulent_3
    ):
        # These are THE critical broadband cases: they must NOT be treated as
        # coherent. Earlier buggy tonality implementations scored 0.53 for f^-3
        # and 0.68 for f^-4 and routed them to the periodogram.
        for x in (turbulent_53, turbulent_3):
            choice = choose_method(x, FS)
            assert choice.tonality < 0.1
            assert choice.method == "blackman_tukey"


# ---------------------------------------------------------------------------
# power_recovered
# ---------------------------------------------------------------------------


class TestPowerRecovered:
    @pytest.mark.parametrize(
        "fixture_name",
        [
            "tone_signal",
            "tone_plus_small_noise",
            "tone_plus_large_noise",
            "three_tones",
            "white_noise",
            "turbulent_53",
            "turbulent_3",
            "turbulent_plus_tone",
        ],
    )
    def test_periodogram_recovers_full_variance(self, request, fixture_name):
        x = request.getfixturevalue(fixture_name)
        result = spectrum(x, fs=FS, method="periodogram")
        assert result.power_recovered == pytest.approx(1.0, abs=0.03)

    def test_welch_on_turbulent_53_is_genuinely_lossy(self, turbulent_53):
        result = spectrum(turbulent_53, fs=FS, method="welch")
        assert result.power_recovered < 0.35

    def test_welch_on_turbulent_3_is_severely_lossy(self, turbulent_3):
        result = spectrum(turbulent_3, fs=FS, method="welch")
        assert result.power_recovered < 0.05

    def test_welch_on_tone_recovers_full_variance(self, tone_signal):
        result = spectrum(tone_signal, fs=FS, method="welch")
        assert result.power_recovered == pytest.approx(1.0, abs=0.03)

    def test_power_recovered_is_nan_for_zero_variance_signal(self):
        x = np.ones(256)  # constant signal, detrend='constant' -> zero variance
        result = spectrum(x, fs=FS, method="periodogram", detrend="constant")
        assert np.isnan(result.power_recovered)


# ---------------------------------------------------------------------------
# spectrum: variable-dt integration
# ---------------------------------------------------------------------------


class TestSpectrumVariableDt:
    def test_recovers_50hz_peak_to_within_005hz(self, variable_dt_record):
        tv, xv = variable_dt_record
        result = spectrum(xv, t=tv, method="periodogram")
        peak_idx = np.argmax(result.psd)
        peak_freq = result.freqs[peak_idx]
        assert abs(peak_freq - 50.0) < 0.05

    def test_variable_dt_takes_resample_path(self, variable_dt_record):
        tv, xv = variable_dt_record
        result = spectrum(xv, t=tv)
        assert result.resampling is not None
        assert result.provenance["resampled"] is True


# ---------------------------------------------------------------------------
# spectrum: error handling and fast paths
# ---------------------------------------------------------------------------


class TestSpectrumErrorsAndPaths:
    def test_raises_when_neither_t_nor_fs_given(self, tone_signal):
        with pytest.raises(ValueError, match="pass fs"):
            spectrum(tone_signal)

    def test_raises_on_unknown_method(self, tone_signal):
        with pytest.raises(ValueError, match="unknown method"):
            spectrum(tone_signal, fs=FS, method="bogus")

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError, match="at least two samples"):
            spectrum(np.array([1.0]), fs=FS)

    def test_uniform_grid_with_fs_only_gives_no_resampling(self, tone_signal):
        result = spectrum(tone_signal, fs=FS, method="periodogram")
        assert result.resampling is None

    def test_uniform_t_with_no_fs_takes_no_resample_fast_path(self, t_uniform, tone_signal):
        result = spectrum(tone_signal, t=t_uniform, method="periodogram")
        assert result.resampling is None

    def test_auto_method_populates_method_choice_matching_result_method(
        self, turbulent_53
    ):
        result = spectrum(turbulent_53, fs=FS, method="auto")
        assert result.method_choice is not None
        assert result.method_choice.method == result.method
        assert result.provenance["method_selected_automatically"] is True

    def test_explicit_method_leaves_method_choice_none(self, turbulent_53):
        result = spectrum(turbulent_53, fs=FS, method="welch")
        assert result.method_choice is None
        assert result.provenance["method_selected_automatically"] is False


# ---------------------------------------------------------------------------
# spectrum: detrend
# ---------------------------------------------------------------------------


class TestDetrend:
    def test_all_detrend_modes_run(self, tone_signal):
        for mode in (None, "constant", "linear"):
            result = spectrum(tone_signal, fs=FS, method="periodogram", detrend=mode)
            assert isinstance(result, SpectrumResult)

    def test_linear_detrend_removes_far_more_low_frequency_power_than_none(self):
        t = np.arange(N) / FS
        x = np.sin(2 * np.pi * 50 * t) + 5.0 * (t - t.mean())  # strong linear ramp
        no_detrend = spectrum(x, fs=FS, method="periodogram", detrend=None)
        linear_detrend = spectrum(x, fs=FS, method="periodogram", detrend="linear")

        low_band = (0.5, 5.0)
        low_none = no_detrend.band_power(*low_band)
        low_linear = linear_detrend.band_power(*low_band)
        assert low_linear > 0
        assert low_none / low_linear > 100


# ---------------------------------------------------------------------------
# SpectrumResult: band_power, summary
# ---------------------------------------------------------------------------


class TestSpectrumResultMethods:
    def test_band_power_over_full_range_equals_total_power(self, tone_signal):
        result = spectrum(tone_signal, fs=FS, method="periodogram")
        full = result.band_power(result.freqs[0], result.freqs[-1])
        assert full == pytest.approx(result.total_power(), rel=1e-9)

    def test_summary_nonempty(self, tone_signal):
        result = spectrum(tone_signal, fs=FS, method="periodogram")
        text = result.summary()
        assert isinstance(text, str)
        assert len(text) > 0

    def test_summary_shows_low_recovery_note_for_welch_on_turbulence(self, turbulent_3):
        result = spectrum(turbulent_3, fs=FS, method="welch")
        assert "NOTE" in result.summary()

    def test_summary_omits_low_recovery_note_for_periodogram_on_tone(self, tone_signal):
        result = spectrum(tone_signal, fs=FS, method="periodogram")
        assert "NOTE" not in result.summary()

    def test_summary_reports_the_resampling_it_performed(self, variable_dt_record):
        """The resampling provenance line must actually appear.

        Every other summary() test passes fs= with no t=, so resampling is None
        and this branch never ran. It is the line that tells a reader their data
        was interpolated, with which interpolant and onto how many points --
        which is what makes a published spectrum defensible -- so it needs a
        test of its own rather than being inferred from the resampling tests.
        """
        t, x = variable_dt_record
        result = spectrum(x, t=t, interpolation="cubic")
        assert result.resampling is not None
        text = result.summary()
        assert "resampled: cubic" in text
        assert f"{result.resampling.x.size} points" in text
        assert "fast length" in text

    def test_summary_reports_antialiasing_when_it_was_applied(self, tone_signal):
        """Same branch, the anti-aliased variant: a filtered record must say so."""
        # Downsampling with antialias on (the default) filters and does not warn;
        # only antialias=False or an fs above what the data supports warns.
        result = spectrum(tone_signal, t=np.arange(N) / FS, fs=FS / 5.0)
        assert result.resampling is not None
        assert result.resampling.antialiased
        assert "anti-aliased" in result.summary()


# ---------------------------------------------------------------------------
# compare_methods
# ---------------------------------------------------------------------------


class TestCompareMethods:
    def test_returns_both_methods_and_expected_keys(self, turbulent_53):
        result = compare_methods(turbulent_53, fs=FS)
        assert set(result["methods"].keys()) == {"periodogram", "welch", "blackman_tukey"}
        assert result["recommended"] in {"periodogram", "welch", "blackman_tukey"}
        expected_keys = {
            "df",
            "effective_resolution",
            "n_bins",
            "roughness",
            "total_power",
            "power_recovered",
            "f_min_resolved",
            "peak_freq",
            "peak_value",
        }
        for name in ("periodogram", "welch", "blackman_tukey"):
            assert expected_keys.issubset(result["methods"][name].keys())

    def test_welch_has_fewer_bins_and_larger_df_than_periodogram(self, turbulent_53):
        result = compare_methods(turbulent_53, fs=FS)
        welch = result["methods"]["welch"]
        periodogram = result["methods"]["periodogram"]
        assert welch["n_bins"] < periodogram["n_bins"]
        assert welch["df"] > periodogram["df"]

    def test_welch_has_lower_roughness_than_periodogram_on_turbulent_signal(
        self, turbulent_53
    ):
        result = compare_methods(turbulent_53, fs=FS)
        assert result["methods"]["welch"]["roughness"] < result["methods"]["periodogram"]["roughness"]


# ---------------------------------------------------------------------------
# Blackman-Tukey: the reason it is the broadband default
# ---------------------------------------------------------------------------


class TestBlackmanTukeyIsTheBetterBroadbandEstimator:
    """The default for broadband signals is Blackman-Tukey rather than Welch.

    That is a deliberate departure from the more common choice, so the claim
    behind it gets asserted rather than asserted in a docstring: BT reduces
    variance comparably while keeping the largest scales, which Welch discards
    by segmenting.
    """

    @pytest.mark.parametrize("fixture_name", ["turbulent_53", "turbulent_3"])
    def test_preserves_variance_where_welch_loses_most_of_it(self, request, fixture_name):
        x = request.getfixturevalue(fixture_name)
        bt = spectrum(x, fs=FS, method="blackman_tukey")
        welch = spectrum(x, fs=FS, method="welch")
        # BT integrates to the signal variance; Welch does not come close.
        assert bt.power_recovered == pytest.approx(1.0, abs=0.03)
        assert welch.power_recovered < 0.35
        assert bt.power_recovered > welch.power_recovered * 2

    def test_reduces_scatter_relative_to_the_raw_periodogram(self, turbulent_53):
        result = compare_methods(turbulent_53, fs=FS)
        bt = result["methods"]["blackman_tukey"]
        periodogram = result["methods"]["periodogram"]
        assert bt["roughness"] < periodogram["roughness"]

    def test_effective_resolution_is_reported_honestly_not_as_bin_width(
        self, turbulent_53
    ):
        """BT's output sits on the fine grid but is smoothed over a wider band.

        Reporting df alone would make it look ~8x sharper than Welch when the two
        are in fact matched, so effective_resolution must exceed df and land at
        fs / nlags.
        """
        result = compare_methods(turbulent_53, fs=FS)
        bt = result["methods"]["blackman_tukey"]
        welch = result["methods"]["welch"]
        assert bt["df"] < bt["effective_resolution"]
        assert bt["effective_resolution"] == pytest.approx(FS / bt["nlags"], rel=1e-9)
        # Matched with Welch by construction: both default to N/8.
        assert bt["effective_resolution"] == pytest.approx(welch["df"], rel=0.05)
        # ...while the periodogram's effective resolution really is its bin width.
        assert result["methods"]["periodogram"]["effective_resolution"] == pytest.approx(
            result["methods"]["periodogram"]["df"], rel=1e-9
        )

    def test_estimate_is_non_negative_with_the_default_lag_window(self, turbulent_53):
        """Only some lag windows guarantee a non-negative PSD; bartlett is the
        default precisely because it does."""
        result = spectrum(turbulent_53, fs=FS, method="blackman_tukey")
        assert result.psd.min() >= 0.0

    def test_nlags_controls_resolution(self, turbulent_53):
        coarse = spectrum(turbulent_53, fs=FS, method="blackman_tukey", nlags=256)
        fine = spectrum(turbulent_53, fs=FS, method="blackman_tukey", nlags=2048)
        assert coarse.nlags == 256
        assert fine.nlags == 2048
        # More lags retained means finer effective resolution and more scatter.
        assert FS / fine.nlags < FS / coarse.nlags

    def test_reports_the_lag_window_not_a_data_window(self, turbulent_53):
        """No data taper is applied on this path, so claiming window='hann'
        would misdescribe the estimate."""
        result = spectrum(turbulent_53, fs=FS, method="blackman_tukey", window="hann")
        assert "lag window" in result.window
        assert result.nperseg is None

    def test_nlags_is_none_for_the_other_estimators(self, turbulent_53):
        assert spectrum(turbulent_53, fs=FS, method="welch").nlags is None
        assert spectrum(turbulent_53, fs=FS, method="periodogram").nlags is None

    def test_auto_selects_it_for_broadband_and_records_why(self, turbulent_53):
        result = spectrum(turbulent_53, fs=FS)
        assert result.method == "blackman_tukey"
        assert result.method_choice is not None
        assert "Blackman-Tukey rather than Welch" in result.method_choice.reason
