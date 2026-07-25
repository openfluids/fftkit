"""High-level entry point for spectra of physical signals and simulation output.

The rest of fftkit is a thin, neutral wrapper over FFT backends: it does what you
ask and nothing more. This module is the opposite by design. It takes a time
series from a simulation or an experiment -- possibly on a variable time step --
and returns a power spectral density with the choices a physical analysis needs
already made, plus a record of what those choices were.

The defaults are opinionated because the failure modes are quiet. A spectrum
computed without detrending, without a window, or after resampling that aliased
high-frequency energy downward still looks like a spectrum. It plots on a log-log
axis, it has a slope, and the slope is wrong. Nothing raises.

What ``spectrum()`` does, and why each step is on by default:

1. **Reports the sampling actually present.** Variable-dt output has no single
   Nyquist frequency. The coarsest interval in the record sets the highest
   frequency you can defend, which is usually well below what the median
   interval suggests. Both are reported; nothing is silently assumed.
2. **Resamples onto a uniform grid whose length is already FFT-friendly.** The
   FFT needs uniform spacing. Choosing the number of points to be a fast length
   costs nothing here -- the grid spacing is ours to pick -- and avoids
   zero-padding later, which would perturb the power normalisation and
   interpolate the spectrum onto a grid finer than the data supports.
3. **Detrends.** A mean offset puts all its power in the zero-frequency bin; a
   linear drift smears power across the lowest frequencies, which is exactly
   where large-scale structure lives. Removing both is nearly always what was
   intended.
4. **Windows.** An abrupt record boundary is a discontinuity, and its leakage
   spreads across decades of a log-log spectrum. That matters when reading an
   inertial-range slope.
5. **Returns a density-scaled one-sided PSD**, so that
   ``np.sum(psd) * df == np.var(x_detrended)``. This is the scaling that makes
   integrating a band give the energy in that band.

Every choice is overridable, and every choice that was made is returned in the
result, so a figure can be reproduced and defended.
"""

from __future__ import annotations

import functools
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import signal as _signal

from .backends import next_fast_len
from .spectral import blackman_tukey_rfft

__all__ = [
    "MethodChoice",
    "SamplingReport",
    "SpectrumResult",
    "UniformResampling",
    "choose_method",
    "compare_methods",
    "describe_sampling",
    "resample_uniform",
    "spectrum",
    "tonality",
]

InterpolationName = Literal["linear", "cubic", "akima", "pchip", "nearest"]


class ResamplingWarning(UserWarning):
    """A resampling choice may have altered the physics of the spectrum.

    Emitted for the cases that produce a plausible-looking but wrong answer:
    downsampling without an anti-alias filter, or requesting a sample rate the
    source data cannot support.
    """


@dataclass(frozen=True)
class SamplingReport:
    """What the time base of a record actually looks like.

    ``f_max_defensible`` is the conservative one: with non-uniform sampling the
    largest gap limits the highest frequency you can claim, because within that
    gap the signal is unconstrained. ``f_nyquist_nominal`` follows from the
    median interval and is the optimistic figure people usually quote. When the
    two differ substantially, the band between them is interpolation, not
    measurement.
    """

    n_samples: int
    duration: float
    dt_min: float
    dt_max: float
    dt_median: float
    dt_mean: float
    jitter: float
    """Coefficient of variation of the intervals: std(dt) / mean(dt). Zero for a
    uniform grid."""
    is_uniform: bool
    f_nyquist_nominal: float
    f_max_defensible: float
    f_resolution: float
    """Lowest resolvable frequency, 1 / duration. No amount of padding improves
    it; only a longer record does."""

    def summary(self) -> str:
        kind = "uniform" if self.is_uniform else f"non-uniform (jitter {self.jitter:.1%})"
        return (
            f"{self.n_samples} samples over {self.duration:.6g} s, {kind}\n"
            f"  dt: min {self.dt_min:.6g}  median {self.dt_median:.6g}  max {self.dt_max:.6g}\n"
            f"  frequency range: {self.f_resolution:.6g} Hz (1/T) "
            f"to {self.f_max_defensible:.6g} Hz (defensible)\n"
            f"  nominal Nyquist from median dt: {self.f_nyquist_nominal:.6g} Hz"
        )


def describe_sampling(t: ArrayLike) -> SamplingReport:
    """Characterise a (possibly non-uniform) time base without modifying it.

    Cheap, read-only, and worth calling before any spectral work: it tells you
    which part of a spectrum will be physics and which will be interpolation.
    """
    time = np.asarray(t, dtype=np.float64).ravel()
    if time.size < 2:
        raise ValueError("need at least two samples to describe a time base")
    dt = np.diff(time)
    if np.any(dt <= 0):
        raise ValueError(
            "time values must be strictly increasing; got a non-positive interval "
            f"(min dt = {dt.min():.6g}). Sort the record first."
        )
    dt_mean = float(dt.mean())
    dt_max = float(dt.max())
    dt_median = float(np.median(dt))
    jitter = float(dt.std() / dt_mean) if dt_mean > 0 else 0.0
    # 1e-12 relative: float64 time arrays built by accumulation are never
    # exactly uniform, and treating a 1e-16 wobble as "variable dt" would send
    # every uniformly-sampled record down the resampling path for nothing.
    is_uniform = bool(jitter < 1e-12)
    duration = float(time[-1] - time[0])
    return SamplingReport(
        n_samples=int(time.size),
        duration=duration,
        dt_min=float(dt.min()),
        dt_max=dt_max,
        dt_median=dt_median,
        dt_mean=dt_mean,
        jitter=jitter,
        is_uniform=is_uniform,
        f_nyquist_nominal=1.0 / (2.0 * dt_median),
        f_max_defensible=1.0 / (2.0 * dt_max),
        f_resolution=1.0 / duration if duration > 0 else float("inf"),
    )


@dataclass(frozen=True)
class UniformResampling:
    """A record placed on a uniform grid, with the provenance of how it got there."""

    t: NDArray[np.float64]
    x: NDArray[np.float64]
    fs: float
    interpolation: str
    antialiased: bool
    source: SamplingReport
    fast_length: bool
    """True when the sample count is a length the FFT transforms quickly, so no
    zero-padding is needed downstream."""


def _interpolator(name: str, t: NDArray[np.float64], x: NDArray[np.float64]) -> Any:
    """Build an interpolator by name.

    Default is cubic rather than linear. For a fixed record produced by an
    expensive simulation the interpolation cost is irrelevant next to the cost
    of generating the data (measured: CubicSpline 69 ms versus np.interp 11 ms
    on 2**20 points, against a 5.8 ms transform), so fidelity is the right thing
    to optimise. Linear interpolation acts as a low-pass filter with a sinc^2
    response, attenuating the high-frequency end of the very spectrum being
    measured.
    """
    from scipy.interpolate import Akima1DInterpolator, CubicSpline, PchipInterpolator

    if name == "linear":
        # np.interp, not interp1d(kind="linear"): identical result, measured
        # 1.8x faster. (interp1d(kind="slinear") is also the same interpolant.)
        return lambda target: np.interp(target, t, x)
    if name == "cubic":
        return CubicSpline(t, x)
    if name == "akima":
        return Akima1DInterpolator(t, x, method="akima")
    if name == "pchip":
        return PchipInterpolator(t, x)
    if name == "nearest":
        from scipy.interpolate import interp1d

        return interp1d(t, x, kind="nearest")
    raise ValueError(
        f"unknown interpolation {name!r}; choose from "
        "'cubic', 'linear', 'akima', 'pchip', 'nearest'"
    )


def resample_uniform(
    t: ArrayLike,
    x: ArrayLike,
    fs: float | None = None,
    interpolation: InterpolationName = "cubic",
    antialias: bool = True,
    fast_length: bool = True,
) -> UniformResampling:
    """Place a possibly non-uniform record on a uniform grid fit for an FFT.

    Args:
        t: Sample times, strictly increasing. Not modified.
        x: Sample values. Not modified.
        fs: Target sample rate. Default: the rate implied by the median input
            interval, which neither invents resolution nor discards any.
        interpolation: ``'cubic'`` (default), ``'linear'``, ``'akima'``,
            ``'pchip'`` or ``'nearest'``. See :func:`_interpolator` for why the
            default is cubic.
        antialias: When ``fs`` is below the source rate, low-pass filter before
            decimating. On by default because the alternative folds
            high-frequency energy down into the band you are about to measure a
            slope in, producing a plausible spectrum with the wrong physics.
        fast_length: Choose the sample count to be a length the FFT transforms
            quickly, adjusting the grid spacing by the fraction of a percent
            needed. Free here, and it removes any later need to zero-pad.

    Returns:
        :class:`UniformResampling` with the new grid and a record of the choices.

    Note:
        The grid spans exactly the input record; no extrapolation is performed,
        so the result never contains invented samples beyond the data.
    """
    time = np.asarray(t, dtype=np.float64).ravel()
    values = np.asarray(x, dtype=np.float64).ravel()
    if time.shape != values.shape:
        raise ValueError(f"t and x must have the same length; got {time.shape} and {values.shape}")
    report = describe_sampling(time)

    target_fs = float(fs) if fs is not None else 1.0 / report.dt_median
    if target_fs <= 0:
        raise ValueError(f"fs must be positive, got {target_fs}")

    if fs is not None and target_fs > report.f_nyquist_nominal * 2:
        warnings.warn(
            f"requested fs={target_fs:.6g} Hz exceeds the rate implied by the median "
            f"input interval ({1.0 / report.dt_median:.6g} Hz). The extra points are "
            "interpolation, not measurement, and add no spectral information above "
            f"{report.f_max_defensible:.6g} Hz.",
            ResamplingWarning,
            stacklevel=2,
        )

    n_nominal = max(2, int(round(report.duration * target_fs)) + 1)
    n_target = next_fast_len(n_nominal) if fast_length else n_nominal

    # Spacing is ours to choose, so absorb the change into dt and keep the full
    # record. Truncating to reach a fast length would throw away data from a
    # simulation that may have cost core-hours.
    grid = np.linspace(time[0], time[-1], n_target)
    actual_fs = (n_target - 1) / report.duration if report.duration > 0 else target_fs

    downsampling = actual_fs < (1.0 / report.dt_median) * (1 - 1e-9)
    did_antialias = False
    if downsampling and antialias:
        # Resample onto the source median rate first (uniform), then let
        # resample_poly's FIR do the band-limiting on the way down. Applying a
        # filter directly to non-uniformly spaced samples would be meaningless,
        # since filter coefficients assume a fixed spacing.
        n_dense = max(2, int(round(report.duration / report.dt_median)) + 1)
        dense_grid = np.linspace(time[0], time[-1], n_dense)
        dense = np.asarray(_interpolator(interpolation, time, values)(dense_grid), dtype=np.float64)
        # Rational resampling ratio, kept small to bound filter length.
        from fractions import Fraction

        ratio = Fraction(n_target - 1, n_dense - 1).limit_denominator(1000)
        filtered = _signal.resample_poly(dense, ratio.numerator, ratio.denominator)
        # resample_poly's output length follows the ratio, so trim or pad-by-edge
        # to land exactly on the requested grid.
        resampled = np.interp(
            grid,
            np.linspace(time[0], time[-1], filtered.size),
            filtered,
        )
        did_antialias = True
    elif downsampling and not antialias:
        warnings.warn(
            f"downsampling {1.0 / report.dt_median:.6g} Hz -> {actual_fs:.6g} Hz with "
            "antialias=False. Energy above the new Nyquist folds back into the band, "
            "which corrupts spectral slopes (an inertial-range fit will read wrong) "
            "without raising anything.",
            ResamplingWarning,
            stacklevel=2,
        )
        resampled = np.asarray(_interpolator(interpolation, time, values)(grid), dtype=np.float64)
    else:
        resampled = np.asarray(_interpolator(interpolation, time, values)(grid), dtype=np.float64)

    return UniformResampling(
        t=grid,
        x=resampled,
        fs=actual_fs,
        interpolation=interpolation,
        antialiased=did_antialias,
        source=report,
        fast_length=bool(n_target == next_fast_len(n_target)),
    )


def tonality(psd: ArrayLike, n_blocks: int = 64, prominence: float = 10.0) -> float:
    """Fraction of spectral power sitting in narrow peaks above the local baseline.

    This is the statistic that separates the two kinds of signal that need
    different estimators. It returns roughly 1 for a signal whose power lives in
    a few discrete tones -- a shedding frequency, a growing eigenmode, an
    acoustic resonance -- and roughly 0 for one whose power is spread smoothly
    across a band, as turbulence is.

    The obvious candidate for this job, spectral flatness (geometric mean over
    arithmetic mean), does not work here. A turbulent spectrum falling as
    :math:`f^{-5/3}` spans decades in amplitude, so its geometric mean sits far
    below its arithmetic mean and flatness reports "tonal" for exactly the
    broadband case we need to detect. The discriminator has to be blind to the
    slope.

    So the baseline is estimated as a piecewise median of the spectrum and
    divided out. That removes any smooth power law, whatever its exponent, while
    leaving narrow peaks standing proud of it -- a peak is narrow by definition,
    so it cannot move the median of the block containing it. What remains is the
    share of total power in bins exceeding their local baseline by ``prominence``.

    The blocks are spaced **geometrically**, not uniformly, and that detail is
    load-bearing. Blocked uniformly, the lowest block of a red spectrum spans
    decades of amplitude, its median sits near the block's quiet upper end, and
    the lowest bins tower over it and are scored as tones. Since a red spectrum
    holds nearly all its variance in exactly those bins, the score then runs to
    ~1 for the broadband case this function exists to recognise. (Measured
    before the fix: 0.63 for :math:`f^{-5/3}` and 0.99 for :math:`f^{-3}`, both
    routed to the wrong estimator.) Geometric blocks each span a fixed frequency
    *ratio*, over which a power law varies by a small constant factor, so the
    baseline follows the slope instead of being confused by it.

    Args:
        psd: A power spectral density. The zero-frequency bin should already be
            excluded or detrended; a surviving DC spike reads as a tone.
        n_blocks: Number of geometric blocks for the median baseline. More blocks
            track a curved spectrum more closely but risk a block narrow enough
            for a peak to dominate its own median.
        prominence: How far above baseline a bin must rise to count as tonal.
            10 is one decade in power, ~10 dB.

    Returns:
        Power fraction in tonal bins, in [0, 1].
    """
    power = np.asarray(psd, dtype=np.float64).ravel()
    power = power[np.isfinite(power)]
    if power.size < 8:
        return 0.0
    total = power.sum()
    if total <= 0:
        return 0.0

    # At least 8 bins per block: a median over 2 or 3 bins is not a baseline, and
    # a block containing a single bin defines that bin as its own baseline, which
    # would hide a genuine low-frequency tone.
    min_width = 8
    blocks = max(1, min(n_blocks, power.size // min_width))
    raw = np.geomspace(1, power.size, blocks + 1)
    edges = [0]
    for value in raw[1:]:
        candidate = int(round(value))
        if candidate - edges[-1] >= min_width:
            edges.append(min(candidate, power.size))
    if edges[-1] != power.size:
        if power.size - edges[-1] >= min_width:
            edges.append(power.size)
        else:
            edges[-1] = power.size

    floor = max(float(np.median(power)) * 1e-12, np.finfo(np.float64).tiny)

    centres, medians = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi > lo:
            # Geometric centre, to match the geometric spacing of the edges.
            centres.append(np.sqrt((lo + 1) * hi))
            medians.append(max(float(np.median(power[lo:hi])), floor))

    if len(medians) < 2:
        baseline = np.full(power.size, max(float(np.median(power)), floor))
    else:
        # Interpolate in log-log, not linear. This is what makes the baseline
        # exact for a power law rather than merely close: a power law is a
        # straight line in log-log, and a straight line through nodes lying on
        # it is that same line, whatever the node spacing. Interpolated
        # linearly instead, the chord between two nodes sags below the convex
        # true curve, and the steepest part of the spectrum -- the lowest
        # frequencies -- rises above its own baseline and is scored as tonal.
        # (Measured with linear interpolation and geometric blocks: f^-3 still
        # scored 0.53; with log-log interpolation it scores ~0.)
        index = np.arange(1, power.size + 1, dtype=np.float64)
        baseline = np.exp(
            np.interp(np.log(index), np.log(np.asarray(centres)), np.log(np.asarray(medians)))
        )
    # A block of exact zeros (an empty band above a filter cutoff) gives a zero
    # baseline; anything over it would otherwise divide to infinity and read as
    # a tone.
    baseline = np.maximum(baseline, floor)

    # Score everything except the first block. Those bins are the very sample
    # their own baseline is estimated from, so a "peak" there cannot be
    # distinguished from the spectrum's own low-frequency rolloff -- and in a red
    # spectrum the lowest bin carries most of the variance, with chi-squared
    # scatter of a factor of several on 2 degrees of freedom. Weighting by power
    # therefore let one noisy bin decide the classification: measured on
    # f^-3, the single lowest bin held 52.7% of the total power and exceeded its
    # baseline by 13.9x, tipping the whole record to "tonal" by luck.
    #
    # Excluding one block width is the smallest exclusion that fixes it. Measured
    # across skips of 3/8/16/32 bins: f^-3 scored 0.53/0/0/0 and f^-4
    # 0.68/0/0/0, while a genuine 2 Hz tone held 1.0 until a 32-bin skip
    # swallowed it. One block is the conservative end of that plateau.
    skip = min_width if power.size > 4 * min_width else 0
    scored, floor_line = power[skip:], baseline[skip:]
    scored_total = scored.sum()
    if scored_total <= 0:
        return 0.0

    tonal = scored > floor_line * prominence
    return float(scored[tonal].sum() / scored_total)


def _detrend_arg(
    detrend: str | None,
) -> Callable[[NDArray[Any]], NDArray[Any]] | Literal[False]:
    """Build the ``detrend=`` argument for ``welch``/``periodogram``.

    Passed as a callable rather than the string scipy documents, because
    scipy-stubs declares the string option as
    ``Literal["literal", "constant", False]`` -- ``"literal"`` is a typo for
    ``"linear"`` (scipy-stubs ``signal/_spectral_py.pyi:28``, checked against
    scipy 1.18.0, whose runtime accepts ``"linear"`` perfectly well). So
    ``detrend="linear"`` runs correctly but fails ``mypy --strict``.

    A callable is in the stub's declared union and is exactly what scipy invokes
    per segment internally, so this costs nothing and needs no cast asserting
    something untrue. Revert to the plain string once the stub is fixed.
    """
    if detrend is None:
        return False
    return functools.partial(_signal.detrend, type=detrend)


@dataclass(frozen=True)
class MethodChoice:
    """Which estimator was chosen for a signal, and the evidence for it."""

    method: str
    tonality: float
    threshold: float
    reason: str


def choose_method(
    x: ArrayLike, fs: float, threshold: float = 0.5, window: str = "hann"
) -> MethodChoice:
    """Pick ``periodogram`` or ``welch`` from the character of the signal.

    The trade-off is fixed and well understood: Welch splits the record into
    overlapping segments and averages their periodograms, which reduces the
    variance of the estimate by roughly the number of independent segments but
    coarsens the frequency resolution by the same factor. Whether that is a good
    deal depends entirely on what the signal is.

    - **Coherent, low-noise signals** (much DNS: a shedding tone, a linear
      instability growing at one frequency). The quantity of interest is a sharp
      peak's frequency and amplitude. There is no random scatter to average
      down, so segmenting buys nothing and costs resolution -- and a narrow peak
      spread over fewer, wider bins loses amplitude accuracy too. Periodogram.
    - **Turbulence.** The signal is one realisation of a random process, so a
      raw periodogram has ~100% standard error per bin no matter how long the
      record: more samples buy more bins, each still as noisy. Fitting a
      :math:`-5/3` slope through that scatter is guesswork. Smoothing is the
      only thing that helps. Blackman-Tukey.

    The broadband choice is Blackman-Tukey rather than Welch, which is the more
    common answer, and the reason is measured. Both reduce variance by roughly
    the same amount -- BT's effective resolution is ``fs / nlags``, which at
    ``nlags = N/8`` equals Welch's bin width at ``nperseg = N/8`` -- but Welch
    gets there by segmenting, which raises its lowest resolvable frequency
    eightfold and discards whatever variance lay below it. On synthetic fields:

    ======================  ==========  =========  ==============
    estimator               f^-5/3      f^-3       largest scales
    ======================  ==========  =========  ==============
    periodogram             1.02        1.00       kept
    Welch, nperseg=N/8      0.27        0.04       lost
    Blackman-Tukey, N/8     1.00        1.00       kept
    ======================  ==========  =========  ==============

    (Fraction of signal variance recovered by integrating the estimate.) BT
    smooths by tapering the autocorrelation instead of cutting the record up, so
    the largest scales survive. With the default bartlett lag window the estimate
    is also guaranteed non-negative, which not every window gives.

    **The strongest argument for Welch anyway**, and the reason it stays one
    keyword away: its segments are independent estimates, so they support error
    bars and a direct check for non-stationarity -- compare segment to segment
    and a drifting flow shows itself. BT returns a single smooth curve with
    correlated bins and no comparable diagnostic. If statistical convergence is
    the question rather than the spectrum itself, pass ``method='welch'``.

    Uses :func:`tonality` on a pilot periodogram; the pilot costs one real FFT,
    negligible against the resampling that usually precedes it.

    Args:
        x: Uniformly sampled signal.
        fs: Sample rate.
        threshold: Tonality above which the signal is treated as coherent.
        window: Window for the pilot estimate.

    Returns:
        :class:`MethodChoice` recording the decision and the measured tonality,
        so an automatic choice is never an invisible one.
    """
    values = np.asarray(x, dtype=np.float64).ravel()
    _, pilot = _signal.periodogram(
        values, fs=fs, window=window, detrend=_detrend_arg("linear"), scaling="density"
    )
    # Drop DC only. Whatever the detrend leaves behind lands in the first few
    # bins, and tonality() already declines to score its own first block, which
    # covers them.
    score = tonality(pilot[1:]) if pilot.size > 8 else 0.0
    if score >= threshold:
        return MethodChoice(
            method="periodogram",
            tonality=score,
            threshold=threshold,
            reason=(
                f"{score:.1%} of the power sits in narrow peaks, so the signal is "
                "coherent rather than stochastic. Segment averaging would trade "
                "resolution for variance reduction that is not needed, and would "
                "blunt the peaks being measured."
            ),
        )
    return MethodChoice(
        method="blackman_tukey",
        tonality=score,
        threshold=threshold,
        reason=(
            f"only {score:.1%} of the power is in narrow peaks, so the spectrum is "
            "broadband and each periodogram bin carries ~100% standard error. "
            "Smoothing reduces that scatter, which is what makes a slope fit "
            "meaningful. Blackman-Tukey rather than Welch because it tapers the "
            "autocorrelation instead of segmenting the record, giving comparable "
            "variance reduction while keeping the largest scales (measured: it "
            "recovers 1.00 of the variance on an f^-3 field where Welch at "
            "nperseg=N/8 recovers 0.04). Pass method='welch' if you need "
            "independent segments for error bars or a stationarity check."
        ),
    )


@dataclass(frozen=True)
class SpectrumResult:
    """A power spectral density plus everything that was done to obtain it.

    The provenance fields exist so a figure can be reproduced and defended. A
    spectrum is not interpretable without knowing whether it was detrended,
    which window was applied, and whether the record was resampled.
    """

    freqs: NDArray[np.float64]
    psd: NDArray[np.float64]
    fs: float
    method: str
    window: str
    detrend: str
    nperseg: int | None
    nlags: int | None
    """Autocorrelation lags retained, on the Blackman-Tukey path only. ``None``
    for the other estimators."""
    variance: float
    """Variance of the detrended signal that was analysed. The reference for
    :attr:`power_recovered`."""
    sampling: SamplingReport
    resampling: UniformResampling | None
    provenance: dict[str, Any] = field(default_factory=dict)
    method_choice: MethodChoice | None = None
    """Present when ``method='auto'`` selected the estimator, carrying the
    measured tonality and the reasoning. ``None`` when the caller chose."""

    @property
    def f_max_defensible(self) -> float:
        """Highest frequency the source sampling supports. Above this the
        spectrum is interpolation artefact, not measurement."""
        return self.sampling.f_max_defensible

    def total_power(self) -> float:
        """Integral of the PSD over the resolved band.

        For a periodogram this equals the variance of the analysed signal, and
        comparing the two is a direct check that the scaling is right. For Welch
        it does **not**, and the gap is physical rather than a scaling error --
        see :attr:`power_recovered`.
        """
        if self.freqs.size < 2:
            return float(self.psd.sum())
        return float(np.sum(self.psd) * (self.freqs[1] - self.freqs[0]))

    @property
    def power_recovered(self) -> float:
        """Integrated PSD divided by the variance of the analysed signal.

        Should sit at 1.0 for a periodogram; a departure there means the scaling
        is wrong. For Welch on a steep spectrum it is legitimately far below 1,
        and knowing by how much matters.

        Welch segments the record, so its lowest resolvable frequency is higher
        than the full record's by the number of segments. A red spectrum keeps
        most of its variance below that new limit, and per-segment detrending
        removes more of it. Measured on synthetic fields with ``nperseg = N/8``:
        a :math:`f^{-5/3}` field recovers 0.20 of its variance and :math:`f^{-3}`
        recovers 0.013, against 1.02 and 1.00 for the periodogram on the same
        data.

        That is the accepted price of a converged inertial range, not a defect --
        the energy lost sits at the largest scales, below the segment length.
        But it means an integrated Welch spectrum is not the signal's energy, and
        a value well under 1 here says the largest scales fell outside the
        estimate. If those scales are the object of study, use
        ``method='periodogram'`` or a larger ``nperseg``.
        """
        if self.variance <= 0:
            return float("nan")
        return self.total_power() / self.variance

    def band_power(self, f_lo: float, f_hi: float) -> float:
        """Energy between two frequencies, the reason density scaling matters."""
        mask = (self.freqs >= f_lo) & (self.freqs <= f_hi)
        if self.freqs.size < 2:
            return float(self.psd[mask].sum())
        return float(np.sum(self.psd[mask]) * (self.freqs[1] - self.freqs[0]))

    def summary(self) -> str:
        lines = [
            f"PSD via {self.method}: {self.freqs.size} bins, "
            f"{self.freqs[0]:.6g} to {self.freqs[-1]:.6g} Hz at fs={self.fs:.6g} Hz",
            f"  window={self.window}  detrend={self.detrend}"
            + (f"  nperseg={self.nperseg}" if self.nperseg else ""),
            f"  integrated power {self.total_power():.6g} "
            f"= {self.power_recovered:.3f} x variance",
            f"  trust up to {self.f_max_defensible:.6g} Hz",
        ]
        if np.isfinite(self.power_recovered) and self.power_recovered < 0.9:
            lines.append(
                f"  NOTE: {1 - self.power_recovered:.0%} of the variance lies below the "
                f"lowest resolved frequency ({self.freqs[1] if self.freqs.size > 1 else 0:.6g} Hz). "
                "Raise nperseg or use method='periodogram' to reach the largest scales."
            )
        if self.resampling is not None:
            r = self.resampling
            lines.append(
                f"  resampled: {r.interpolation}"
                + (", anti-aliased" if r.antialiased else "")
                + f", {r.x.size} points"
                + (" (fast length)" if r.fast_length else "")
            )
        return "\n".join(lines)


def spectrum(
    x: ArrayLike,
    t: ArrayLike | None = None,
    fs: float | None = None,
    method: Literal["auto", "welch", "periodogram", "blackman_tukey"] = "auto",
    window: str = "hann",
    detrend: Literal["linear", "constant"] | None = "linear",
    nperseg: int | None = None,
    nlags: int | None = None,
    interpolation: InterpolationName = "cubic",
    antialias: bool = True,
) -> SpectrumResult:
    """Power spectral density of a physical signal, with defaults that suit one.

    This is the entry point for "I have a signal from a simulation and I want its
    spectrum". Pass ``t`` when the record is on a variable time step and it will
    be resampled onto a uniform, FFT-friendly grid first.

    Args:
        x: Signal values.
        t: Sample times. Optional for an already-uniform record, in which case
            pass ``fs``.
        fs: Sample rate. Required if ``t`` is None; otherwise derived from ``t``.
        method: ``'auto'`` (default) measures how much of the power sits in
            narrow peaks and picks accordingly: ``'periodogram'`` for a coherent
            signal, ``'blackman_tukey'`` for a broadband one. See
            :func:`choose_method` for why the broadband pick is not Welch, and
            when to override it with ``'welch'``. The decision and its evidence
            come back in :attr:`SpectrumResult.method_choice`. Force any
            estimator by naming it; use :func:`compare_methods` to see all three
            side by side on the same record.
        window: Taper applied per segment. Default Hann.
        detrend: ``'linear'`` (default) removes offset and drift, ``'constant'``
            removes the mean only, ``None`` disables it. Drift otherwise smears
            power across the lowest frequencies, where large-scale structure is.
        nperseg: Segment length for Welch. Default: an eighth of the record,
            rounded to a fast length, which is a common compromise between
            variance reduction and low-frequency reach.
        nlags: Autocorrelation lags retained by Blackman-Tukey, the knob that
            trades resolution against variance on that path: effective
            resolution is about ``fs / nlags``. Default ``N / 8``, chosen to
            match the default Welch resolution so the two are comparable.
        interpolation: Passed to :func:`resample_uniform`.
        antialias: Passed to :func:`resample_uniform`.

    Returns:
        :class:`SpectrumResult`, carrying the PSD and the full provenance.

    Note:
        ``np.sum(psd) * df`` equals the variance of the detrended, analysed
        signal -- not of the raw input, since detrending removes power by
        design. :meth:`SpectrumResult.total_power` computes it.
    """
    values = np.asarray(x, dtype=np.float64).ravel()
    if values.size < 2:
        raise ValueError("need at least two samples")

    resampling: UniformResampling | None = None
    if t is not None:
        report = describe_sampling(t)
        if report.is_uniform and fs is None:
            analysed, actual_fs = values, 1.0 / report.dt_median
        else:
            resampling = resample_uniform(
                t, values, fs=fs, interpolation=interpolation, antialias=antialias
            )
            analysed, actual_fs = resampling.x, resampling.fs
    else:
        if fs is None:
            raise ValueError("pass fs when t is not given, or pass t to derive it")
        actual_fs = float(fs)
        analysed = values
        step = 1.0 / actual_fs
        report = describe_sampling(np.arange(values.size) * step)

    choice: MethodChoice | None = None
    if method == "auto":
        choice = choose_method(analysed, actual_fs, window=window)
        method = choice.method  # type: ignore[assignment]

    seg = nperseg
    if method == "welch" and seg is None:
        # An eighth of the record gives ~8 independent segments (more with
        # overlap), enough to calm the estimator without giving up an order of
        # magnitude of low-frequency reach. Rounded to a fast length so each
        # segment transform takes the radix path.
        seg = min(analysed.size, next_fast_len(max(16, analysed.size // 8)))

    detrend_arg = _detrend_arg(detrend)
    if method == "welch":
        freqs, psd = _signal.welch(
            analysed, fs=actual_fs, window=window, nperseg=seg,
            detrend=detrend_arg, scaling="density",
        )
    elif method == "periodogram":
        freqs, psd = _signal.periodogram(
            analysed, fs=actual_fs, window=window, detrend=detrend_arg, scaling="density",
        )
        seg = None
    elif method == "blackman_tukey":
        # Detrend explicitly: blackman_tukey_rfft takes the signal as given, and
        # an undetrended mean would dominate the autocorrelation at every lag.
        lags = nlags if nlags is not None else max(8, analysed.size // 8)
        prepared = _signal.detrend(analysed, type=detrend) if detrend is not None else analysed
        # No data taper here, deliberately. BT's variance reduction comes from
        # the lag window applied to the autocorrelation; tapering the data as
        # well would smooth twice and bias the level. `window` is therefore
        # unused on this path, and reported as such.
        freqs, psd = blackman_tukey_rfft(prepared, actual_fs, nlags=lags)
        seg = None
        nlags = lags
    else:
        raise ValueError(
            f"unknown method {method!r}; choose 'periodogram', 'welch' or 'blackman_tukey'"
        )

    # Reference variance for power_recovered, detrended the same way the
    # estimator detrends, so the comparison isolates the estimator's own
    # low-frequency reach rather than re-measuring the trend removal.
    reference = (
        _signal.detrend(analysed, type=detrend) if detrend is not None else analysed
    )

    return SpectrumResult(
        freqs=np.asarray(freqs, dtype=np.float64),
        psd=np.asarray(psd, dtype=np.float64),
        fs=actual_fs,
        method=method,
        # Report the lag window on the BT path, since the data window is unused
        # there. Claiming window='hann' would misdescribe the estimate.
        window="bartlett (lag window)" if method == "blackman_tukey" else window,
        detrend=str(detrend),
        nperseg=seg,
        nlags=nlags if method == "blackman_tukey" else None,
        variance=float(np.var(reference)),
        sampling=report,
        resampling=resampling,
        provenance={
            "n_analysed": int(analysed.size),
            "resampled": resampling is not None,
            "antialiased": bool(resampling.antialiased) if resampling else False,
            "method_selected_automatically": choice is not None,
        },
        method_choice=choice,
    )


def compare_methods(
    x: ArrayLike,
    t: ArrayLike | None = None,
    fs: float | None = None,
    window: str = "hann",
    detrend: Literal["linear", "constant"] | None = "linear",
    **kwargs: Any,
) -> dict[str, Any]:
    """Run every estimator on the same record and quantify the differences.

    The point is to make the resolution-versus-variance trade visible on your own
    data instead of taken on faith. For each estimator it reports:

    - ``df``: bin width.
    - ``effective_resolution``: the frequency separation the estimate can
      actually distinguish. Equal to ``df`` for the periodogram and for Welch,
      but ``fs / nlags`` for Blackman-Tukey, whose output sits on the full fine
      grid while being smoothed over a wider band. Compare estimators on this
      column, not on ``df``, or BT looks eight times sharper than it is.
    - ``roughness``: median absolute difference between neighbouring bins in
      log-power. A measure of visible scatter, and the quantity smoothing buys --
      but read it alongside ``effective_resolution``, because BT's neighbouring
      bins are correlated by construction, so part of its low roughness is
      oversampling rather than extra degrees of freedom.
    - ``power_recovered``: integrated PSD over the signal variance. ~1.0 for the
      periodogram; below 1 for Welch by the share of variance sitting below its
      segment-limited lowest frequency. This is the column that shows what
      averaging costs at the largest scales.
    - ``peak_freq`` / ``peak_value``: where and how high the largest peak is.
      Watch ``peak_value`` fall as resolution coarsens and a narrow tone gets
      spread across wider bins -- the specific cost of averaging a coherent
      signal.

    Also returns the ``recommended`` estimator and the measured ``tonality``, so
    the automatic choice can be checked against the numbers rather than trusted.

    Args:
        x, t, fs: As :func:`spectrum`.
        window, detrend: Held identical across estimators so the comparison is
            of the estimators alone.
        **kwargs: Forwarded to :func:`spectrum` (e.g. ``interpolation``).

    Returns:
        ``{'recommended': str, 'tonality': float, 'reason': str,
        'methods': {name: {...metrics...}}}``.
    """
    results: dict[str, SpectrumResult] = {}
    for name in ("periodogram", "welch", "blackman_tukey"):
        results[name] = spectrum(
            x, t=t, fs=fs, method=name, window=window, detrend=detrend, **kwargs
        )

    reference = results["periodogram"]
    choice = choose_method(
        reference.resampling.x if reference.resampling is not None else np.asarray(x, float).ravel(),
        reference.fs,
        window=window,
    )

    table: dict[str, Any] = {}
    for name, res in results.items():
        psd = res.psd
        positive = psd[psd > 0]
        # Median absolute log-difference between neighbours: scatter, measured in
        # a way that a steep but smooth power law does not inflate.
        roughness = (
            float(np.median(np.abs(np.diff(np.log10(positive))))) if positive.size > 2 else 0.0
        )
        peak = int(np.argmax(psd))
        df = float(res.freqs[1] - res.freqs[0]) if res.freqs.size > 1 else float("nan")
        table[name] = {
            "df": df,
            "effective_resolution": (res.fs / res.nlags) if res.nlags else df,
            "n_bins": int(res.freqs.size),
            "nperseg": res.nperseg,
            "nlags": res.nlags,
            "roughness": roughness,
            "total_power": res.total_power(),
            "power_recovered": res.power_recovered,
            "f_min_resolved": float(res.freqs[1]) if res.freqs.size > 1 else float("nan"),
            "peak_freq": float(res.freqs[peak]),
            "peak_value": float(psd[peak]),
        }

    return {
        "recommended": choice.method,
        "tonality": choice.tonality,
        "reason": choice.reason,
        "variance_of_signal": float(
            np.var(
                reference.resampling.x
                if reference.resampling is not None
                else np.asarray(x, float).ravel()
            )
        ),
        "methods": table,
    }
