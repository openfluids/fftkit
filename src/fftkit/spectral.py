from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import signal
from scipy.fft import rfft, rfftfreq


def periodogram_rfft(x: ArrayLike, fs: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute PSD using periodogram with real FFT."""
    freqs, psd = signal.periodogram(np.asarray(x), fs, scaling='density')
    return freqs, psd


def blackman_tukey_rfft(
    x: ArrayLike, fs: float, nlags: int | None = None, window: str = 'bartlett'
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute PSD using the Blackman-Tukey method.

    The classic Blackman-Tukey estimator windows (tapers) the biased
    autocorrelation sequence before Fourier transforming it, which reduces
    the variance of the periodogram at the cost of frequency resolution.

    Args:
        x: Input signal
        fs: Sampling frequency
        nlags: Number of autocorrelation lags to retain (the window is
            applied to lags -nlags..+nlags). Default: N // 4, a common
            Blackman-Tukey rule of thumb that trades frequency resolution
            for reduced estimator variance. Internally capped at
            (N - 1) // 2 so the mirrored, zero-padded lag sequence used to
            build the PSD never overlaps itself.
        window: Name of the lag (taper) window, passed to
            scipy.signal.get_window. Default 'bartlett' (triangular), the
            traditional Blackman-Tukey choice.

    Returns:
        freqs: Frequency array (matches periodogram_rfft's rfftfreq grid)
        psd: Power spectral density estimate, real-valued and scaled to
            match periodogram_rfft's density ('density') scaling.
    """
    x = np.asarray(x)
    N = len(x)
    if nlags is None:
        nlags = max(1, N // 4)
    nlags = min(nlags, max(1, (N - 1) // 2))

    autocorr_full = signal.correlate(x, x, mode='full') / N
    mid = N - 1
    r = autocorr_full[mid:mid + nlags + 1]  # biased autocorrelation, lags 0..nlags

    # One-sided taper: lag_window[0] is the window's peak (lag 0), tapering
    # to the edge value at lag = nlags.
    lag_window = signal.get_window(window, 2 * nlags + 1, fftbins=False)[nlags:]
    r_windowed = r * lag_window

    # Rebuild a symmetric, real, even sequence (r[-k] == r[k]) of length N so
    # its FFT is real by construction; this mirrors the Wiener-Khinchin
    # relation between autocorrelation and PSD.
    padded = np.zeros(N)
    padded[0] = r_windowed[0]
    padded[1:nlags + 1] = r_windowed[1:]
    padded[N - nlags:] = r_windowed[1:][::-1]

    X = rfft(padded)
    freqs = rfftfreq(N, 1 / fs)
    # Real part only: the PSD of a real, even autocorrelation is real;
    # dividing by fs matches periodogram_rfft's density scaling.
    psd = X.real / fs
    # One-sided density: double every bin except DC, and except Nyquist
    # when N is even (rfft's last bin is the Nyquist bin only if N is
    # even; for odd N there is no distinct Nyquist bin).
    psd[1:] *= 2
    if N % 2 == 0:
        psd[-1] /= 2
    return freqs, psd


def welch_method(
    x: ArrayLike, fs: float, nperseg: int | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute PSD using Welch's method with segment averaging.

    Args:
        x: Input signal
        fs: Sampling frequency
        nperseg: Segment length for averaging. Default None uses scipy's default
                 (256 or len(x) if shorter). Use larger values for finer
                 frequency resolution, smaller for more averaging/smoother PSD.

    Returns:
        freqs: Frequency array
        psd: Power spectral density estimate
    """
    # scipy default is 256, which gives good averaging for most signals
    freqs, psd = signal.welch(np.asarray(x), fs, nperseg=nperseg, scaling='density')
    return freqs, psd


def find_peaks(
    freqs: NDArray[np.float64], psd: NDArray[np.float64], threshold: float = 0.01
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return peak locations and values above a fraction of the maximum."""
    peak_indices = signal.find_peaks(psd, height=max(psd) * threshold)[0]
    return freqs[peak_indices], psd[peak_indices]


def calculate_error(
    detected_peaks: NDArray[np.float64],
    true_peaks: NDArray[np.float64],
    symmetric: bool = False,
) -> np.float64:
    """Mean distance between a detected peak set and the known true peaks.

    Args:
        detected_peaks: Frequencies reported by a peak finder.
        true_peaks: Known ground-truth frequencies.
        symmetric: Whether to also penalise spurious detections. See below.
            Defaults to False, which reproduces the 0.1.x metric exactly.

    Returns:
        Mean distance in the same units as the inputs. Lower is better.

    Note:
        With ``symmetric=False`` (the default, kept for backwards
        compatibility) this walks each TRUE peak to its nearest DETECTED peak
        and averages. That direction alone says nothing about detections which
        correspond to no true peak, so **extra detections are free**: a method
        that reports a peak in every bin scores a perfect 0.0. When comparing
        estimators, the one that over-detects can win on this number while
        being the worse estimator.

        ``symmetric=True`` averages both directions -- true-to-detected and
        detected-to-true -- so a spurious peak costs its distance to the
        nearest real one. Prefer it for ranking methods against each other.

        The empty-``detected_peaks`` case also differs. Under the legacy metric
        it returns ``mean(true_peaks)``, which is a frequency standing in for a
        distance and is not comparable to any real error value; under
        ``symmetric=True`` it returns ``inf``, since detecting nothing is
        unambiguously the worst outcome rather than a middling score.
    """
    detected = np.asarray(detected_peaks, dtype=np.float64).ravel()
    true = np.asarray(true_peaks, dtype=np.float64).ravel()

    if symmetric:
        if detected.size == 0 or true.size == 0:
            return np.float64(np.inf)
        # Chamfer-style mean of both nearest-neighbour directions.
        distances = np.abs(true[:, None] - detected[None, :])
        true_to_detected = distances.min(axis=1).mean()
        detected_to_true = distances.min(axis=0).mean()
        return np.float64(0.5 * (true_to_detected + detected_to_true))

    errors: list[np.float64] = []
    for true_peak in true:
        if detected.size > 0:
            error = min(abs(detected_peak - true_peak) for detected_peak in detected)
            errors.append(error)
        else:
            errors.append(true_peak)
    return np.mean(errors)
