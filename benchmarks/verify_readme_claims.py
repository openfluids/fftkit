"""Re-derive every measured number the README asserts.

The README makes specific numerical claims -- a 219x interpolation error ratio,
a 0.355 aliasing figure, a tonality table, a variance-recovery table. Until this
script existed, none of them could be checked without redoing the measurement by
hand, so a claim that drifted with the library would have gone unnoticed.

Run it after any change to resampling, detrending, or the estimators:

    python benchmarks/verify_readme_claims.py

Every block prints the README's wording next to what this build actually
produces. Nothing here asserts; disagreement is reported, not raised, because
the useful output is the size of the discrepancy.
"""

from __future__ import annotations

import numpy as np

import fftkit

RNG_SEED = 12345


def band_power(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    """Integrate a PSD over [lo, hi]. Rectangular sum: DC and Nyquist are full bins."""
    df = float(freqs[1] - freqs[0])
    return float(np.sum(psd[(freqs >= lo) & (freqs <= hi)]) * df)


def jittered_time_base(n: int, fs: float, jitter: float, rng: np.random.Generator) -> np.ndarray:
    """A monotonically increasing, non-uniform time base with the given relative jitter."""
    dt = (1.0 / fs) * (1.0 + jitter * rng.uniform(-1.0, 1.0, size=n))
    return np.cumsum(dt) - dt[0]


def claim_interpolation_error() -> None:
    """README: cubic gives '219x lower band-power error than linear' on a jittered record."""
    print("\n=== Claim: cubic vs linear interpolation, band-power error ===")
    print("README says: 23x lower band-power error than linear interpolation")

    rng = np.random.default_rng(RNG_SEED)
    fs, n = 1000.0, 8192
    t = jittered_time_base(n, fs, jitter=0.30, rng=rng)

    # A band-limited multi-tone: every component sits well inside the band, so a
    # perfect resampler would recover each one's power exactly.
    tones = [(37.0, 1.0), (113.0, 0.7), (211.0, 0.5)]
    x = sum(a * np.sin(2 * np.pi * f * t) for f, a in tones)
    truth = sum(0.5 * a**2 for _, a in tones)  # time-average power of a sine sum

    errors = {}
    for kind in ("linear", "cubic"):
        r = fftkit.spectrum(x, t=t, interpolation=kind, detrend=None, window="boxcar")
        measured = band_power(r.freqs, r.psd, 0.0, 250.0)
        errors[kind] = abs(measured - truth) / truth

    ratio = errors["linear"] / errors["cubic"]
    for kind, err in errors.items():
        print(f"  {kind:6s} relative band-power error: {err:.3e}")
    print(f"  MEASURED RATIO: cubic is {ratio:.1f}x lower error than linear")


def claim_antialias() -> None:
    """README: 400 Hz tone into a 200 Hz band -> 0.000 spurious with filter, 0.355 without."""
    print("\n=== Claim: anti-alias filter on downsampling ===")
    print("README says: 0.000 in-band power with the filter, 0.064 without")

    fs_in, fs_out, duration = 1000.0, 400.0, 8.192
    t = np.arange(0, duration, 1.0 / fs_in)
    x = np.sin(2 * np.pi * 400.0 * t)  # above the 200 Hz output Nyquist

    for antialias in (True, False):
        res = fftkit.resample_uniform(t, x, fs=fs_out, antialias=antialias, fast_length=False)
        freqs, psd = fftkit.periodogram_rfft(res.x, res.fs)
        spurious = band_power(freqs, psd, 0.0, res.fs / 2)
        print(f"  antialias={str(antialias):5s} -> in-band power {spurious:.4f}")


def claim_tonality_table() -> None:
    """README: a 7-row table of tonality and the estimator auto-choice."""
    print("\n=== Claim: tonality drives the estimator choice (fs=1000 Hz, N=8192) ===")
    rng = np.random.default_rng(RNG_SEED)
    fs, n = 1000.0, 8192
    t = np.arange(n) / fs

    def coloured_noise(exponent: float) -> np.ndarray:
        """Noise with a power-law PSD, built by shaping white noise in the frequency domain."""
        white = rng.standard_normal(n)
        spec = np.fft.rfft(white)
        f = np.fft.rfftfreq(n, 1.0 / fs)
        scale = np.zeros_like(f)
        scale[1:] = f[1:] ** (-exponent / 2.0)
        return np.fft.irfft(spec * scale, n=n)

    tone = np.sin(2 * np.pi * 50 * t)
    signals = {
        "Single 50 Hz tone": tone,
        "Three modes (50/120/213 Hz)": (
            np.sin(2 * np.pi * 50 * t) + 0.6 * np.sin(2 * np.pi * 120 * t) + 0.3 * np.sin(2 * np.pi * 213 * t)
        ),
        "Tone + 50% noise": tone + 0.5 * rng.standard_normal(n),
        "Turbulence + shedding tone": coloured_noise(5 / 3) / 3 + tone,
        "Turbulence, f^-5/3": coloured_noise(5 / 3),
        "Steeper broadband, f^-3": coloured_noise(3.0),
        "White noise": rng.standard_normal(n),
    }

    print(f"  {'signal':<30} {'tonality':>9}  chosen")
    for label, sig in signals.items():
        r = fftkit.spectrum(sig, fs=fs)
        print(f"  {label:<30} {r.method_choice.tonality:9.3f}  {r.method}")


def claim_estimator_variance_recovery() -> None:
    """README: a table of variance recovered by each estimator on f^-5/3 and f^-3."""
    print("\n=== Claim: fraction of variance recovered, and effective resolution ===")
    print("README says: periodogram 1.00/1.00, Welch 0.38/0.05, Blackman-Tukey 1.00/1.00")

    rng = np.random.default_rng(RNG_SEED)
    fs, n = 1000.0, 8192

    def coloured_noise(exponent: float) -> np.ndarray:
        white = rng.standard_normal(n)
        spec = np.fft.rfft(white)
        f = np.fft.rfftfreq(n, 1.0 / fs)
        scale = np.zeros_like(f)
        scale[1:] = f[1:] ** (-exponent / 2.0)
        return np.fft.irfft(spec * scale, n=n)

    for label, exponent in (("f^-5/3", 5 / 3), ("f^-3", 3.0)):
        x = coloured_noise(exponent)
        var = float(np.var(x))
        row = {}
        for name, (fn, kwargs) in {
            "periodogram": (fftkit.periodogram_rfft, {}),
            "welch": (fftkit.welch_method, {"nperseg": n // 8}),
            "blackman_tukey": (fftkit.blackman_tukey_rfft, {"nlags": n // 8}),
        }.items():
            freqs, psd = fn(x, fs, **kwargs)
            row[name] = float(np.sum(psd) * (freqs[1] - freqs[0])) / var
        print(f"  {label:8s} " + "  ".join(f"{k}={v:.2f}" for k, v in row.items()))


def claim_welch_flattens_a_tone() -> None:
    """README: 'On a single tone, Welch lowers the peak from 2.215 to 0.324.'"""
    print("\n=== Claim: Welch blunts a coherent peak ===")
    print("README says: peak falls from 2.344 to 0.324")

    fs, n = 1000.0, 8192
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * 50 * t)

    _, psd_p = fftkit.periodogram_rfft(x, fs)
    _, psd_w = fftkit.welch_method(x, fs, nperseg=n // 8)
    print(f"  periodogram peak: {psd_p.max():.3f}")
    print(f"  welch peak      : {psd_w.max():.3f}")


def claim_trapezoid_bias() -> None:
    """README: trapezoid is biased low, '0.9996 at N=64 where the sum is exact'."""
    print("\n=== Claim: sum(psd)*df is exact, np.trapezoid is biased low ===")
    print("README says: 0.9986 at N=64")

    rng = np.random.default_rng(RNG_SEED)
    for n in (64, 256, 1024):
        x = rng.standard_normal(n)
        freqs, psd = fftkit.periodogram_rfft(x, 1.0)
        df = float(freqs[1] - freqs[0])
        var = float(np.var(x))
        print(
            f"  N={n:5d}  sum*df/var = {np.sum(psd) * df / var:.10f}"
            f"   trapezoid/var = {np.trapezoid(psd, freqs) / var:.6f}"
        )


def main() -> None:
    print("Re-deriving the README's measured claims")
    print(f"fftkit {fftkit.__version__}, numpy {np.__version__}, seed {RNG_SEED}")
    claim_interpolation_error()
    claim_antialias()
    claim_tonality_table()
    claim_estimator_variance_recovery()
    claim_welch_flattens_a_tone()
    claim_trapezoid_bias()
    print("\nCompare each MEASURED line against the README wording printed above it.")


if __name__ == "__main__":
    main()
