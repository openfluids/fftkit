"""
Zero-padding: what it buys, what it does not, and two ways to get it wrong.

Four questions, each producing one figure and one JSON block:

2a. SPEED. Padding a transform length up to the next "fast" length
    (`fftkit.next_fast_len`) can be dramatically faster than transforming an
    awkward length directly, because the FFT falls back to Bluestein's
    algorithm (or a direct DFT) on lengths with large prime factors. Measured
    across every available backend here, since radix preferences differ
    (scipy/numpy/MKL/FFTW favor 2, 3, 5, 7, 11; cuFFT favors 2, 3, 5, 7 only
    -- see `fftkit.next_fast_len`'s docstring).

2b. RESOLUTION. Padding does NOT add resolution. Two tones closer together
    than the Rayleigh limit (1/T, set by the observation window T = N*dt, not
    by the transform length) stay unresolved no matter how much zero-padding
    is applied. What padding DOES buy: the padded spectrum is a finer
    interpolation of the same underlying continuous transform, so the
    location *estimate* of the (still merged) peak improves with padding
    even though it never splits into two.

2c. ORDER OF OPERATIONS. Window first, then pad, is correct: the window
    tapers the real data's edges to zero, so appending more zeros afterwards
    introduces no new discontinuity. Pad first, then window, is wrong: it
    stretches the window shape across the padded (real + zero) length, so
    the window's tapering region moves into the zero tail and the real
    data's edges are left with a comparatively sharp cut against the window
    body, corrupting the sidelobe structure the window was supposed to fix.

2d. NORMALIZATION. fftkit's PSD estimators are scaled so that
    `np.sum(psd) * df == np.var(x)` for whatever array `x` was actually
    transformed (exact for `periodogram_rfft`; see tests/test_spectral.py).
    Zero-padding dilutes that same array's own variance (more samples, same
    sum of squares), so the invariant still holds exactly against the
    *padded* array's variance, but the padded PSD's total power undershoots
    the *original* (unpadded) signal's variance by roughly the padding
    ratio. A user who pads for speed and then checks power against the
    original signal's variance will see a mismatch that grows with the pad
    factor -- not a bug, but worth knowing.

All numbers below are computed at run time on this machine; none are copied
from prior measurements. Results are machine- and library-build-dependent
(CPU, scipy/MKL/FFTW version) -- do not compare absolute timings across
machines.
"""

import argparse
import json
import os
import timeit

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks as scipy_find_peaks
from scipy.signal import get_window
from tabulate import tabulate

import fftkit
from fftkit import get_available_backends, get_fft_func, next_fast_len, periodogram_rfft

# -----------------------------------------------------------------------
# 2a. Padding buys speed
# -----------------------------------------------------------------------

def _time_fft(fft_func, x, n=None, reps=5, discard=1):
    """Median wall time per call, discarding the first (warm-up) rep."""
    times = []
    for i in range(reps + discard):
        t = timeit.timeit(lambda: fft_func(x, n=n), number=3)
        if i >= discard:
            times.append(t / 3)
    return float(np.median(times))


def section_speed(args, out_dir):
    sizes = [1009, 10007, 65536] if args.quick else \
        [1009, 8191, 10007, 16384, 65521, 65536, 131071, 262139, 262144]
    reps = 2 if args.quick else 5
    backends = get_available_backends()
    print(f"\n{'=' * 70}\n2a. PADDING BUYS SPEED\n{'=' * 70}")
    print(f"Backends: {backends}")

    rows = []
    per_backend = {b: {'sizes': [], 'raw_ms': [], 'padded_ms': [], 'padded_n': [], 'speedup': []}
                   for b in backends}

    for n in sizes:
        n_fast = next_fast_len(n)
        x = np.random.randn(n).astype(np.complex128)
        for backend in backends:
            fft_func = get_fft_func(backend)
            try:
                t_raw = _time_fft(fft_func, x, n=None, reps=reps)
                t_pad = _time_fft(fft_func, x, n=n_fast, reps=reps)
            except Exception as e:
                print(f"  {backend} failed at N={n}: {e}")
                continue
            speedup = t_raw / t_pad if t_pad > 0 else float('nan')
            per_backend[backend]['sizes'].append(n)
            per_backend[backend]['raw_ms'].append(t_raw * 1e3)
            per_backend[backend]['padded_ms'].append(t_pad * 1e3)
            per_backend[backend]['padded_n'].append(n_fast)
            per_backend[backend]['speedup'].append(speedup)
            extra_pct = 100 * (n_fast - n) / n
            rows.append([backend, n, n_fast, f"{extra_pct:.2f}%", t_raw * 1e3, t_pad * 1e3, speedup])

    print(tabulate(rows, headers=['Backend', 'N', 'next_fast_len(N)', '+points',
                                   'raw [ms]', 'padded [ms]', 'speedup'],
                    floatfmt=('', '', '', '', '.4f', '.4f', '.2f')))

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(backends)))
    for color, backend in zip(colors, backends):
        d = per_backend[backend]
        if not d['sizes']:
            continue
        ax.plot(d['sizes'], d['raw_ms'], marker='o', linestyle='-', color=color,
                label=f'{backend} (raw N)')
        ax.plot(d['padded_n'], d['padded_ms'], marker='s', linestyle='--', color=color,
                label=f'{backend} (next_fast_len)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Transform length [samples]')
    ax.set_ylabel('Time per FFT call [ms]')
    ax.set_title('FFT Time: Raw Length vs. next_fast_len Padding')
    ax.legend(fontsize=7)
    ax.grid(True, which='both', ls='-', alpha=0.2)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, 'study_padding_speed.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {os.path.abspath(fig_path)}")
    plt.close(fig)

    best_speedup = max((s for b in per_backend.values() for s in b['speedup']), default=float('nan'))
    print(f"\nLargest measured speedup from padding to next_fast_len: {best_speedup:.2f}x")

    return {'sizes': sizes, 'per_backend': per_backend, 'max_speedup': best_speedup}


# -----------------------------------------------------------------------
# 2b. Padding does not add resolution
# -----------------------------------------------------------------------

def section_resolution(args, out_dir):
    print(f"\n{'=' * 70}\n2b. PADDING DOES NOT ADD RESOLUTION\n{'=' * 70}")

    fs = 100.0
    N = 100 if args.quick else 200
    T = N / fs
    rayleigh_limit = 1.0 / T

    f1 = 10.0
    separation = 0.6 * rayleigh_limit  # deliberately inside the Rayleigh limit
    f2 = f1 + separation
    true_center = 0.5 * (f1 + f2)  # continuous-time peak of two equal-amplitude tones

    print(f"N={N}, fs={fs} Hz, T={T:.3f} s, Rayleigh limit 1/T = {rayleigh_limit:.4f} Hz")
    print(f"Tones at f1={f1:.4f} Hz, f2={f2:.4f} Hz, separation={separation:.4f} Hz "
          f"({'below' if separation < rayleigh_limit else 'above'} the Rayleigh limit)")

    t = np.arange(N) / fs
    x = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)

    pad_factors = [1, 2, 4, 8, 16] if not args.quick else [1, 2, 4]
    rows = []
    fig, ax = plt.subplots(figsize=(10, 7))
    center_errors = []
    n_peaks_found = []

    for factor in pad_factors:
        n_pad = N * factor
        X = fftkit.rfft(x, n=n_pad)
        freqs = np.fft.rfftfreq(n_pad, d=1.0 / fs)
        mag = np.abs(X)

        window_mask = (freqs > f1 - 3 * rayleigh_limit) & (freqs < f2 + 3 * rayleigh_limit)
        # Peaks within +/- half the tone separation of the two tones, i.e.
        # in the region where two genuinely resolved peaks would appear.
        peak_idx, _ = scipy_find_peaks(mag, prominence=0.05 * mag.max())
        peak_idx = peak_idx[(freqs[peak_idx] > f1 - separation) & (freqs[peak_idx] < f2 + separation)]
        n_found = len(peak_idx)

        if n_found > 0:
            # Amplitude-weighted centroid of the detected peak(s): the best
            # available *location estimate* from this spectrum.
            detected_center = np.average(freqs[peak_idx], weights=mag[peak_idx])
        else:
            detected_center = float('nan')
        center_error = abs(detected_center - true_center)

        rows.append([factor, n_pad, n_found, detected_center, center_error])
        center_errors.append(center_error)
        n_peaks_found.append(n_found)

        ax.plot(freqs[window_mask], mag[window_mask], marker='.', markersize=3,
                label=f'{factor}x padding ({n_found} peak{"s" if n_found != 1 else ""})')

    ax.axvline(f1, color='k', linestyle=':', alpha=0.5, label='true tones')
    ax.axvline(f2, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('|X(f)|')
    ax.set_title(f'Two tones separated by {separation:.3f} Hz '
                 f'(Rayleigh limit {rayleigh_limit:.3f} Hz): padding does not split them')
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, 'study_padding_resolution.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {os.path.abspath(fig_path)}")
    plt.close(fig)

    print(tabulate(rows, headers=['Pad factor', 'N_padded', 'Peaks detected',
                                   'Detected center [Hz]', 'Error vs true center [Hz]'],
                    floatfmt=('', '', '', '.4f', '.5f')))

    all_unresolved = all(n <= 1 for n in n_peaks_found)
    print(f"\nUnresolved (<=1 peak in the tone region) at every padding factor tested: {all_unresolved}")
    if center_errors[0] > 0 and not np.isnan(center_errors[-1]):
        print(f"Peak-location error: {center_errors[0]:.5f} Hz (no padding) "
              f"-> {center_errors[-1]:.5f} Hz ({pad_factors[-1]}x padding)")

    return {
        'fs': fs, 'N': N, 'rayleigh_limit_hz': rayleigh_limit,
        'f1': f1, 'f2': f2, 'separation_hz': separation,
        'pad_factors': pad_factors, 'peaks_detected': n_peaks_found,
        'center_error_hz': center_errors, 'all_unresolved': bool(all_unresolved),
    }


# -----------------------------------------------------------------------
# 2c. Padding and windowing do not commute
# -----------------------------------------------------------------------

def _sidelobe_level_db(freqs, mag, main_lobe_idx, exclude_bins):
    """Peak dB level of the highest sidelobe outside the main lobe."""
    mask = np.ones(len(mag), dtype=bool)
    lo = max(0, main_lobe_idx - exclude_bins)
    hi = min(len(mag), main_lobe_idx + exclude_bins + 1)
    mask[lo:hi] = False
    if not np.any(mask):
        return float('-inf')
    sidelobe_peak = mag[mask].max()
    return 20 * np.log10(sidelobe_peak / mag[main_lobe_idx])


def section_windowing_order(args, out_dir):
    print(f"\n{'=' * 70}\n2c. PADDING AND WINDOWING DO NOT COMMUTE\n{'=' * 70}")

    fs = 1000.0
    N = 256 if args.quick else 1024
    f0 = 50.3  # off-bin on purpose, so leakage/sidelobes are visible
    pad_factor = 4
    n_pad = N * pad_factor

    t = np.arange(N) / fs
    x = np.sin(2 * np.pi * f0 * t)
    window = get_window('hann', N)

    # Correct: window the real data first, then append zeros. The window
    # already tapers the real data's edges to ~0, so the zeros that follow
    # introduce no new discontinuity.
    x_correct = np.concatenate([x * window, np.zeros(n_pad - N)])

    # Wrong: pad first, then apply a window stretched over the full padded
    # length. The window's taper region slides into the zero tail, so the
    # real data's edges are left comparatively untapered against the window
    # body -- reintroducing the discontinuity the window was meant to fix.
    window_full = get_window('hann', n_pad)
    x_wrong = np.concatenate([x, np.zeros(n_pad - N)]) * window_full

    freqs = np.fft.rfftfreq(n_pad, d=1.0 / fs)
    mag_correct = np.abs(fftkit.rfft(x_correct))
    mag_wrong = np.abs(fftkit.rfft(x_wrong))

    peak_idx_correct = int(np.argmax(mag_correct))
    peak_idx_wrong = int(np.argmax(mag_wrong))
    peak_amp_correct = mag_correct[peak_idx_correct]
    peak_amp_wrong = mag_wrong[peak_idx_wrong]

    exclude_bins = max(1, int(round(pad_factor * 3)))
    sidelobe_correct_db = _sidelobe_level_db(freqs, mag_correct, peak_idx_correct, exclude_bins)
    sidelobe_wrong_db = _sidelobe_level_db(freqs, mag_wrong, peak_idx_wrong, exclude_bins)

    peak_amp_error_pct = 100 * abs(peak_amp_wrong - peak_amp_correct) / peak_amp_correct

    print(f"N={N}, pad_factor={pad_factor}, window=hann, tone f0={f0} Hz (fs={fs} Hz)")
    print(tabulate([
        ['window -> pad (correct)', peak_amp_correct, sidelobe_correct_db],
        ['pad -> window (wrong)', peak_amp_wrong, sidelobe_wrong_db],
    ], headers=['Order', 'Peak amplitude', 'Highest sidelobe [dB below peak]'], floatfmt='.4f'))
    print(f"\nPeak amplitude difference: {peak_amp_error_pct:.2f}%")
    print(f"Sidelobe level difference: {sidelobe_wrong_db - sidelobe_correct_db:+.2f} dB "
          f"({'worse' if sidelobe_wrong_db > sidelobe_correct_db else 'better'} for pad-then-window)")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(freqs, 20 * np.log10(mag_correct / peak_amp_correct), label='window then pad (correct)')
    ax.plot(freqs, 20 * np.log10(mag_wrong / peak_amp_correct), label='pad then window (wrong)', alpha=0.8)
    ax.set_xlim(0, 4 * f0)
    ax.set_ylim(-100, 5)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Magnitude relative to correct-order peak [dB]')
    ax.set_title('Window/Pad Order: Effect on Sidelobe Structure')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', ls='-', alpha=0.2)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, 'study_padding_window_order.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {os.path.abspath(fig_path)}")
    plt.close(fig)

    return {
        'fs': fs, 'N': N, 'f0': f0, 'pad_factor': pad_factor,
        'peak_amplitude_correct': float(peak_amp_correct),
        'peak_amplitude_wrong': float(peak_amp_wrong),
        'peak_amplitude_error_pct': float(peak_amp_error_pct),
        'sidelobe_db_correct': float(sidelobe_correct_db),
        'sidelobe_db_wrong': float(sidelobe_wrong_db),
    }


# -----------------------------------------------------------------------
# 2d. Padding changes power normalization
# -----------------------------------------------------------------------

def section_normalization(args, out_dir):
    print(f"\n{'=' * 70}\n2d. PADDING CHANGES POWER NORMALIZATION\n{'=' * 70}")

    fs = 500.0
    N = 500 if args.quick else 2000
    rng = np.random.default_rng(7)
    t = np.arange(N) / fs
    x = np.sin(2 * np.pi * 30 * t) + 0.3 * rng.standard_normal(N)
    var_x = np.var(x)

    freqs0, psd0 = periodogram_rfft(x, fs)
    df0 = freqs0[1] - freqs0[0]
    ratio_unpadded = np.sum(psd0) * df0 / var_x
    print(f"Unpadded sanity check: sum(psd)*df / var(x) = {ratio_unpadded:.10f} (expected 1.0)")

    pad_factors = [1, 2, 4, 8] if not args.quick else [1, 2, 4]
    rows = []
    ratio_self_list, ratio_vs_original_list = [], []

    for factor in pad_factors:
        n_pad = N * factor
        x_padded = np.concatenate([x, np.zeros(n_pad - N)])
        freqs, psd = periodogram_rfft(x_padded, fs)
        df = freqs[1] - freqs[0]
        total_power = np.sum(psd) * df

        ratio_self = total_power / np.var(x_padded)
        ratio_vs_original = total_power / var_x

        rows.append([factor, n_pad, np.var(x_padded), ratio_self, ratio_vs_original])
        ratio_self_list.append(float(ratio_self))
        ratio_vs_original_list.append(float(ratio_vs_original))

    print(tabulate(rows, headers=['Pad factor', 'N_padded', 'var(x_padded)',
                                   'sum(psd)*df / var(x_padded)',
                                   'sum(psd)*df / var(x_original)'],
                    floatfmt=('', '', '.6f', '.10f', '.6f')))

    print("\nRule: the Parseval identity sum(psd)*df == var(x) holds exactly for whatever "
          "array is actually transformed, padded or not (ratio_self stays 1.0 above). But "
          "appending zeros dilutes that array's own variance, so a PSD computed on a "
          "zero-padded signal understates the ORIGINAL signal's total power by roughly the "
          "padding ratio -- compare against var() of the padded array you fed in, not the "
          "pre-padding signal, or rescale by N_padded/N_original.")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(pad_factors, ratio_self_list, marker='o', label='sum(psd)*df / var(padded array)')
    ax.plot(pad_factors, ratio_vs_original_list, marker='s',
            label='sum(psd)*df / var(original signal)')
    ax.axhline(1.0, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('Zero-padding factor (N_padded / N_original)')
    ax.set_ylabel('Power ratio')
    ax.set_title('Effect of Zero-Padding on PSD Power Normalization')
    ax.legend(fontsize=8)
    ax.grid(True, ls='-', alpha=0.2)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, 'study_padding_normalization.png')
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {os.path.abspath(fig_path)}")
    plt.close(fig)

    return {
        'fs': fs, 'N': N, 'ratio_unpadded_sanity_check': float(ratio_unpadded),
        'pad_factors': pad_factors,
        'ratio_self': ratio_self_list,
        'ratio_vs_original': ratio_vs_original_list,
    }


def main():
    parser = argparse.ArgumentParser(description='Study zero-padding effects on FFT-based analysis.')
    parser.add_argument('--out', type=str, default='benchmarks/out', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Quick mode')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    results = {
        'speed': section_speed(args, args.out),
        'resolution': section_resolution(args, args.out),
        'windowing_order': section_windowing_order(args, args.out),
        'normalization': section_normalization(args, args.out),
    }

    json_path = os.path.join(args.out, 'study_padding.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {os.path.abspath(json_path)}")


if __name__ == "__main__":
    main()
