"""
Interpolation methods for resampling a time series onto a uniform grid.

Motivation: simulations often write output at variable time intervals, while
the FFT requires constant ones, so the data must be interpolated first. Which
interpolation method distorts the spectrum least?

This script covers TWO structurally different resampling problems and reports
them separately -- they are not comparable to each other and must not be
merged into one ranking:

1. NON-UNIFORM -> UNIFORM (the case that motivates the study). A jittered
   source grid (`variable_time_steps`) is interpolated onto a uniform target
   grid at roughly the same average rate, so there is no decimation and
   (mostly) no aliasing -- this isolates how much error the interpolant
   itself introduces when regularising variable-dt simulation output. The
   ground truth here is the exact analytic signal (noise_level=0) evaluated
   directly on the target grid: since the test signal is a known sum of
   sinusoids, its value at any time is known exactly, with no interpolation
   needed to obtain it.

2. UNIFORM DECIMATION (what earlier versions of this script silently did
   under the "non-uniform" heading). A fine uniform grid is downsampled onto
   a much coarser uniform grid (116x here) with no anti-alias filter in the
   naive interpolants, so content above the new Nyquist folds back into the
   band. Measured on a noiseless two-tone signal, this aliasing error swamps
   interpolation-method differences: nearest / linear / cubic land within
   0.1% of each other (five significant figures identical). Comparing naive
   interpolants against the *unfiltered* high-resolution spectrum was
   therefore ranking aliasing noise, not interpolation quality.

   Fix applied here: the ground-truth reference for this case is
   `scipy.signal.decimate` (which low-pass filters before downsampling), not
   the raw high-resolution spectrum. `scipy.signal.decimate` is also included
   as a candidate "method" in the table. If it beats every naive interpolant
   by a wide margin -- which is the expected and, on this signal, observed
   result -- that is this study's most useful finding: for heavy decimation,
   whether you anti-alias filter matters far more than which interpolant you
   pick afterwards.

Two metrics are reported for both cases:

- Whole-spectrum MSE against the case's ground truth (defined above; for case
  2 this is restricted to the decimated Nyquist band by construction, since
  the decimated PSD only has content up to that frequency).
- Peak-frequency error: `fftkit.calculate_error(..., symmetric=True)` between
  each method's detected peaks and the exactly-known planted tone/harmonic
  frequencies (restricted to those below the new Nyquist for case 2). This is
  closer to what a user usually cares about than whole-spectrum MSE, which is
  dominated by the broadband noise floor.

Caveats when reading the output, all measured rather than assumed:

- The score is a single realization per noise level. Methods that finish
  close together swap order between seeds; only large gaps mean anything.
- `interp1d(kind='linear')` and `kind='slinear'` are the same interpolant
  (`slinear` IS a first-order spline): measured max|difference| = 2.220e-16,
  exactly float64 eps. Only `Linear` is kept; ranking it against `Slinear`
  would be ranking rounding noise. `Zero` and `Nearest` are NOT duplicates of
  each other (measured max|difference| = 0.379) and both stay.
"""

import argparse
import json
import os
from collections import Counter

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import Akima1DInterpolator, CubicSpline, interp1d, splev, splrep
from scipy.signal import decimate as scipy_decimate
from tabulate import tabulate

from fftkit import calculate_error, find_peaks, generate_complex_signal, periodogram_rfft


def variable_time_steps(T, base_dt, variability=0.1):
    """Generate non-uniform time steps by jittering a nominal step."""
    times = [0]
    while times[-1] < T:
        next_time = times[-1] + abs(np.random.normal(base_dt, variability * base_dt))
        if next_time < T:
            times.append(next_time)
        else:
            break
    return np.array(times)


def true_peak_frequencies(f1, f2, num_harmonics_f1, num_harmonics_f2):
    """Exactly-known frequencies planted by generate_complex_signal."""
    peaks = [f1 * i for i in range(1, num_harmonics_f1 + 1)]
    peaks += [f2 * i for i in range(1, num_harmonics_f2 + 1)]
    return np.array(sorted(peaks))


def build_methods(time_source, data_source):
    """Interpolators sharing one (time, data) source series.

    Slinear is deliberately excluded: interp1d(kind='linear') and
    interp1d(kind='slinear') are the same interpolant (slinear IS a
    first-order spline); see module docstring for the measured difference.
    """
    return {
        'Linear': interp1d(time_source, data_source, kind='linear'),
        'Zero': interp1d(time_source, data_source, kind='zero'),
        'Nearest': interp1d(time_source, data_source, kind='nearest'),
        'Cubic Spline': CubicSpline(time_source, data_source),
        'Quintic Spline': lambda x: splev(x, splrep(time_source, data_source, k=5)),
        'Akima': Akima1DInterpolator(time_source, data_source, method='akima'),
        'Makima': Akima1DInterpolator(time_source, data_source, method='makima'),
    }


def _clip_to_source_range(t_new, t_start, t_end):
    """Keep target times strictly inside the source range.

    interp1d raises on out-of-range targets, and np.arange with an
    irrational float step can overshoot t_end by rounding error, so this
    cannot be left to luck.
    """
    return t_new[(t_new >= t_start) & (t_new <= t_end)]


def _plot_case(fig_title, time_source, data_source, time_new, freq_ref, psd_ref,
               methods_output, best_method):
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    fig.suptitle(fig_title, fontsize=16)

    axs[0].plot(time_source, data_source, 'ko-', label='Source', markersize=3)
    for name, (data_interp, _freq, _psd, _mse, _perr) in methods_output.items():
        t_plot = time_new[:len(data_interp)]
        axs[0].plot(t_plot, data_interp, label=name)
    axs[0].set_title('Interpolated Signals')
    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('Amplitude')
    axs[0].legend(fontsize=7)
    span = time_new[-1] - time_new[0]
    axs[0].set_xlim(time_new[0], time_new[0] + min(span, 20 * (time_new[1] - time_new[0])))

    axs[1].semilogy(freq_ref, psd_ref, 'k--', label='Reference')
    for name, (_data, freq_m, psd_m, mse, _perr) in methods_output.items():
        axs[1].semilogy(freq_m, psd_m, label=f'{name} (MSE: {mse:.2e})')
    axs[1].set_title('PSD of Interpolated Signals vs. Reference')
    axs[1].set_xlabel('Frequency [Hz]')
    axs[1].set_ylabel('Power Spectral Density')
    axs[1].legend(fontsize=7)

    axs[2].set_title('Frequency-Resolved Absolute Error vs. Reference')
    axs[2].set_xlabel('Frequency [Hz]')
    axs[2].set_ylabel('Absolute Error')
    for name, (_data, freq_m, psd_m, _mse, _perr) in methods_output.items():
        ref_interp = np.interp(freq_m, freq_ref, psd_ref)
        axs[2].semilogy(freq_m, np.abs(ref_interp - psd_m), label=name)
    axs[2].legend(fontsize=7)

    if best_method in methods_output:
        _data, freq_best, psd_best, _mse, _perr = methods_output[best_method]
        axs[1].semilogy(freq_best, psd_best, label=f'{best_method} (best)', linewidth=3, color='red')
        axs[1].annotate(f'Best: {best_method}', xy=(0.05, 0.9), xycoords='axes fraction')

    plt.tight_layout()
    return fig


def compare_nonuniform(t_nonuniform, x_nonuniform, dt_new, St1, St2, n1, n2):
    """Case 1: non-uniform source grid -> uniform target grid (no decimation)."""
    t_start, t_end = t_nonuniform[0], t_nonuniform[-1]
    t_new = np.arange(t_start, t_end, dt_new)
    t_new = _clip_to_source_range(t_new, t_start, t_end)
    fs_new = 1.0 / dt_new

    true_peaks = true_peak_frequencies(St1, St2, n1, n2)

    # Ground truth: the exact analytic (noiseless) signal evaluated directly
    # on the target grid. No interpolation is needed to obtain it because the
    # signal's functional form is known.
    clean_ref = generate_complex_signal(t_new, St1, St2, num_harmonics_f1=n1,
                                         num_harmonics_f2=n2, noise_level=0.0)
    freq_ref, psd_ref = periodogram_rfft(clean_ref, fs_new)

    methods = build_methods(t_nonuniform, x_nonuniform)
    methods_output = {}
    for name, interpolator in methods.items():
        data_interp = interpolator(t_new)
        freq_m, psd_m = periodogram_rfft(data_interp, fs_new)
        mse = float(np.mean((psd_ref - psd_m) ** 2))
        peaks_detected, _ = find_peaks(freq_m, psd_m)
        peak_err = float(calculate_error(peaks_detected, true_peaks, symmetric=True))
        methods_output[name] = (data_interp, freq_m, psd_m, mse, peak_err)

    best_method = min(methods_output.items(), key=lambda kv: kv[1][3])[0]
    fig = _plot_case('Non-Uniform Source -> Uniform Target', t_nonuniform, x_nonuniform,
                      t_new, freq_ref, psd_ref, methods_output, best_method)

    mse_scores = {name: out[3] for name, out in methods_output.items()}
    peak_errors = {name: out[4] for name, out in methods_output.items()}
    return mse_scores, peak_errors, fig


def compare_decimation(t_orig, x_orig, dt_orig, dt_new, St1, St2, n1, n2):
    """Case 2: uniform decimation of a fine uniform grid onto a coarse one."""
    t_new = np.arange(0, t_orig[-1], dt_new)
    t_new = _clip_to_source_range(t_new, t_orig[0], t_orig[-1])
    fs_new = 1.0 / dt_new
    nyquist_new = fs_new / 2.0

    true_peaks_all = true_peak_frequencies(St1, St2, n1, n2)
    true_peaks_band = true_peaks_all[true_peaks_all < nyquist_new]

    # Band-limited reference: scipy.signal.decimate low-pass filters before
    # downsampling, so its PSD is the achievable target once you commit to
    # this output rate -- unlike the raw high-resolution spectrum, comparing
    # against it does not conflate unavoidable aliasing error (present for
    # every naive interpolant alike) with genuine interpolation-method error.
    q = max(1, int(round(dt_new / dt_orig)))
    decimated = scipy_decimate(x_orig, q, ftype='fir', zero_phase=True)
    n_common = min(len(decimated), len(t_new))
    decimated = decimated[:n_common]
    t_dec = t_new[:n_common]  # noqa: F841 (kept for clarity of what t_dec represents)
    freq_dec, psd_dec = periodogram_rfft(decimated, fs_new)

    methods = build_methods(t_orig, x_orig)
    methods_output = {}
    for name, interpolator in methods.items():
        data_interp = interpolator(t_new)
        freq_m, psd_m = periodogram_rfft(data_interp, fs_new)
        ref_interp = np.interp(freq_m, freq_dec, psd_dec)
        mse = float(np.mean((ref_interp - psd_m) ** 2))
        peaks_detected, _ = find_peaks(freq_m, psd_m)
        peak_err = float(calculate_error(peaks_detected, true_peaks_band, symmetric=True))
        methods_output[name] = (data_interp, freq_m, psd_m, mse, peak_err)

    # scipy.signal.decimate as a baseline "method". Its MSE against itself is
    # 0 by construction (it IS the reference) -- that is the point: it shows
    # the achievable floor once you anti-alias filter, against which the
    # naive interpolants above are being measured.
    peaks_dec, _ = find_peaks(freq_dec, psd_dec)
    peak_err_dec = float(calculate_error(peaks_dec, true_peaks_band, symmetric=True))
    methods_output['Decimate (scipy)'] = (decimated, freq_dec, psd_dec, 0.0, peak_err_dec)

    best_method = min(methods_output.items(), key=lambda kv: kv[1][3])[0]
    fig = _plot_case(f'Uniform Decimation ({q}x, no source anti-alias filter)', t_orig, x_orig,
                      t_new, freq_dec, psd_dec, methods_output, best_method)

    mse_scores = {name: out[3] for name, out in methods_output.items()}
    peak_errors = {name: out[4] for name, out in methods_output.items()}
    return mse_scores, peak_errors, fig


def _summarize(case_name, mse_results, peak_error_results, noise_levels, methods,
               exclude_from_percent_diff=()):
    """Print ranking tables and return the JSON-serializable summary dict."""
    print(f"\n{'=' * 60}\n{case_name}\n{'=' * 60}")

    best_methods = [min(mse_results[n].items(), key=lambda x: x[1])[0] for n in noise_levels]
    method_counts = Counter(best_methods)
    most_robust = method_counts.most_common(1)[0][0]

    for n in noise_levels:
        sorted_methods = sorted(mse_results[n].items(), key=lambda x: x[1])
        print(f"\nMSE ranking (best to worst), noise_level={n}:")
        print(tabulate(sorted_methods, headers=['Method', 'MSE'], tablefmt='orgtbl'))
        sorted_by_peak = sorted(peak_error_results[n].items(), key=lambda x: x[1])
        print(f"\nPeak-frequency error ranking (symmetric, Hz), noise_level={n}:")
        print(tabulate(sorted_by_peak, headers=['Method', 'Peak error [Hz]'], tablefmt='orgtbl'))

    print(f'\nMost robust method by MSE across all noise levels: {most_robust}')
    for method, count in method_counts.items():
        print(f'{method}: Best at {count} out of {len(noise_levels)} noise levels')

    # Percent-difference is computed only across genuine interpolants: a
    # baseline like "Decimate (scipy)" scores exactly 0 by construction (it
    # IS the reference), so including it as the denominator would make every
    # method's percent-difference come out as 0/0 -> a meaningless "0.00" row
    # rather than the real, much larger gap to the interpolants.
    diff_methods = [m for m in methods if m not in exclude_from_percent_diff]
    if exclude_from_percent_diff:
        print(f"\n(Percent-difference table below excludes baseline method(s) "
              f"{list(exclude_from_percent_diff)}, which score 0 by construction.)")
    print("\nPercent difference in MSE from best method at each noise level:")
    perc_diff_table = []
    for n in noise_levels:
        best_mse = min(mse_results[n][m] for m in diff_methods)
        row = {'Noise Level': n}
        for method in diff_methods:
            perc_diff = 100 * (mse_results[n][method] - best_mse) / best_mse if best_mse > 0 else 0.0
            row[method] = perc_diff
        perc_diff_table.append(row)
    print(tabulate([[row['Noise Level']] + [row[m] for m in diff_methods] for row in perc_diff_table],
                   headers=['Noise Level'] + diff_methods, floatfmt=".2f"))

    return {
        'mse_results': mse_results,
        'peak_error_results': peak_error_results,
        'percent_difference': perc_diff_table,
        'best_method_per_noise': best_methods,
        'method_counts': dict(method_counts),
        'most_robust': most_robust,
    }


def main():
    parser = argparse.ArgumentParser(description='Study interpolation methods.')
    parser.add_argument('--out', type=str, default='benchmarks/out', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Quick mode')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for the noise in the test signal. Rankings between the '
                             'closest-performing methods are otherwise a coin flip from run '
                             'to run. Pass --seed -1 for unseeded noise.')
    args = parser.parse_args()
    seed = None if args.seed < 0 else args.seed

    os.makedirs(args.out, exist_ok=True)

    St1 = 0.1212131
    St2 = 0.0874888
    num_harmonics_f1 = 8 if not args.quick else 2
    num_harmonics_f2 = 6 if not args.quick else 2

    periods = 20 if not args.quick else 2
    T = periods / St2
    noise_levels = [0.01, 0.04, 0.07, 0.1] if not args.quick else [0.05]

    # --- Case 1: non-uniform source -> uniform target ---
    base_dt = 0.01
    dt_new_nonuniform = base_dt

    # --- Case 2: uniform decimation ---
    dt_orig = 0.00043231321123124
    t_orig = np.arange(0, T, dt_orig)
    dt_new_decim = 0.05

    mse_nonuniform, peak_nonuniform = {}, {}
    mse_decim, peak_decim = {}, {}

    for i, noise_level in enumerate(noise_levels):
        run_seed = None if seed is None else seed + i

        # Case 1
        t_nonuniform = variable_time_steps(T, base_dt, variability=0.15)
        x_nonuniform = generate_complex_signal(t_nonuniform, St1, St2,
                                                num_harmonics_f1=num_harmonics_f1,
                                                num_harmonics_f2=num_harmonics_f2,
                                                noise_level=noise_level, seed=run_seed)
        mse1, peak1, fig1 = compare_nonuniform(t_nonuniform, x_nonuniform, dt_new_nonuniform,
                                                St1, St2, num_harmonics_f1, num_harmonics_f2)
        mse_nonuniform[noise_level] = mse1
        peak_nonuniform[noise_level] = peak1
        fig1_path = os.path.join(args.out, f'study_interpolation_nonuniform_noise_{noise_level}.png')
        fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {os.path.abspath(fig1_path)}")
        plt.close(fig1)

        # Case 2
        x_orig = generate_complex_signal(t_orig, St1, St2,
                                          num_harmonics_f1=num_harmonics_f1,
                                          num_harmonics_f2=num_harmonics_f2,
                                          noise_level=noise_level, seed=run_seed)
        mse2, peak2, fig2 = compare_decimation(t_orig, x_orig, dt_orig, dt_new_decim,
                                                St1, St2, num_harmonics_f1, num_harmonics_f2)
        mse_decim[noise_level] = mse2
        peak_decim[noise_level] = peak2
        fig2_path = os.path.join(args.out, f'study_interpolation_decimation_noise_{noise_level}.png')
        fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {os.path.abspath(fig2_path)}")
        plt.close(fig2)

    methods_nonuniform = list(mse_nonuniform[noise_levels[0]].keys())
    methods_decim = list(mse_decim[noise_levels[0]].keys())

    summary_nonuniform = _summarize(
        'CASE 1: non-uniform source -> uniform target (no decimation)',
        mse_nonuniform, peak_nonuniform, noise_levels, methods_nonuniform)
    summary_decim = _summarize(
        'CASE 2: uniform decimation (116x here, no source anti-alias filter)',
        mse_decim, peak_decim, noise_levels, methods_decim,
        exclude_from_percent_diff=('Decimate (scipy)',))

    # MSE vs noise, one plot per case
    for case_name, summary, methods_list, out_name in (
        ('Non-Uniform -> Uniform', summary_nonuniform, methods_nonuniform, 'nonuniform'),
        ('Uniform Decimation', summary_decim, methods_decim, 'decimation'),
    ):
        fig, ax = plt.subplots(figsize=(8, 6))
        for method in methods_list:
            mse_values = [summary['mse_results'][n][method] for n in noise_levels]
            ax.plot(noise_levels, mse_values, marker='o', label=method)
        ax.set_title(f'MSE vs. Noise for Each Method ({case_name})')
        ax.set_xlabel('Noise Level (relative)')
        ax.set_ylabel('MSE')
        ax.legend(fontsize=7)
        plt.tight_layout()
        mse_path = os.path.join(args.out, f'study_interpolation_mse_vs_noise_{out_name}.png')
        fig.savefig(mse_path, dpi=300, bbox_inches='tight')
        print(f"Saved MSE plot to {os.path.abspath(mse_path)}")
        plt.close(fig)

    json_results = {
        'noise_levels': noise_levels,
        'nonuniform': summary_nonuniform,
        'decimation': summary_decim,
    }
    json_path = os.path.join(args.out, 'study_interpolation.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {os.path.abspath(json_path)}")


if __name__ == "__main__":
    main()
