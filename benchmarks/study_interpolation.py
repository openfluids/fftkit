"""
Interpolation methods for resampling a time series onto a coarser uniform grid.

Motivation: simulations often write output at variable time intervals, while the
FFT requires constant ones, so the data must be interpolated first. Which
interpolation method distorts the spectrum least?

WHAT THIS ACTUALLY MEASURES, AND WHAT IT DOES NOT

This compares methods on a DECIMATION problem: a fine UNIFORM grid
(dt = 4.3e-4) resampled onto a coarse UNIFORM grid (dt = 0.05), scoring each
method by the mean squared error between the resulting spectrum and the
spectrum of the original series.

It does NOT yet test genuinely non-uniform input. `variable_time_steps()` below
generates such a grid but is not wired into `main()` -- the same gap existed in
the script this was ported from, where the call was commented out. Until that is
connected, treat these rankings as evidence about downsampling, not about the
non-uniform-dt case that motivates the study.

Two further caveats when reading the output:
- The score is a single realization per noise level. Methods that finish close
  together swap order between seeds; only large gaps are meaningful.
- MSE is computed across the whole spectrum, so the broadband noise floor
  contributes alongside the signal peaks.
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
from tabulate import tabulate

from fftkit import generate_complex_signal, periodogram_rfft


def variable_time_steps(T, base_dt, variability=0.1):
    """Generate non-uniform time steps by jittering a nominal step.

    Currently unused by main(); see the module docstring. Kept because it is the
    piece needed to extend this study to genuinely non-uniform input.
    """
    times = [0]
    while times[-1] < T:
        next_time = times[-1] + abs(np.random.normal(base_dt, variability * base_dt))
        if next_time < T:
            times.append(next_time)
        else:
            break
    return np.array(times)


def compare_interpolations_and_ffts(time_original, data_original, time_new):
    """Compare interpolation methods and their FFT results."""
    dt_orig = time_original[1] - time_original[0]
    freq_orig, fft_orig = periodogram_rfft(data_original, 1.0/dt_orig)

    methods = {
        'Linear': interp1d(time_original, data_original, kind='linear'),
        'Slinear': interp1d(time_original, data_original, kind='slinear'),
        'Zero': interp1d(time_original, data_original, kind='zero'),
        'Nearest': interp1d(time_original, data_original, kind='nearest'),
        'Cubic Spline': CubicSpline(time_original, data_original),
        'Quintic Spline': lambda x: splev(x, splrep(time_original, data_original, k=5)),
        'Akima': Akima1DInterpolator(time_original, data_original, method='akima'),
        'Makima': Akima1DInterpolator(time_original, data_original, method='makima'),
    }

    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    fig.suptitle('Comparison of Interpolation Methods and Their FFTs', fontsize=16)

    mse_scores = {}

    # Plot interpolated signals
    axs[0].plot(time_original, data_original, 'ko-', label='Original', markersize=3)
    for name, interpolator in methods.items():
        data_interp = interpolator(time_new) if callable(interpolator) else interpolator(time_new)
        axs[0].plot(time_new, data_interp, label=name)

        # Compute FFT and MSE
        dt_new = time_new[1] - time_new[0]
        freq_new, fft_new = periodogram_rfft(data_interp, 1.0/dt_new)
        mse = np.mean((np.interp(freq_new, freq_orig, fft_orig) - fft_new) ** 2)
        mse_scores[name] = mse

        axs[1].semilogy(freq_new, fft_new, label=f'{name} (MSE: {mse:.2e})')

    axs[0].set_title('Interpolated Signals')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Amplitude')
    axs[0].legend()
    axs[0].set_xlim(1, 2)

    axs[1].semilogy(freq_orig, fft_orig, 'k--', label='Original')
    axs[1].set_title('FFT of Interpolated Signals')
    axs[1].set_xlabel('Frequency')
    axs[1].set_ylabel('Power Spectral Density')
    axs[1].legend()
    axs[1].set_xlim(1e-2, 1)
    axs[1].set_ylim(1e-6, 1e2)

    # Plot frequency-resolved absolute error
    axs[2].set_title('Frequency-Resolved Absolute Error')
    axs[2].set_xlabel('Frequency')
    axs[2].set_ylabel('Absolute Error')
    for name, interpolator in methods.items():
        data_interp = interpolator(time_new) if callable(interpolator) else interpolator(time_new)
        dt_new = time_new[1] - time_new[0]
        freq_new, fft_new = periodogram_rfft(data_interp, 1.0/dt_new)
        error = np.abs(np.interp(freq_new, freq_orig, fft_orig) - fft_new)
        axs[2].semilogy(freq_new, error, label=name)
    axs[2].legend()

    # Sort and print
    sorted_methods = sorted(mse_scores.items(), key=lambda x: x[1])
    print("Interpolation methods ranked from best to worst based on MSE:")
    print(tabulate(sorted_methods, headers=['Method', 'MSE'], tablefmt='orgtbl'))
    best_method = sorted_methods[0][0]
    print(f"Best method: {best_method}\n")

    # Highlight best in plot
    dt_new = time_new[1] - time_new[0]
    best_data_interp = methods[best_method](time_new) if callable(methods[best_method]) else methods[best_method](time_new)
    freq_best, fft_best = periodogram_rfft(best_data_interp, 1.0/dt_new)
    axs[1].semilogy(freq_best, fft_best, label=f'{best_method} (best)', linewidth=3, color='red')
    axs[1].annotate(f'Best: {best_method}', xy=(0.05, 0.9), xycoords='axes fraction')

    plt.tight_layout()
    return sorted_methods, fig


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

    # Parameters
    L = 1.0  # noqa: F841
    U = 1.0  # noqa: F841
    St1 = 0.1212131
    St2 = 0.0874888
    num_harmonics_f1 = 8 if not args.quick else 2
    num_harmonics_f2 = 6 if not args.quick else 2

    periods = 20 if not args.quick else 2
    T = periods / St2
    dt_orig = 0.00043231321123124
    t_orig = np.arange(0, T, dt_orig)

    dt_new = 0.05
    t_new = np.arange(0, T, dt_new)

    noise_levels = [0.01, 0.04, 0.07, 0.1] if not args.quick else [0.05]
    mse_results = {}

    for i, noise_level in enumerate(noise_levels):
        # Vary the seed per noise level so the levels are independent
        # realizations rather than the same noise pattern rescaled.
        x_orig = generate_complex_signal(t_orig, St1, St2,
                                          num_harmonics_f1=num_harmonics_f1,
                                          num_harmonics_f2=num_harmonics_f2,
                                          noise_level=noise_level,
                                          seed=None if seed is None else seed + i)
        mse_list, fig = compare_interpolations_and_ffts(t_orig, x_orig, t_new)
        mse_results[noise_level] = {method: mse for method, mse in mse_list}

        fig_path = os.path.join(args.out, f'study_interpolation_noise_{noise_level}.png')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {os.path.abspath(fig_path)}")
        plt.close(fig)

    # MSE vs noise
    fig, ax = plt.subplots(figsize=(8, 6))
    methods = list(mse_results[noise_levels[0]].keys())
    for method in methods:
        mse_values = [mse_results[noise_level][method] for noise_level in noise_levels]
        ax.plot(noise_levels, mse_values, marker='o', label=method)
    ax.set_title('MSE vs. Noise for Each Method')
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('MSE')
    ax.legend()
    plt.tight_layout()
    mse_path = os.path.join(args.out, 'study_interpolation_mse_vs_noise.png')
    ax.figure.savefig(mse_path, dpi=300, bbox_inches='tight')
    print(f"Saved MSE plot to {os.path.abspath(mse_path)}")
    plt.close(fig)

    # Summary statistics
    best_methods = [min(mse_results[n].items(), key=lambda x: x[1])[0] for n in noise_levels]
    method_counts = Counter(best_methods)
    most_robust = method_counts.most_common(1)[0][0]
    print(f'\nMost robust method across all noise levels: {most_robust}')
    for method, count in method_counts.items():
        print(f'{method}: Best at {count} out of {len(noise_levels)} noise levels')

    # Percent difference table
    print("\nPercent difference in MSE from best method at each noise level:")
    perc_diff_table = []
    for n in noise_levels:
        best_mse = min(mse_results[n].values())
        row = {'Noise Level': n}
        for method in methods:
            perc_diff = 100 * (mse_results[n][method] - best_mse) / best_mse if best_mse > 0 else 0.0
            row[method] = perc_diff
        perc_diff_table.append(row)

    print(tabulate([[row['Noise Level']] + [row[m] for m in methods] for row in perc_diff_table],
                   headers=['Noise Level'] + methods, floatfmt=".2f"))

    # Save JSON
    json_results = {
        'noise_levels': noise_levels,
        'mse_results': mse_results,
        'percent_difference': perc_diff_table,
        'best_method_per_noise': best_methods,
        'method_counts': dict(method_counts),
        'most_robust': most_robust
    }
    json_path = os.path.join(args.out, 'study_interpolation.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Results saved to {os.path.abspath(json_path)}")


if __name__ == "__main__":
    main()
