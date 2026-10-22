"""
Comparison of PSD Estimation Methods

Compares periodogram vs Welch vs Blackman-Tukey for spectral estimation accuracy and runtime.
Tests on a signal with known frequencies to measure peak detection error.
"""

import argparse
import json
import os
import time

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from fftkit import (
    blackman_tukey_rfft,
    calculate_error,
    find_peaks,
    generate_complex_signal,
    periodogram_rfft,
    welch_method,
)


def main():
    parser = argparse.ArgumentParser(description='Compare PSD methods.')
    parser.add_argument('--out', type=str, default='benchmarks/out', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Quick mode')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Parameters
    St1 = 0.1212131
    St2 = 0.0874888
    T1 = 1 / St1  # noqa: F841
    T2 = 1 / St2

    periods = 10.231312 if not args.quick else 2
    T = periods * T2
    fs = 1000
    N = int(T * fs)

    # Generate signal
    t = np.arange(0, T, 1/fs)
    x = generate_complex_signal(t, St1, St2)

    # True peaks
    true_peaks = [St2] + [i*St1 for i in range(1, 6)] + [i*St2 for i in range(2, 4)]

    results = {}

    # Test methods
    methods = [
        ("Periodogram", periodogram_rfft),
        ("Blackman-Tukey", blackman_tukey_rfft),
        ("Welch", lambda x, fs: welch_method(x, fs, nperseg=len(x)))
    ]

    print(f"Testing on signal with {N} samples, fs={fs} Hz")
    print(f"True peaks (Strouhal): {[f'{p:.4f}' for p in true_peaks[:3]]}\n")

    for method_name, method_func in methods:
        start_time = time.perf_counter()
        st, psd = method_func(x, fs)
        execution_time = time.perf_counter() - start_time
        peaks_st, _ = find_peaks(st, psd)
        error = calculate_error(peaks_st, true_peaks)
        results[method_name] = {
            "st": st.tolist(),
            "psd": psd.tolist(),
            "peaks": peaks_st.tolist(),
            "time": execution_time,
            "error": error
        }

        print(f"{method_name}:")
        print(f"  Execution time: {execution_time:.4f} seconds")
        print(f"  Error: {error:.6f}")
        print(f"  Detected peaks: {', '.join([f'{peak:.4f}' for peak in peaks_st[:3]])}")
        print()

    # Frequency resolution
    st_resolution = fs / N
    print(f"Frequency (St) resolution: {st_resolution:.6f}\n")

    # Plot spectra
    size_factor = 0.6
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.8*size_factor, 8*size_factor), height_ratios=[1, 3])

    # Plot signal
    ax1.plot(t, x)
    ax1.set_title('Signal with Harmonics')
    ax1.set_xlabel('t')
    ax1.set_ylabel('Amplitude')

    # Plot spectra
    for method, data in results.items():
        ax2.loglog(data['st'], data['psd'], label=f"{method}")

    ax2.set_title('Power Spectral Density Estimates')
    ax2.set_xlabel('Strouhal Number (St)')
    ax2.set_ylabel('PSD')
    ax2.legend()
    ax2.set_xlim(0.8*St2, 2)
    ax2.set_ylim(1e-4, 1e4)
    ax2.grid(True, which="both", ls="-", alpha=0.5)

    # Add vertical lines for true peaks
    for i in range(1, 6):
        ax2.axvline(x=i*St1, color='r', linestyle='--', alpha=0.3)
        ax2.text(i*St1, ax2.get_ylim()[1], f'${i}St_1$', rotation=90, va='top', ha='right', color='r', alpha=0.5)
    for i in range(1, 4):
        ax2.axvline(x=i*St2, color='g', linestyle='--', alpha=0.3)
        ax2.text(i*St2, ax2.get_ylim()[1], f'${i}St_2$', rotation=90, va='top', ha='right', color='g', alpha=0.5)

    plt.tight_layout()
    spec_path = os.path.join(args.out, 'compare_psd_methods_spectral.png')
    fig.savefig(spec_path, dpi=400, bbox_inches='tight')
    print(f"Saved spectral plot to {os.path.abspath(spec_path)}")
    plt.close(fig)

    # Performance table
    method_performance = {}
    for method, data in results.items():
        error_percentage = data['error'] * 100
        time_taken = data['time']
        performance = 1 / max(error_percentage, 1e-6)
        method_performance[method] = {
            'error_percentage': error_percentage,
            'time': time_taken,
            'performance': performance
        }

    # Find best method
    best_method = min(method_performance, key=lambda x: method_performance[x]['error_percentage'])
    print(f"Method with smallest error: {best_method}")
    print(f"Error: {method_performance[best_method]['error_percentage']:.2f}%")
    print(f"Time: {method_performance[best_method]['time']:.4f} seconds")

    # Plot performance
    plt.figure(figsize=(10, 6))
    for method, data in method_performance.items():
        plt.scatter(data['time'], data['performance'], label=method, s=100)

    plt.xlabel('Execution Time (seconds)')
    plt.ylabel('Performance (1 / Error Percentage)')
    plt.title('Performance vs Execution Time')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.5)

    for method, data in method_performance.items():
        plt.annotate(method, (data['time'], data['performance']),
                    xytext=(5, 5), textcoords='offset points')

    plt.tight_layout()
    perf_path = os.path.join(args.out, 'compare_psd_methods_perf.png')
    plt.savefig(perf_path, dpi=400, bbox_inches='tight')
    print(f"Saved performance plot to {os.path.abspath(perf_path)}")
    plt.close()

    # Performance table
    print("\nPerformance Table:")
    print("{:<15} {:<20} {:<20} {:<20}".format("Method", "Error (%)", "Time (s)", "Performance"))
    print("-" * 75)
    for method, data in method_performance.items():
        print("{:<15} {:<20.2f} {:<20.4f} {:<20.4f}".format(
            method,
            data['error_percentage'],
            data['time'],
            data['performance']
        ))

    # Save JSON
    json_results = {
        'methods': list(results.keys()),
        'errors': {m: results[m]['error'] for m in results},
        'execution_times': {m: results[m]['time'] for m in results},
        'best_method': best_method,
        'true_peaks': true_peaks,
        'frequency_resolution': st_resolution,
        'signal_samples': N,
        'sampling_rate': fs
    }
    json_path = os.path.join(args.out, 'compare_psd_methods.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {os.path.abspath(json_path)}")


if __name__ == "__main__":
    main()
