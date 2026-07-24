"""
FFT Performance Benchmarking Suite for fftkit.

WHAT THIS SCRIPT MEASURES:
1. Single FFT Performance across backends and signal sizes
2. Batch FFT Performance (GPU vs CPU comparison)
"""

import argparse
import json
import os

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from fftkit import get_available_backends, get_fft_func, gpu_available


def generate_signal(size):
    """Generate a signal with multiple frequency components."""
    np.random.seed(42)
    t = np.linspace(0, 1, size)
    signal_clean = (
        0.5 * np.sin(2 * np.pi * 10 * t) +
        0.3 * np.sin(2 * np.pi * 25 * t) +
        0.2 * np.sin(2 * np.pi * 50 * t)
    )
    noise_level = 0.05
    white_noise = np.random.normal(0, noise_level, size)
    colored_noise = signal.lfilter([1.0], [1.0, -0.9], white_noise)
    impulse_locations = np.random.choice(size, size=int(size * 0.01), replace=False)
    impulse_noise = np.zeros(size)
    impulse_noise[impulse_locations] = np.random.normal(0, noise_level * 5, size=len(impulse_locations))
    noisy_signal = signal_clean + colored_noise + impulse_noise
    return noisy_signal.astype(np.float32)


def compare_fft(size, N_times=3, discard=1):
    """Benchmark all available backends for a signal size."""
    import timeit
    sig = generate_signal(size)
    backends = get_available_backends()
    result_dict = {}
    timings = {}
    results = {}

    for backend in backends:
        fft_func = get_fft_func(backend)
        try:
            times = []
            total_runs = N_times + discard
            for i in range(total_runs):
                def wrapper():
                    return fft_func(sig)
                t = timeit.timeit(wrapper, number=10)
                if i >= discard:
                    times.append(t)
            avg_time = float(np.mean(times)) if times else 0
            res = np.abs(wrapper())
            timings[backend] = avg_time
            results[backend] = res
        except Exception:
            timings[backend] = None
            results[backend] = None

    valid = {b: timings[b] for b in backends if timings[b] is not None}
    if not valid:
        for b in backends:
            result_dict[f"{b}_fft_time"] = 0
            result_dict[f"{b}_error"] = 0
        return result_dict

    ref_backend = min(valid, key=valid.get)
    ref_result = results[ref_backend]
    for b in backends:
        result_dict[f"{b}_fft_time"] = timings[b] if timings[b] is not None else 0
        if results[b] is not None:
            result_dict[f"{b}_error"] = float(np.mean(np.abs(ref_result[:len(results[b])] - results[b][:len(ref_result)])))
        else:
            result_dict[f"{b}_error"] = 0
    result_dict['reference_backend'] = ref_backend
    return result_dict


def main():
    parser = argparse.ArgumentParser(description='Benchmark FFT backends.')
    parser.add_argument('--out', type=str, default='benchmarks/out', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Quick mode (fewer sizes/iterations)')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Available backends: {get_available_backends()}")

    # Define sizes based on quick mode
    if args.quick:
        sizes_pow2 = [2**10, 2**14, 2**16]
        sizes_off = [1001, 10001, 65537]
        sizes = sorted(set(sizes_pow2 + sizes_off))
        N_times = 1
    else:
        powers = list(range(10, 19))
        sizes_pow2 = [2 ** p for p in powers]
        sizes_off = []
        for n in sizes_pow2:
            for delta in [-3, -1, +1, +3]:
                off_val = n + delta
                if 0 < off_val <= 262144 and (off_val & (off_val - 1)) != 0:
                    sizes_off.append(off_val)
        real_world_sizes = [80001, 100000, 150000, 200000]
        sizes_off.extend(real_world_sizes)
        sizes = sorted(set(sizes_pow2 + sizes_off))
        N_times = 3

    print(f"Testing {len(sizes)} sizes: {sizes}")

    backends = get_available_backends()
    results = {}
    for backend in backends:
        results[f"{backend}_fft_time"] = []
        results[f"{backend}_error"] = []
    results['reference_backend'] = []

    for size in sizes:
        print(f"Processing size: {size}")
        result = compare_fft(size, N_times=N_times)
        for key in result:
            if key in results:
                results[key].append(result[key])
        results['reference_backend'].append(result.get('reference_backend', ''))

    # Save JSON
    results_json = {k: [float(x) if isinstance(x, (int, float)) else x for x in v] for k, v in results.items()}
    results_json['sizes'] = [float(x) for x in sizes]
    json_path = os.path.join(args.out, 'bench_backends.json')
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=4)
    print(f"Results saved to {os.path.abspath(json_path)}")

    # Plot results
    pow2_idx = [i for i, s in enumerate([int(s) for s in sizes]) if (s & (s-1)) == 0]
    non_pow2_idx = [i for i, s in enumerate([int(s) for s in sizes]) if (s & (s-1)) != 0]

    pow2_sizes = np.array(sizes)[pow2_idx]
    non_pow2_sizes = np.array(sizes)[non_pow2_idx]

    # Plot power-of-2
    if len(pow2_idx) > 0:
        plt.figure(figsize=(12, 8))
        for backend in backends:
            times = np.array(results[f"{backend}_fft_time"])
            cat_times = times[pow2_idx]
            plt.plot(pow2_sizes, cat_times, marker='o', label=f"{backend} FFT")
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Signal Size (samples)')
        plt.ylabel('Time (seconds) for 10 iterations')
        plt.title('FFT Performance (Power-of-2 Sizes)')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend()
        pow2_path = os.path.join(args.out, 'bench_backends_pow2.png')
        plt.savefig(pow2_path, dpi=300, bbox_inches='tight')
        print(f"Saved power-of-2 plot to {os.path.abspath(pow2_path)}")
        plt.close()

    # Plot non-power-of-2
    if len(non_pow2_idx) > 0:
        plt.figure(figsize=(12, 8))
        for backend in backends:
            times = np.array(results[f"{backend}_fft_time"])
            cat_times = times[non_pow2_idx]
            plt.plot(non_pow2_sizes, cat_times, marker='o', label=f"{backend} FFT")
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Signal Size (samples)')
        plt.ylabel('Time (seconds) for 10 iterations')
        plt.title('FFT Performance (Non-Power-of-2 Sizes)')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend()
        non_pow2_path = os.path.join(args.out, 'bench_backends_non_pow2.png')
        plt.savefig(non_pow2_path, dpi=300, bbox_inches='tight')
        print(f"Saved non-power-of-2 plot to {os.path.abspath(non_pow2_path)}")
        plt.close()

    # Batch/GPU comparison
    print("\n" + "="*60)
    print("BATCH FFT BENCHMARK (CPU vs GPU)")
    print("="*60)

    if gpu_available():
        print("GPU backend available, running batch benchmark...")
        try:
            from fftkit import benchmark_cpu_vs_gpu
            batch_results = benchmark_cpu_vs_gpu(sizes=[4096, 16384, 65536] if not args.quick else [4096, 16384],
                                                 batch_sizes=[1, 16, 64] if not args.quick else [1, 16],
                                                 iterations=10 if not args.quick else 3)
            batch_json_path = os.path.join(args.out, 'bench_backends_batch.json')
            with open(batch_json_path, 'w') as f:
                json.dump(batch_results, f, indent=4)
            print(f"Batch results saved to {os.path.abspath(batch_json_path)}")
        except Exception as e:
            print(f"GPU batch benchmark failed: {e}")
    else:
        print("GPU backend NOT available - skipping GPU batch benchmark")


if __name__ == "__main__":
    main()
