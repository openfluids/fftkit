"""
NOTE: All checks in this script assume the standard unnormalized FFT convention (no scaling by N).
This is the default in numpy, scipy, torch, tensorflow, pyfftw, and most scientific libraries:
    X[k] = sum_{n=0}^{N-1} x[n] * exp(-2j*pi*k*n/N)
No normalization is applied in the forward FFT; if you want a unitary FFT, divide by N (or sqrt(N)) as needed.

# IMPORTANT ON NORMALIZATION:
# Many FFT libraries (including numpy, scipy, torch, tensorflow, pyfftw) use the default 'unnormalized' (a.k.a. 'forward') convention:
#   X[k] = sum_{n=0}^{N-1} x[n] * exp(-2j*pi*k*n/N)
#   x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * exp(2j*pi*k*n/N)
# That is, the forward FFT applies no scaling, and the inverse FFT divides by N.
# Some libraries (e.g., MATLAB's fft, or numpy/scipy with norm='ortho') allow or default to a 'unitary' (energy-preserving) normalization,
# where both the forward and inverse transform are scaled by 1/sqrt(N). This can make Parseval's theorem and other energy relations more symmetrical.
#
# **Mixing normalization conventions can lead to large differences in results!**
# For example, if you switch from an unnormalized FFT (no scaling) to a unitary FFT (scaling by 1/sqrt(N)),
# all your FFT amplitudes and spectral energies will change by factors of N or sqrt(N).
# This is a common source of confusion and bugs when changing FFT engines or porting code between libraries.
#
# Always check the normalization convention used by your FFT function, and apply the appropriate scaling if you need consistent results across engines.
# This script assumes the unnormalized convention for all checks and theoretical calculations.
"""

import argparse
import os

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from fftkit import get_available_backends, get_fft_func

# Configuration
FS = 1024
DURATION = 1.0
FREQ = 50
AMPLITUDE = 1.0
TOL = 1e-6


def generate_sine_wave(freq=FREQ, fs=FS, duration=DURATION, amplitude=AMPLITUDE, phase=0.0):
    t = np.arange(0, duration, 1/fs)
    x = amplitude * np.sin(2 * np.pi * freq * t + phase)
    return t, x


def theoretical_fft_peak(amplitude, N):
    """Theory: unnormalized FFT peak for sine wave amplitude A, length N is A*N/2."""
    return amplitude * N / 2


def test_fft_normalization(x, N, freq=FREQ, amplitude=AMPLITUDE, fs=FS, out_dir=None):
    """Check normalization across available backends."""
    backends = get_available_backends()
    print(f"\nAvailable backends: {backends}")
    print(f"Testing FFT normalization for a {freq} Hz sine wave, {N} samples, amplitude={amplitude}\n")
    results = {}
    for backend in backends:
        try:
            fft_func = get_fft_func(backend)
            X = fft_func(x)
            freqs = np.fft.fftfreq(N, 1/fs)
            idx = np.argmin(np.abs(freqs - freq))
            amp_measured = np.abs(X[idx])
            amp_theory = theoretical_fft_peak(amplitude, N)
            norm_ratio = amp_measured / amp_theory
            is_normalized = np.isclose(norm_ratio, 1.0, rtol=0.05)
            results[backend] = (amp_measured, amp_theory, norm_ratio, is_normalized)
            print(f"Backend: {backend:9s} | FFT peak: {amp_measured:.2f} | Theory: {amp_theory:.2f} | Ratio: {norm_ratio:.2f} | Normalized? {is_normalized}")
        except Exception as e:
            print(f"Backend: {backend:9s} | ERROR: {e}")

    # Plot spectra
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    peak_amplitudes = {}

    ax1 = axes[0]
    colors = plt.cm.tab10.colors
    n_backends = len(backends)
    for i, backend in enumerate(backends):
        try:
            fft_func = get_fft_func(backend)
            X = fft_func(x)
            freqs = np.fft.fftfreq(N, 1/fs)
            amp = np.abs(X[:N//2])
            lw = 6 - (i * 5 / max(n_backends - 1, 1))
            ax1.plot(freqs[:N//2], amp, label=backend, linewidth=lw, color=colors[i % len(colors)])
            idx = np.argmin(np.abs(freqs[:N//2] - freq))
            peak_amplitudes[backend] = amp[idx]
        except Exception:
            pass

    ax1.set_xlabel('Frequency [Hz]')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Full Spectrum (all backends overlap)')
    ax1.legend()

    ax2 = axes[1]
    zoom_width = 5
    for i, backend in enumerate(backends):
        try:
            fft_func = get_fft_func(backend)
            X = fft_func(x)
            freqs = np.fft.fftfreq(N, 1/fs)
            amp = np.abs(X[:N//2])
            lw = 6 - (i * 5 / max(n_backends - 1, 1))
            ax2.plot(freqs[:N//2], amp, label=backend, linewidth=lw, color=colors[i % len(colors)])
            idx = np.argmin(np.abs(freqs[:N//2] - freq))
            ax2.scatter([freqs[idx]], [amp[idx]], s=60 - i*10, color=colors[i % len(colors)], zorder=5)
        except Exception:
            pass

    ax2.set_xlim(freq - zoom_width, freq + zoom_width)
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel('Amplitude')

    if peak_amplitudes:
        peaks_str = ', '.join([f'{b}: {v:.1f}' for b, v in peak_amplitudes.items()])
        ax2.set_title(f'Zoomed: Peak amplitudes match\n({peaks_str})')
    else:
        ax2.set_title(f'Zoomed: {freq-zoom_width}–{freq+zoom_width} Hz')
    ax2.legend()

    plt.tight_layout()
    if out_dir:
        out_path = os.path.join(out_dir, 'plot_correctness_spectrum.png')
        plt.savefig(out_path, dpi=300)
        print(f"Saved spectrum plot to {os.path.abspath(out_path)}")
    plt.close()


def test_fft_inverse_consistency(x, N, out_dir=None):
    """Check IFFT consistency."""
    print("\nTesting inverse FFT consistency for each backend\n")
    backends = get_available_backends()
    for backend in backends:
        try:
            fft_func = get_fft_func(backend)
            X = fft_func(x)

            # Import ifft based on backend
            if backend == 'scipy':
                from scipy.fft import ifft as ifft_func
            elif backend == 'numpy':
                from numpy.fft import ifft as ifft_func
            else:
                print(f"Backend: {backend:9s} | No IFFT implemented")
                continue

            x_rec = ifft_func(X)
            if np.allclose(x, x_rec.real, atol=TOL):
                print(f"Backend: {backend:9s} | IFFT consistency: PASS")
            else:
                print(f"Backend: {backend:9s} | IFFT consistency: FAIL (max abs diff: {np.max(np.abs(x - x_rec.real)):.2e})")
        except Exception as e:
            print(f"Backend: {backend:9s} | ERROR: {e}")


def test_fft_parseval(x, N, out_dir=None):
    """Check Parseval's theorem."""
    print("\nTesting Parseval's theorem for each backend\n")
    E_time = np.sum(np.abs(x) ** 2)
    backends = get_available_backends()
    for backend in backends:
        try:
            fft_func = get_fft_func(backend)
            X = fft_func(x)
            E_freq = np.sum(np.abs(X) ** 2) / N
            if np.allclose(E_time, E_freq, rtol=TOL):
                print(f"Backend: {backend:9s} | Parseval: PASS | Time energy: {E_time:.6f} | Freq energy: {E_freq:.6f}")
            else:
                print(f"Backend: {backend:9s} | Parseval: FAIL | Time energy: {E_time:.6f} | Freq energy: {E_freq:.6f} | Diff: {abs(E_time-E_freq):.2e}")
        except Exception as e:
            print(f"Backend: {backend:9s} | ERROR: {e}")


def test_rfft_parseval(x, N, out_dir=None):
    """Check Parseval's theorem for rfft."""
    print("\nTesting Parseval's theorem for real FFT (rfft)\n")
    E_time = np.sum(np.abs(x) ** 2)
    backends = ['numpy']
    try:
        import scipy.fft  # noqa: F401
        backends.append('scipy')
    except ImportError:
        pass

    for backend in backends:
        try:
            if backend == 'numpy':
                rfft_func = np.fft.rfft
            elif backend == 'scipy':
                from scipy.fft import rfft as rfft_func
            else:
                continue

            Xr = rfft_func(x)
            if N % 2 == 0:
                E_freq = (np.abs(Xr[0])**2 + np.abs(Xr[-1])**2 + 2*np.sum(np.abs(Xr[1:-1])**2)) / N
            else:
                E_freq = (np.abs(Xr[0])**2 + 2*np.sum(np.abs(Xr[1:])**2)) / N

            if np.allclose(E_time, E_freq, rtol=TOL):
                print(f"Backend: {backend:9s} | rfft Parseval: PASS | Time energy: {E_time:.6f} | Freq energy: {E_freq:.6f}")
            else:
                print(f"Backend: {backend:9s} | rfft Parseval: FAIL | Time energy: {E_time:.6f} | Freq energy: {E_freq:.6f} | Diff: {abs(E_time-E_freq):.2e}")
        except Exception as e:
            print(f"Backend: {backend:9s} | ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(description='FFT correctness tests across backends.')
    parser.add_argument('--out', type=str, default='benchmarks/out', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Quick mode (smaller tests)')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    t, x = generate_sine_wave()
    N = len(x)
    test_fft_normalization(x, N, out_dir=args.out)
    test_fft_inverse_consistency(x, N, out_dir=args.out)
    test_fft_parseval(x, N, out_dir=args.out)
    test_rfft_parseval(x, N, out_dir=args.out)


if __name__ == "__main__":
    main()
