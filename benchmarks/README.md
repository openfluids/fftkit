# fftkit Benchmark Suite

Collection of analysis scripts to study FFT backend performance, correctness, and spectral estimation methods.

## Installation

Install fftkit with benchmark dependencies:
```bash
pip install "fftkit[bench]"
```

Requires: numpy, scipy, matplotlib, tabulate

## Running the Benchmarks

All scripts run from the repository root:

```bash
cd /path/to/fftkit
uv run python benchmarks/bench_backends.py [--quick] [--out DIR]
uv run python benchmarks/study_interpolation.py [--quick] [--out DIR]
uv run python benchmarks/study_padding.py [--quick] [--out DIR]
uv run python benchmarks/compare_psd_methods.py [--quick] [--out DIR]
uv run python benchmarks/plot_correctness.py [--quick] [--out DIR]
```

Options:
- `--quick`: Fast mode with smaller/fewer tests (for smoke testing)
- `--out DIR`: Output directory for results (default: `benchmarks/out`)

## Scripts

### bench_backends.py
**What it measures**: FFT performance across backends and signal sizes

**Outputs**:
- `bench_backends.json` - Timing and error data for all backends
- `bench_backends_pow2.png` - Performance plot (power-of-2 sizes)
- `bench_backends_non_pow2.png` - Performance plot (non-power-of-2 sizes)
- `bench_backends_batch.json` - GPU batch performance comparison (if GPU available)

**Notes**: Machine-dependent. Backends not installed on your system are skipped gracefully.

### study_interpolation.py
**What it measures**: Which interpolation method distorts the spectrum least when a series is resampled onto a uniform grid, covering two structurally different cases separately (they are not comparable and are never merged into one ranking):

1. **Non-uniform source → uniform target** (the motivating case for this study: adaptive/event-driven simulations write variable-dt output, but the FFT needs constant sampling). No decimation, so aliasing is not the dominant error term here — this isolates how much distortion the interpolant itself introduces. Ground truth is the exact analytic (noiseless) signal evaluated directly on the target grid, since the test signal's functional form is known.
2. **Uniform decimation** (a fine uniform grid downsampled 116x onto a coarse one, with no anti-alias filter in the naive interpolants). Ground truth here is `scipy.signal.decimate`'s output, which low-pass filters before downsampling — comparing against the raw high-resolution spectrum instead would rank aliasing error, which is large and essentially interpolant-independent (measured: nearest / linear / cubic agree to within 0.1% on a noiseless signal), rather than genuine interpolation quality. `scipy.signal.decimate` is included as a candidate method in this case's table; it beats every naive interpolant by a wide margin, which is the case's headline finding — for heavy decimation, anti-alias filtering matters far more than which interpolant you use afterwards.

Two metrics are reported per case: whole-spectrum MSE against that case's ground truth, and peak-frequency error (`fftkit.calculate_error(..., symmetric=True)`) against the exactly-known planted tone/harmonic frequencies. `interp1d(kind='slinear')` is not included as a separate method: it is numerically identical to `kind='linear'` (measured max|difference| = float64 eps), so ranking one against the other would be ranking rounding noise. Pass `--seed` (default 0) for reproducibility; close-running methods swap places between seeds.

**Outputs**:
- `study_interpolation_nonuniform_noise_*.png`, `study_interpolation_decimation_noise_*.png` - per-case comparison plots
- `study_interpolation_mse_vs_noise_nonuniform.png`, `study_interpolation_mse_vs_noise_decimation.png` - MSE stability across noise levels, per case
- `study_interpolation.json` - MSE and peak-frequency-error scores, robustness statistics, for both cases

### study_padding.py
**What it measures**: Four questions about zero-padding, each with its own figure and JSON block:

- **2a Speed**: time to transform an awkward length directly vs. padded to `fftkit.next_fast_len`, across every available backend (radix preferences differ by backend/library).
- **2b Resolution**: two tones closer together than the Rayleigh limit (`1/T`, set by the observation window, not the transform length) stay unresolved at every padding factor tested — padding does not add resolution. What padding does buy: the padded spectrum is interpolated onto a finer frequency grid, so the *location estimate* of the (still merged) peak improves with padding.
- **2c Windowing order**: window-then-pad (correct) vs. pad-then-window (wrong, applies the taper across the zero-padded region and corrupts the sidelobe structure) — reported as a peak-amplitude difference and a sidelobe-level difference in dB.
- **2d Normalization**: fftkit's PSD estimators satisfy `np.sum(psd) * df == np.var(x)` exactly for whatever array is transformed, padded or not — but zero-padding dilutes that array's own variance, so a PSD computed on a zero-padded signal understates the *original* signal's total power by roughly the padding ratio. Compare against `var()` of the padded array you fed in, not the pre-padding signal.

**Outputs**:
- `study_padding_speed.png`, `study_padding_resolution.png`, `study_padding_window_order.png`, `study_padding_normalization.png`
- `study_padding.json` - all four sections' measured numbers

### compare_psd_methods.py
**What it measures**: Accuracy (peak detection error) and runtime of three spectral estimation methods

Compares:
- Periodogram (raw FFT)
- Blackman-Tukey (autocorrelation-based)
- Welch (segment averaging)

On a signal with two known frequencies plus harmonics.

**Outputs**:
- `compare_psd_methods_spectral.png` - PSD estimates and true peaks
- `compare_psd_methods_perf.png` - Performance vs accuracy tradeoff
- `compare_psd_methods.json` - Errors and execution times

### plot_correctness.py
**What it measures**: FFT correctness across backends

Tests:
- Normalization (unnormalized FFT convention)
- IFFT consistency (perfect reconstruction)
- Parseval's theorem (energy conservation, complex FFT and rfft)

**Outputs**:
- `plot_correctness_spectrum.png` - Spectra overlay (backends should match)

## Machine Dependency

All results are hardware- and configuration-dependent:
- Backend availability varies (MKL, GPU, etc. may not be installed)
- CPU/GPU performance depends on hardware
- Numerical values should not be compared across machines

Scripts handle missing backends gracefully: they report what was skipped and proceed with available backends.
