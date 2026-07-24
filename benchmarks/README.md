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
**What it measures**: Which interpolation method best preserves spectrum when resampling non-uniform data

Resampling to uniform time grid is essential because FFT requires constant sampling intervals, but many simulations (adaptive, event-driven) produce variable dt data.

**Outputs**:
- `study_interpolation_noise_*.png` - Interpolation method comparison plots
- `study_interpolation_mse_vs_noise.png` - MSE stability across noise levels
- `study_interpolation.json` - MSE scores and robustness statistics

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
