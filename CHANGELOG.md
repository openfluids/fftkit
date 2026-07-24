# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-07-24

First release.

### Added
- Initial extraction of the multi-backend FFT layer from
  [`modalpy`](https://github.com/openfluids/modalpy), where it lived as
  `modalpy.fft`.
- Unified FFT interface with automatic backend selection: `get_fft_func`,
  `get_optimal_backend`, `get_available_backends`, `benchmark_backends`.
- Backend implementations for scipy, numpy, Intel MKL, CuPy and Apple
  Accelerate. Every backend is probed at import time and degrades to
  "unavailable" instead of raising, so numpy + scipy alone are sufficient.
- GPU helpers: `GPUBatchFFT`, `GPUConfig`, `gpu_fft`, `gpu_rfft`,
  `should_use_gpu`, `get_gpu_info`, `benchmark_cpu_vs_gpu`.
- Spectral estimators: `periodogram_rfft`, `welch_method`,
  `blackman_tukey_rfft`, plus `find_peaks` and `calculate_error`.
- Backend selection via the `FFTKIT_BACKEND` environment variable, with
  `PYMODAL_FFT_BACKEND` honoured as a fallback for existing modalpy users.
- Test suite covering FFT invariants, backend equivalence and spectral helpers.
- Benchmark scripts under `benchmarks/` for backend timing, PSD method
  comparison and interpolation studies.

[Unreleased]: https://github.com/openfluids/fftkit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/openfluids/fftkit/releases/tag/v0.1.0
