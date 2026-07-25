# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-07-25

`0.1.0` remains MIT-licensed, as published. Apache-2.0 applies from `0.2.0`
onward; see `LICENSE` and `NOTICE`.

### Changed (BREAKING)
- CPU backends' default `axis` moved from `0` to `-1`, matching NumPy and
  SciPy. In `0.1.0`, `fftkit.get_fft_func('scipy')(x_2d)` transformed columns
  (`axis=0`) while `gpu_fft(x_2d)` transformed rows (`axis=-1`) — the same
  package meant two different things depending on backend. 1-D callers are
  unaffected. Callers passing 2-D or N-D arrays and relying on the old
  `axis=0` default must now pass `axis=0` explicitly.
- Because that change alters results without raising, the 1-D transforms now
  emit `fftkit.AxisDefaultWarning` when they receive a multi-dimensional array
  with no explicit `axis`. That is the only case whose meaning changed, so it
  is the only case that warns: 1-D input and any explicit `axis` stay silent.
  The warning covers `get_fft_func()(x)` as well, since that is the `0.1.x`
  idiom and therefore the likeliest place a 2-D array is still relying on the
  old default. Pass `axis` explicitly to resolve it, or filter the category.
  It subclasses `UserWarning` rather than `DeprecationWarning` so that library
  users — who are hidden from `DeprecationWarning` by default — actually see
  it. Slated for removal in `0.3.0`.

### Added
- `calculate_error(..., symmetric=True)` averages both nearest-neighbour
  directions so spurious detections are penalised. The default (unchanged)
  measures only true-to-detected distance, which makes extra detections free:
  a peak reported in every bin scores a perfect `0.0`, beating a correct
  answer shifted by one bin. Prefer `symmetric=True` when ranking estimators.
  With no detections it returns `inf` rather than `mean(true_peaks)`, which is
  a frequency standing in for a distance.
- Full transform matrix across all backends: `fft`, `ifft`, `rfft`, `irfft`,
  `fft2`, `ifft2`, `fftn`, `ifftn`, each accepting `n=`/`s=`, `axis=`/`axes=`,
  and `norm=`. Available as top-level `fftkit.fft(...)`, `fftkit.rfft(...)`,
  etc., plus re-exported `fftfreq`/`rfftfreq`. `0.1.0` only exposed the
  forward complex 1-D FFT.
- `fftkit` CLI: `fftkit info` (backend availability, detected default, GPU
  status) and `fftkit bench` (`--size`, `--iters`, `--batch`, `--json`,
  `--gpu`).
- Type annotations across the public API. The `py.typed` marker shipped
  since `0.1.0` is now truthful, and `mypy --strict` runs in CI.

### Fixed
- `torch` and `tensorflow` backends silently cast input to `complex64`,
  losing float64 precision.
- `gpu_fft`/`gpu_rfft` reset the global CuPy allocator with a new memory
  pool on every call.
- `blackman_tukey_rfft` applied no lag window at all, despite the window
  being the defining step of the Blackman-Tukey method; took `abs()` of the
  transform instead of the real part; and was off by a factor of 2 from the
  correct one-sided density scaling. It now integrates to the signal
  variance.
- `blackman_tukey_rfft` used a periodic window where the method requires a
  symmetric lag window.
- Removed a `scipy.fftpack.rfft` fallback that returned a different
  (real-packed) array layout than the rest of the API, silently producing
  garbage output on scipy < 1.4.
- `pyfftw` backend now enables pyfftw's plan cache, which ships disabled by
  default; every call previously re-planned the transform from scratch.
- `GPUConfig` hardcoded the constants of one 8 GB GPU; memory limits are now
  derived from the actual device.
- `__version__` was a hardcoded string that had already drifted from the
  version in `pyproject.toml`; it now derives from installed package
  metadata.

### Changed
- `GPUBatchFFT.fft_batch`/`rfft_batch` CPU fallback now warns instead of
  degrading silently; pass `fallback=False` to raise instead.
- `find_peaks`'s first parameter renamed `st` -> `freqs`.
- All 8 backends now register unconditionally. Previously `pyfftw`
  registered only if installed, so `get_backend_names()` was
  machine-dependent for that one backend; availability is now decided
  entirely by the existing probe step, not by registration.

### Testing
- Suite grew from 69 to 695 tests. Backend parametrization no longer
  evaluates at collection time, so a backend that isn't installed shows up
  as a named skip instead of silently producing no test at all. Added the
  full transform matrix, property-based tests (hypothesis), PSD scaling
  invariants, and CLI tests.

### Packaging
- The `gpu` extra pointed at the source-only `cupy` sdist, which triggers a
  local CUDA toolkit compile on install; it now installs `cupy-cuda12x`.
  Added `cuda11`, `cuda12`, `torch`, `tensorflow`, `pyfftw`, and `all`
  extras.

### CI
- Added jobs that install the real optional backends, that pin the
  declared minimum numpy/scipy versions, a Windows job, coverage, and a
  packaging check.

## [0.1.0] - 2026-07-24

First release.

### Added
- Initial extraction of the multi-backend FFT layer from
  [`openmodalpy`](https://github.com/openfluids/openmodalpy), where it lived
  as `modalpy.fft` (the project was named `modalpy` at the time).
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

[Unreleased]: https://github.com/openfluids/fftkit/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/openfluids/fftkit/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/openfluids/fftkit/releases/tag/v0.1.0
