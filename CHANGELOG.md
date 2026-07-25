# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-07-25

### Fixed
- The `tensorflow` backend raised on real input. `tf.signal.fft`/`ifft` accept
  only `complex64`/`complex128`, and the adapter forwarded `float64` unchanged,
  so `fftkit.fft(real_array, backend="tensorflow")` failed with
  `InvalidArgumentError: Value for attr 'Tcomplex' of double is not in the list
  of allowed values`. Real input is now cast by precision rather than flattened
  to `complex64`: `float64` to `complex128`, `float32` to `complex64`, complex
  input untouched. Present since `0.3.0`, which removed a blanket `complex64`
  cast to preserve precision without adding a replacement for real input.
  `rfft`/`irfft` were unaffected, since `tf.signal.rfft` takes real input.
- The `tensorflow` backend rejected `norm='ortho'` and `norm='forward'` with
  `NotImplementedError`. `tf.signal` has no `norm=` argument, but the convention
  is only a scale factor, so the limitation was artificial. All three norms now
  work across `fft`, `ifft`, `rfft`, `irfft`, `fft2` and `ifft2`, verified
  against `scipy.fft` for every combination of transform, norm, and `n=`/`s=`.
- **A defect class, not just a defect.** The two fixes above are the first
  changes to the tensorflow backend ever exercised by a test run. CI's
  `test-backends` job installed `pyfftw`, `mkl-fft`, and `torch` and hard-gated
  on those, but never installed or required `tensorflow`, and no local
  environment had it either. Its 123 parametrized tests were collected,
  parametrized, and skipped on every machine. Installing `tensorflow-cpu` ran
  them for the first time and 9 failed immediately. CI now installs it and
  gates on it, so a silent install failure fails the job instead of deleting
  the tests it exists to run.
- The `tensorflow` norm scale for `irfft` without an explicit `n=` used the
  half-spectrum length `m`, where the convention follows the real signal length
  `2*(m-1)`. That gave a 28% error for `norm='ortho'` and 48% for
  `norm='forward'`, with nothing raised. Introduced and caught within this
  unreleased change, so never shipped.

### Changed
- New `backends` dependency group (`pyfftw`, `mkl_fft`, `tensorflow-cpu`), so the
  optional backends the agreement suite needs resolve from `uv.lock` rather than
  from ad-hoc `uv pip install` lines in CI. Install with
  `uv sync --group backends`. It is a separate group rather than part of `test`
  because `tensorflow` publishes no cp314 wheel and `test` is synced on every leg
  of the matrix including Python 3.14; the entry carries a
  `python_version < '3.14'` marker so the group stays safe to sync anywhere.
  `torch` stays outside it, since only PyTorch's own index publishes a CPU-only
  build and a per-package index cannot be expressed in a group.
- The `all` extra now includes `tensorflow`, which it had omitted while claiming
  to cover every CPU-installable backend. Note that the PyPI wheels for both
  `torch` and `tensorflow` bundle CUDA runtime libraries on Linux whether or not
  a GPU is present, so `fftkit[all]` is a multi-gigabyte install; prefer the
  individual extras if that matters.
- `AxisDefaultWarning` was documented as slated for removal in 0.3.0 and is still
  present. That is now deliberate and the docstring says so: 0.1.x callers who
  passed a 2-D array without an explicit `axis` get silently different numbers,
  and this warning is the only thing that tells them.

### Added
- `fftkit.spectrum()`, a high-level entry point for spectra of physical signals
  and simulation output. It accepts a possibly non-uniform time base, resamples
  onto a uniform grid whose length is already an FFT-friendly length (adjusting
  the spacing rather than truncating, so no samples are discarded), detrends,
  windows, and returns a density-scaled one-sided PSD together with the
  provenance of every choice made. New in the same module: `resample_uniform`,
  `describe_sampling`, `compare_methods`, `choose_method`, `tonality`, and the
  result types `SpectrumResult`, `SamplingReport`, `UniformResampling`,
  `MethodChoice`, plus `ResamplingWarning`. All exported at the top level.
- Automatic estimator selection (`method='auto'`, the default). `tonality()`
  measures the share of power in narrow peaks against a baseline that ignores the
  spectral slope, so a coherent signal routes to the periodogram and a broadband
  one to Blackman-Tukey. The decision and the measured tonality are returned in
  `SpectrumResult.method_choice` rather than applied silently. Spectral flatness
  is deliberately not used: a steep power law has a low geometric-to-arithmetic
  mean ratio, so flatness reports "tonal" for the broadband case.
- `method='blackman_tukey'` on `spectrum()`, with an `nlags=` resolution knob,
  and Blackman-Tukey added as a third row in `compare_methods()`. It is the
  broadband default rather than Welch because segmenting raises Welch's lowest
  resolvable frequency and discards the variance below it: on synthetic fields at
  matched effective resolution, Welch recovers 0.27 of the variance for
  `f^-5/3` and 0.04 for `f^-3`, against 1.00 for Blackman-Tukey. `method='welch'`
  remains available, and is the right choice when independent segments are wanted
  for error bars or a stationarity check.
- `SpectrumResult.power_recovered`, the integrated PSD over the signal variance.
  Departure from 1.0 is a scaling error for the periodogram but a physical
  low-frequency loss for Welch, so `summary()` prints a warning below 0.9 instead
  of leaving a 99% variance loss invisible.
- `compare_methods()` reports `effective_resolution` alongside `df`. Blackman-Tukey
  returns a smooth estimate on the full fine grid, so comparing estimators on bin
  width alone overstates its resolution by the segment factor.

## [0.3.0] - 2026-07-25

### Fixed
- The `tensorflow` backend silently discarded `n=` and `s=`.
  `tf.signal.fft`/`ifft` take no `fft_length`, and the adapter's fallback
  branch dropped the argument instead of applying it, so
  `fftkit.fft(x, n=32, backend="tensorflow")` returned an array of the input's
  length with no error or warning. `fft2`/`ifft2` had the same defect for `s=`
  (their `fft_length=s` branch was dead code, since the registry never maps
  them onto `rfft2d`/`irfft2d`). Both now truncate or zero-pad the input to the
  requested length, which is what `n=`/`s=` mean for a forward transform.
  Present in `0.2.0`; a mismatched `s=` length now raises `ValueError` rather
  than being partially applied.

### Fixed
- `register_mkl_scipy_backend()` returned a bare `False` for two unrelated
  causes with different fixes: `mkl_fft` missing entirely, versus `mkl_fft`
  present but `mkl_fft.interfaces.scipy_fft` also needing the separate `mkl`
  package, which `mkl-fft` does not declare as a dependency. The second case is
  actively misleading, because `mkl_available()` is `True` and
  `get_fft_func('mkl')` works while this function reports failure. It now emits
  `fftkit.MklBackendWarning` naming the module that failed and the matching fix.
  The old test could not catch this: it skipped whenever `import mkl_fft`
  succeeded, so on any machine with working MKL it never ran.

### Changed
- Dependency-group floors are now meaningful (`ruff>=0.16`, `pytest>=8.4`,
  `mypy>=1.18`, `hypothesis>=6.130`, `pytest-cov>=6.0`, `scipy-stubs>=1.16`).
  `ruff>=0.1` let a years-old linter satisfy the lint group.
- Warnings are errors in the test suite (`filterwarnings = ["error"]`), so an
  upstream `DeprecationWarning` fails the run instead of scrolling past it.

### Testing
- Coverage raised from 67% to 96%; a 90% floor is enforced via
  `[tool.coverage.report] fail_under`. The only uncovered code left is the
  macOS-only Accelerate/vDSP body.

### CI
- Jobs install dependencies with `uv sync` from the groups declared in
  `pyproject.toml` instead of repeating `pip install pytest hypothesis ...` in
  four places. That duplication is what let `hypothesis` go missing from every
  test job and break collection.
- New `test-latest` job re-resolves the newest compatible dependencies,
  ignoring `uv.lock`. Syncing from the lock is right for reproducibility but
  stops CI seeing a new numpy or scipy the day it ships, and users install
  unpinned. `test-minversion` (declared floors), `test` (locked) and
  `test-latest` (newest) now cover all three.
- `uv lock --check` gate, so the lock cannot drift from the manifest.
- Dependabot now tracks `pip` as well as `github-actions`.

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

[Unreleased]: https://github.com/openfluids/fftkit/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/openfluids/fftkit/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/openfluids/fftkit/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/openfluids/fftkit/releases/tag/v0.1.0
