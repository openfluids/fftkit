![fftkit banner](https://raw.githubusercontent.com/openfluids/fftkit/main/assets/readme-banner-v3.jpg)

[![CI](https://github.com/openfluids/fftkit/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/openfluids/fftkit/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/fftkit.svg)](https://pypi.org/project/fftkit/)
[![Python](https://img.shields.io/pypi/pyversions/fftkit.svg)](https://pypi.org/project/fftkit/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

`fftkit` serves two purposes. `spectrum()` takes a time series from a simulation
or an experiment and returns a power spectral density with the choices a physical
analysis needs already made. Below that sits one FFT API over eight backends:
scipy, numpy, Intel MKL, CuPy, PyTorch, TensorFlow, pyFFTW, Apple Accelerate. The
call signature is identical everywhere, so you can swap `backend=` and leave the
rest of your code alone.

```python
import fftkit

# You have a signal and you want its spectrum.
result = fftkit.spectrum(x, t=t)    # see "Spectra of physical signals" below

# You want a faster FFT behind an API you already know.
fftkit.fft(x)                       # forward complex FFT, default backend
fftkit.rfft(x, backend="mkl")       # real-input FFT on a specific backend
fftkit.fftn(x_3d, axes=(0, 1))      # N-D transform

fftkit.get_available_backends()     # what this machine can actually use
```

`fftkit.fft`, `ifft`, `rfft`, `irfft`, `fft2`, `ifft2`, `fftn`, `ifftn` are
all available at the top level, each taking `n=`/`s=`, `axis=`/`axes=`,
`norm=`, and `backend=`. `fftfreq`/`rfftfreq` are re-exported from scipy for
convenience.

## Spectra of physical signals

Simulation output rarely arrives ready for an FFT. A solver limited by its CFL
condition writes on a variable time step, so the record has no single sample rate
and no single Nyquist frequency. Long runs also drift, and that drift deposits
power in the lowest bins, which is exactly where the largest structures live.

Miss either of those and you still obtain a spectrum, with a slope you can fit and
publish. Nothing raises an error.

`spectrum()` handles the resampling, detrending, and windowing itself, and
reports exactly what it did.

```python
import numpy as np
import fftkit

rng = np.random.default_rng(0)
fs, n = 1000.0, 8192
dt = (1 / fs) * (1 + 0.25 * rng.uniform(-1, 1, n))   # a CFL-limited time step
t = np.cumsum(dt) - dt[0]
x = (
    np.sin(2 * np.pi * 50 * t) + 0.3 * np.sin(2 * np.pi * 120 * t)
    + 0.05 * rng.standard_normal(n) + 0.002 * t      # noise, and a slow drift
)

result = fftkit.spectrum(x, t=t)     # t may be non-uniform
print(result.summary())
```

```
PSD via periodogram: 4117 bins, 0 to 502.994 Hz at fs=1005.99 Hz
  window=hann  detrend=linear
  integrated power 0.545659 = 1.000 x variance
  trust up to 400.001 Hz
  resampled: cubic, 8232 points (fast length)
```

`result.freqs` and `result.psd` are the spectrum. Everything else is provenance:
which interpolant ran, whether an anti-alias filter was applied, and which
estimator was chosen and why. You need those details to defend a figure, and
recording them now costs less than reconstructing them six months later.

Each default, and the reason for it:

| Step | Default | Reason |
|---|---|---|
| Resample to uniform | cubic spline | 23x lower band-power error than linear interpolation, on a record of three tones sampled with 30% jitter. Linear interpolation acts as a low-pass filter with a sinc² response, attenuating the high-frequency end of the spectrum you are measuring. Its extra cost is negligible next to generating the data. |
| Grid length | already FFT-friendly | The grid spacing is free to choose, so landing on a fast length costs nothing and no zero-padding is ever needed. |
| Anti-alias | on when downsampling | Without it, energy above the new Nyquist folds back into the band. Measured on a 400 Hz tone resampled from 1000 Hz to 400 Hz: 0.000 in-band power with the filter, 0.064 without it. The tone sits above the new Nyquist, so it should have vanished entirely; instead an eighth of its power reappears at a frequency it never occupied. |
| Detrend | `linear` | A mean offset lands entirely in the zero-frequency bin; a drift smears power across the lowest frequencies. |
| Window | `hann` | A record boundary is a discontinuity, and its leakage spreads across decades of a log-log plot. |
| Scaling | density | `np.sum(psd) * df` equals the variance, so integrating a band gives that band's energy. |

Every measured number in this section is re-derived by
`benchmarks/verify_readme_claims.py`, which prints each claim next to what the
current build actually produces. The figures depend on the test conditions — how
much jitter, which tones, which seed — so run it rather than trusting the
values here if you are relying on one.

### It chooses the estimator from the signal

Coherent and turbulent signals need different estimators, and choosing the wrong
one fails silently. `spectrum()` measures how much power sits in narrow peaks
above a baseline that ignores the spectral slope, then chooses:

```python
result.method              # 'periodogram' or 'blackman_tukey'
result.method_choice.tonality   # 0.0 (broadband) to 1.0 (tonal)
result.method_choice.reason     # why, in words
```

Measured on synthetic signals at fs = 1000 Hz, N = 8192:

| Signal | Tonality | Chosen |
|---|---|---|
| Single 50 Hz tone | 1.000 | periodogram |
| Three modes (50/120/213 Hz) | 1.000 | periodogram |
| Tone + 50% noise | 0.667 | periodogram |
| Turbulence + shedding tone | 0.999 | periodogram |
| Turbulence, f^-5/3 | 0.001 | blackman_tukey |
| Steeper broadband, f^-3 | 0.000 | blackman_tukey |
| White noise | 0.008 | blackman_tukey |

A coherent signal carries no random scatter for averaging to reduce, so
segmenting the record only costs resolution and blunts the peak being measured.
On a single tone, Welch lowers the peak from 2.344 to 0.324. A turbulent signal
is one realisation of a random process, where a raw periodogram carries about
100% standard error per bin however long the record; there, smoothing is the only
remedy.

### Why the broadband choice is not Welch

Welch is the usual answer, and one keyword still selects it. But it discards the
largest scales: segmenting to `nperseg = N/8` raises the lowest resolvable
frequency eightfold, and a steep spectrum keeps most of its variance below that
new limit. Blackman–Tukey tapers the autocorrelation instead, which reduces
variance by the same amount while retaining the whole record:

| Estimator | f^-5/3 | f^-3 | Effective resolution |
|---|---|---|---|
| periodogram | 1.00 | 1.00 | 0.122 Hz |
| Welch, `nperseg=N/8` | 0.38 | 0.05 | 0.977 Hz |
| Blackman–Tukey, `nlags=N/8` | 1.00 | 1.00 | 0.977 Hz |

Numbers are the fraction of signal variance recovered by integrating the
estimate. Both smoothed estimators resolve the same 0.977 Hz; only Welch
discards the variance below it. With the default `bartlett` lag window the
Blackman–Tukey estimate is also non-negative.

Use `method='welch'` when its independent segments are the point: they yield
error bars and a direct stationarity check, since comparing one segment against
another reveals a drifting flow. Blackman–Tukey returns a single smooth curve
whose bins are correlated, with no equivalent diagnostic.
`result.power_recovered` reports the fraction of variance the estimate accounts
for, and `summary()` prints a warning when it falls below 0.9.

### Comparing estimators on your own data

```python
fftkit.compare_methods(x, fs=fs)
```

Returns bin width, effective resolution, scatter, recovered power, and peak
location and height for all three estimators, plus the recommendation and the
tonality behind it. Compare on `effective_resolution` rather than `df`:
Blackman–Tukey returns a smooth curve on the full fine grid, so its bin width
overstates its resolution by a factor of eight.

### Honest bandwidth

Non-uniform sampling has no single Nyquist frequency. The largest gap in the
record limits the highest frequency you can defend, because within that gap the
signal is unconstrained.

```python
report = fftkit.describe_sampling(t)
print(report.summary())
```

```
8192 samples over 8.18201 s, non-uniform (jitter 14.5%)
  dt: min 0.000750054  median 0.000998286  max 0.00125
  frequency range: 0.122219 Hz (1/T) to 400.001 Hz (defensible)
  nominal Nyquist from median dt: 500.859 Hz
```

The band between 400 Hz and the 501 Hz you would get from the median interval is
interpolation, not measurement. `spectrum()` carries the same figure as
`result.f_max_defensible`.

`resample_uniform(t, x)` is available on its own when you want the uniform record
rather than the spectrum. It never extrapolates beyond the input range. To reach
an FFT-friendly length it adjusts the grid spacing instead of truncating, so no
samples from an expensive run are discarded.

## Backend support

| Backend | fft/ifft | rfft/irfft | fft2/ifft2 | fftn/ifftn | Notes |
|---|---|---|---|---|---|
| numpy | yes | yes | yes | yes | always available |
| scipy | yes | yes | yes | yes | always available |
| mkl | yes | yes | yes | yes | `mkl_fft`, CPU |
| pyfftw | yes | yes | yes | yes | FFTW3 via `pyfftw` |
| cupy | yes | yes | yes | yes | NVIDIA GPU |
| torch | yes | yes | yes | yes | CPU or GPU depending on install |
| tensorflow | yes | yes | yes | **no** | no general N-D transform in `tf.signal` |
| accelerate | forward complex only | no | no | no | macOS only, power-of-two length only |

## Which backend should I use?

For CPU work, start with MKL. It won at every size in the table below, and
did so on an AMD CPU, which is worth knowing because MKL is widely assumed
to help only on Intel hardware.

GPU pays off in two situations: the data already lives on the device, or
the transform is large enough that compute outgrows transfer. A single
host-to-device round trip over PCIe costs roughly 0.5 ms, so for one small
transform a CPU backend often finishes before the data has even landed on
the GPU.

None of this transfers cleanly between machines. CPU generation, GPU model,
PCIe link, and problem size all move the answer, so measure on the hardware
you will actually use:

```bash
fftkit bench --size 65536
```

Trust that over any number in this README.

## CLI

```
$ fftkit info
=== fftkit info ===

fftkit version : 0.4.0
Python version : 3.13.13
numpy version  : 2.5.1
scipy version  : 1.18.0

Default backend: mkl  (auto-detected)

Backends:
  name        available  transforms / reason
  ------------------------------------------
  scipy       yes        fft, ifft, rfft, irfft, fft2, ifft2, fftn, ifftn
  numpy       yes        fft, ifft, rfft, irfft, fft2, ifft2, fftn, ifftn
  mkl         yes        fft, ifft, rfft, irfft, fft2, ifft2, fftn, ifftn
  pyfftw      yes        fft, ifft, rfft, irfft, fft2, ifft2, fftn, ifftn
  cupy        no         not installed / no CUDA GPU
  torch       no         not installed (pip install torch)
  tensorflow  yes        fft, ifft, rfft, irfft, fft2, ifft2
  accelerate  no         macOS only

GPU:
  available            : no (No module named 'cupy')
```

```
$ fftkit bench --size 65536 --iters 30
=== fftkit bench (size=65536, iterations=30) ===
Speedup is relative to the slowest backend in this run.

  backend      time (ms)   speedup
  mkl             0.1252     7.45x
  pyfftw          0.3381     2.76x
  numpy           0.6017     1.55x
  scipy           0.8165     1.14x
  tensorflow      0.9331     1.00x
```

Timings are not stable to the last digit. On a shared or virtualised machine a
single invocation of this command varies by half again between runs, and an
unlucky first run can be several times slower while threads spin up. The table
further down is a best-of-nine to suppress that; this block is one ordinary
invocation, so the two do not match.

`--json` emits machine-readable output instead of the table, and `--gpu`
switches to a CPU-vs-GPU comparison sweep.

## Install

```bash
pip install fftkit               # numpy + scipy only, always works
```

That is the whole library. Every backend is probed at import time and degrades to
"unavailable" rather than failing, so an extra only ever adds a faster path — it
is never required, and omitting one cannot break your code.

### Choosing an extra

Sizes are the largest CPython 3.13 manylinux wheel on PyPI, measured
2026-07-25. They move; treat them as orders of magnitude.

| Extra | Adds | Wheel | Worth it when |
|---|---|---|---|
| `[mkl]` | Intel MKL via `mkl_fft` | 0.2 MB | Intel CPU, large 1-D transforms. Best return of any extra for its size. |
| `[pyfftw]` | FFTW3 via `pyfftw` | 3 MB | You want FFTW, or per-call thread control via `workers=`. |
| `[torch]` | PyTorch | 527 MB | You already use torch, or you want CUDA without CuPy. |
| `[tensorflow]` | TensorFlow | 573 MB | You already use TensorFlow. Partial matrix: no `fftn`/`ifftn`. |
| `[cuda12]` / `[cuda11]` | CuPy for that CUDA major | 144 MB | NVIDIA GPU, large batches. `[gpu]` is an alias for `[cuda12]`. |
| `[all]` | mkl + pyfftw + torch + tensorflow | **see below** | Rarely. Read the note first. |
| `[bench]` | matplotlib, tabulate | small | Running `benchmarks/` yourself. |

### Read this before `pip install "fftkit[all]"`

`[all]` is every CPU-installable backend, and it is a **multi-gigabyte install**
on Linux. Two reasons, and neither is obvious from the table:

- The PyPI wheels for `torch` and `tensorflow` are built against CUDA. On Linux
  `torch` alone pulls **15 separate `nvidia-*` runtime packages**, whether or not
  a GPU exists on the machine.
- Wheel size understates the result. `tensorflow-cpu` is a 274 MB download that
  occupies **1.3 GB** unpacked.

So `[all]` is the wrong default for a container image, a CI job, or a laptop. Two
better options:

```bash
# Install only what you will actually dispatch to.
pip install "fftkit[mkl,pyfftw]"        # both together: about 3 MB

# Or take the CPU-only builds from their own indexes, which skip the CUDA stack.
pip install fftkit tensorflow-cpu
pip install fftkit torch --index-url https://download.pytorch.org/whl/cpu
```

`tensorflow-cpu` registers under the backend name `tensorflow`, exactly as the
full package does — fftkit cannot tell them apart and does not need to.

### GPU

`pip install cupy` is a **source-only sdist** that triggers a full local CUDA
toolkit compile. Use the prebuilt wheel for your CUDA major version instead:

```bash
pip install "fftkit[cuda12]"     # CuPy for CUDA 12.x (gpu is an alias for this)
pip install "fftkit[cuda11]"     # CuPy for CUDA 11.x
```

Check your CUDA version with `nvcc --version` before picking one, and confirm the
driver is present with `nvidia-smi`. A mismatch between wheel and driver shows up
as an import failure, which fftkit reports as the backend being unavailable
rather than as an error.

### Notes per backend

- **MKL**: on Intel systems, load the oneAPI environment first
  (`source /opt/intel/oneapi/setvars.sh`) if you have it installed;
  otherwise `pip install "fftkit[mkl]"` is sufficient. Verify with
  `python -c "import mkl_fft"`. Note that `mkl_fft` does not declare the separate
  `mkl` package as a dependency, so `register_mkl_scipy_backend()` can need it
  even when the backend itself works — the warning it raises says which.
- **pyFFTW**: no extra setup beyond the pip install. fftkit turns on
  pyfftw's plan cache, which pyfftw leaves off by default; without it every
  call re-plans the transform, which costs more than the transform itself at
  small sizes.
- **CuPy**: requires an NVIDIA GPU with a matching CUDA toolkit and driver.
- **PyTorch**: CPU or CUDA build, whichever you install; fftkit uses what is
  present. The CPU-only build comes from PyTorch's own index, not PyPI.
- **TensorFlow**: implements a partial matrix. `fft`, `ifft`, `rfft`, `irfft`,
  `fft2` and `ifft2` all work, with every `norm` and with `n=`/`s=`.
  `fftn`/`ifftn` raise `NotImplementedError`, because `tf.signal` has no general
  N-dimensional transform to wrap.
- **Accelerate**: macOS only, with no separate install. A thin ctypes wrapper
  around the system Accelerate framework, implementing only a 1-D complex
  forward FFT of power-of-two length.

Run `fftkit info` to see which of these the current machine will actually use.

### Environment variables

`FFTKIT_BACKEND=<name>` overrides auto-detection (`PYMODAL_FFT_BACKEND` is
honoured too, for existing modalpy users).

### Working on fftkit

```bash
uv sync --group dev                              # tests, linter, type checker
uv sync --group dev --group backends             # + pyfftw, mkl_fft, tensorflow-cpu
uv sync --group dev --group backends --extra bench   # + matplotlib, tabulate
```

`benchmarks/` holds the scripts behind the numbers in this README.
`verify_readme_claims.py` re-derives every measured claim in the spectral
sections and prints it next to the README's wording, so a claim that drifts
shows up as a disagreement rather than going unnoticed. `bench_backends.py`
sweeps the backend timings across many transform lengths.

The `backends` group matters more than it looks. Cross-backend agreement tests
are parametrized over every registered backend and skip the ones that are absent,
so a missing install does not fail — it quietly removes the tests. On Python 3.14
`tensorflow-cpu` has no wheel yet and is skipped by a marker, so use 3.13 if you
want that backend covered locally.

## Measured numbers

Measured 2026-07-28. Machine: `nexus-dev`, a Linux VM with 18 vCPUs on an
AMD Ryzen 9 9900X host, Python 3.13.13, numpy 2.5.1, scipy 1.18.0,
mkl-fft 2.2.1, pyfftw 0.15.1, tensorflow-cpu 2.21.0. There is no GPU, so the
table covers CPU backends only and says nothing either way about GPU
performance.

`fftkit bench`, ms per FFT, complex 1-D forward transform, lower is better.
Each entry is the best of nine invocations of 50 iterations. Best-of-N rather
than the mean, because contention only ever adds time: on a shared machine the
minimum is the closest estimate of what the transform actually costs.

| N | mkl | pyfftw | scipy | numpy | tensorflow |
|---|---|---|---|---|---|
| 1024 | 0.0032 | 0.0061 | 0.0057 | 0.0064 | 0.0879 |
| 65536 | 0.1815 | 0.3375 | 0.7838 | 0.5941 | 0.8776 |
| 262144 | 0.3753 | 1.0017 | 2.3958 | 2.6466 | 2.3480 |

MKL wins at every size. Its margin over the *fastest alternative* — the number
that should drive the decision — is 1.8x at N=1024, 1.9x at 65536, and 2.7x at
262144. Quoting a margin over the slowest backend instead would read as 27x at
N=1024, which says more about TensorFlow's per-call overhead than about MKL.

pyfftw is second at large N. At N=1024 the three non-MKL CPU backends land
within 12% of each other, which is inside this machine's run-to-run scatter:
at that size you are measuring dispatch overhead, not the transform, so their
ordering there should not be read as meaningful.

Specific to this CPU and this library stack, and to a virtualised host that
other work shares. Run `fftkit bench` on your own hardware before picking a
backend.

Test suite: 1426 tests with every CPU backend installed. How many actually run
depends on what is present:

| Installed | Pass | Skip |
|---|---|---|
| scipy + numpy only | 620 | 773 |
| + mkl, pyfftw | 854 | 539 |
| + tensorflow-cpu | 989 | 437 |

Adding mkl and pyfftw activates 234 tests; adding tensorflow another 135. Most
of the suite is parametrized over every registered backend rather than only the
installed ones, so a missing backend silently removes its tests instead of
failing them. That is the failure mode the `backends` dependency group exists to
prevent — the tensorflow adapter shipped a defect in 0.3.0 precisely because its
tests ran nowhere.

## Spectral helpers

These are the individual estimators, useful when you want one directly rather than
through [`spectrum()`](#spectra-of-physical-signals). They take an already-uniform
signal and apply no detrending or windowing of their own.

`periodogram_rfft`, `welch_method`, and `blackman_tukey_rfft` compute
one-sided power spectral density estimates from a real signal, and `find_peaks`
extracts peaks from one. All three estimators integrate to the signal variance,
so you can check the scaling yourself:

```python
freqs, psd = fftkit.periodogram_rfft(x, fs)
np.sum(psd) * (freqs[1] - freqs[0]) / np.var(x)   # 1.0
```

Use `sum(psd) * df` rather than `np.trapezoid`: the trapezoidal rule
half-weights the first and last bins, but DC and Nyquist are full bins in the
underlying discrete sum, so trapezoid is biased low. On white noise the sum is
exact to every digit at any length, while trapezoid returns 0.9986 at N=64 and
0.9991 at N=256 — the bias is one bin's worth, so it shrinks as N grows and
never quite vanishes.

`calculate_error` compares a detected peak set against known frequencies. Its
default only measures true-to-detected distance, which makes spurious
detections free — pass `symmetric=True` to penalise them, which is what you
want when ranking estimators against each other.

`blackman_tukey_rfft` applies a symmetric lag window to the autocorrelation
before transforming. The estimate is guaranteed non-negative only for lag
windows whose own Fourier transform is non-negative: `bartlett` and `parzen`
qualify. `hann`, `hamming`, and `blackman` all have negative sidelobes in
their transform, and on some signals they return slightly negative PSD
values. Those values are a property of the window you chose. If you need a
strictly non-negative estimate, use `bartlett` or `parzen`.

## Migrating from 0.1.0

The default `axis` for 2-D and higher-dimensional arrays changed from `0`
to `-1` in `0.2.0`, to match NumPy and SciPy conventions. 1-D signals are
unaffected. If you pass a 2-D or N-D array and relied on the old default,
pass `axis=0` (or the appropriate `axes=`) explicitly.

That change gives a different answer without raising, so fftkit warns about
the one case where it can bite:

```python
fftkit.fft(x_2d)            # AxisDefaultWarning: used to be axis=0, now axis=-1
fftkit.fft(x_2d, axis=-1)   # silent, new behaviour
fftkit.fft(x_2d, axis=0)    # silent, 0.1.x behaviour
fftkit.fft(x_1d)            # silent: identical either way
```

`get_fft_func()(x_2d)` warns too, since that is the 0.1.x calling style. Pass
`axis` explicitly to resolve it, or silence the category outright:

```python
warnings.filterwarnings("ignore", category=fftkit.AxisDefaultWarning)
```

This warning was originally slated for removal in `0.3.0`. It is still here, and
now deliberately so: a 0.1.x caller who passes a 2-D array without an explicit
`axis` gets silently different numbers, and this is the only thing that tells
them. It has no removal date.

## License

Apache-2.0. `0.1.0` was released under MIT and remains MIT; Apache-2.0
applies from `0.2.0` onward. See `LICENSE` and `NOTICE`.

Extracted from [`openmodalpy`](https://github.com/openfluids/openmodalpy),
where it lived as `modalpy.fft` before that project was renamed.
