# fftkit

[![CI](https://github.com/openfluids/fftkit/actions/workflows/ci.yml/badge.svg)](https://github.com/openfluids/fftkit/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/fftkit.svg)](https://pypi.org/project/fftkit/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

One FFT API over eight backends: scipy, numpy, Intel MKL, CuPy, PyTorch,
TensorFlow, pyFFTW, Apple Accelerate. Same call signature everywhere; swap
`backend=` and keep the rest of your code.

```python
import fftkit

fftkit.fft(x)                       # forward complex FFT, default backend
fftkit.rfft(x, backend="mkl")       # real-input FFT on a specific backend
fftkit.fftn(x_3d, axes=(0, 1))      # N-D transform

fftkit.get_available_backends()     # what this machine can actually use
```

`fftkit.fft`, `ifft`, `rfft`, `irfft`, `fft2`, `ifft2`, `fftn`, `ifftn` are
all available at the top level, each taking `n=`/`s=`, `axis=`/`axes=`,
`norm=`, and `backend=`. `fftfreq`/`rfftfreq` are re-exported from scipy for
convenience.

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

fftkit version : 0.2.0
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
  tensorflow  no         not installed (pip install tensorflow)
  accelerate  no         macOS only

GPU:
  available            : no (No module named 'cupy')
```

```
$ fftkit bench --size 65536 --iters 30
=== fftkit bench (size=65536, iterations=30) ===
Speedup is relative to the slowest backend in this run.

  backend   time (ms)   speedup
  mkl         0.6994    10.37x
  pyfftw      2.8862     2.51x
  numpy       6.3910     1.13x
  scipy       7.2527     1.00x
```

Timings move by a few percent between runs, so this block and the table
further down come from two different invocations and do not match to the
last digit.

`--json` emits machine-readable output instead of the table, and `--gpu`
switches to a CPU-vs-GPU comparison sweep.

## Install

```bash
pip install fftkit               # numpy + scipy only, always works
pip install "fftkit[mkl]"        # + Intel MKL
pip install "fftkit[pyfftw]"     # + pyFFTW / FFTW3
pip install "fftkit[torch]"      # + PyTorch
pip install "fftkit[tensorflow]" # + TensorFlow
pip install "fftkit[all]"        # mkl + pyfftw + torch (CPU-installable set)
```

`fftkit[gpu]` needs a CUDA-matched CuPy wheel and will not build from a
plain `pip install cupy`:

```bash
pip install "fftkit[cuda12]"     # CuPy for CUDA 12.x (gpu is an alias for this)
pip install "fftkit[cuda11]"     # CuPy for CUDA 11.x
```

Check your CUDA version with `nvcc --version` before picking one, and
confirm the NVIDIA driver is present with `nvidia-smi`.

Notes per backend:

- **MKL**: on Intel systems, load the oneAPI environment first
  (`source /opt/intel/oneapi/setvars.sh`) if you have it installed;
  otherwise `pip install "fftkit[mkl]"` is sufficient. Verify with
  `python -c "import mkl_fft"`.
- **pyFFTW**: no extra setup beyond the pip install. fftkit turns on
  pyfftw's plan cache, which pyfftw leaves off by default; without it every
  call re-plans the transform, which costs more than the transform itself at
  small sizes.
- **CuPy**: requires an NVIDIA GPU with a matching CUDA toolkit/driver.
- **PyTorch / TensorFlow**: install the CPU or CUDA build per their own
  instructions; fftkit picks up whichever is present.
- **Accelerate**: macOS only, with no separate install. It is a thin ctypes
  wrapper around the system Accelerate framework, and implements only a 1-D
  complex forward FFT of power-of-two length.

`FFTKIT_BACKEND=<name>` overrides auto-detection (`PYMODAL_FFT_BACKEND` is
honoured too, for existing modalpy users).

## Measured numbers

Machine: `nexus-dev`, AMD Ryzen 9 9900X (12-core), Linux, Python 3.13.13,
numpy 2.5.1, scipy 1.18.0, mkl-fft 2.2.1, pyfftw 0.15.1. This machine has
no GPU, so the table covers CPU backends only and says nothing either way
about GPU performance.

`fftkit bench`, ms per FFT, 30 iterations, lower is better:

| N | mkl | pyfftw | scipy | numpy |
|---|---|---|---|---|
| 1024 | 0.0548 | 0.2161 | 0.1607 | 0.1770 |
| 65536 | 0.6944 | 2.8463 | 7.5495 | 6.2320 |
| 262144 | 2.1393 | 11.2381 | 23.5816 | 26.4207 |

MKL wins at every size here: 10.9x over the slowest backend at 64K points
and 12.4x at 256K. pyfftw is second at large N but the slowest option at
N=1024, where per-call planning/dispatch overhead dominates a transform
that small.

Specific to this CPU and this library stack. Run `fftkit bench` on your own
hardware before picking a backend.

Test suite: 695 tests. 277 pass with only scipy and numpy available; 389 pass
once mkl and pyfftw are installed. Adding those two backends activates 112
tests that are otherwise skipped, because most of the suite is parametrized
over every registered backend rather than only the installed ones.

## Spectral helpers

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
underlying discrete sum, so trapezoid is biased low (0.9996 at N=64 where the
sum is exact).

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

The warning goes away in `0.3.0`.

## License

Apache-2.0. `0.1.0` was released under MIT and remains MIT; Apache-2.0
applies from `0.2.0` onward. See `LICENSE` and `NOTICE`.

Extracted from [`openmodalpy`](https://github.com/openfluids/openmodalpy),
where it lived as `modalpy.fft` before that project was renamed.
