# fftkit

One FFT API over many backends.

```python
import fftkit

fftkit.get_available_backends()      # what this machine can actually use
fft = fftkit.get_fft_func("mkl")     # or scipy, numpy, cupy, torch, ...
spectrum = fft(signal)
```

Swapping FFT libraries usually means rewriting call sites, because each library
spells the same transform differently. `fftkit` puts a single callable in front
of eight of them, reports which ones are installed, and measures which one is
fastest for your array sizes.

Extracted from [`openmodalpy`](https://github.com/openfluids/openmodalpy).

## Status

Released — `0.1.0` is on PyPI. The backend dispatch layer, the correctness
invariants, and the benchmark suite are all in place. See `CHANGELOG.md`.

## Install

```bash
pip install fftkit               # numpy + scipy only
pip install "fftkit[mkl]"        # + Intel MKL
pip install "fftkit[gpu]"        # + CuPy / PyTorch
pip install "fftkit[bench]"      # + benchmark plotting deps
```

## License

MIT — see `LICENSE`.
