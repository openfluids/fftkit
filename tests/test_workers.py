"""Tests for the `workers=` threading hint on the transform matrix.

`workers` is a performance hint, never part of the numerical contract (see
the fftkit.backends module docstring), so the coverage here checks three
things and explicitly nothing about timing:

1. `workers=None` (the default) is a true no-op: identical to omitting the
   keyword entirely, on every backend.
2. Threading never changes the answer: `workers=1`/`workers=-1` agree with
   `workers=None` to the same float64 tolerance test_transforms.py uses for
   cross-backend agreement.
3. The parameter actually reaches the two backends that honour it (scipy's
   own `workers=`, pyfftw's `threads=`), and is silently swallowed -- never
   raised -- by every backend that cannot honour it.

No test in this file asserts a timing/speedup; that is measured manually
(see backends.py module docstring) and is load-dependent, which would make
an automated assertion here flaky.
"""

import os

import numpy as np
import pytest
from conftest import all_backends_param, assert_declared_limit

import fftkit
from fftkit.backends import BACKENDS, TRANSFORM_NAMES

# Reuse the float64 tolerance tier from test_transforms.py: threading must
# not change the answer, so this asserts bit-for-bit agreement is not
# required (different thread counts can still sum in a different order) but
# the same couple-of-ULP headroom used everywhere else in the suite applies.
RTOL_F64 = 1e-9
ATOL_F64 = 1e-10

_ND_TRANSFORMS = ("fft2", "ifft2", "fftn", "ifftn")


def _signal(transform):
    """A structurally valid input for `transform`, real for rfft (the only
    transform that requires it), complex everywhere else. Shapes are 1-D for
    the 1-D transform family and 2-D for the N-D family; fftn/ifftn accept
    any input, so 2-D avoids a second dedicated shape for no benefit.
    """
    rng = np.random.default_rng(0)
    shape = (8, 8) if transform in _ND_TRANSFORMS else (64,)
    if transform == "rfft":
        return rng.standard_normal(shape)
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _call(transform, x, backend, workers=None, pass_workers=True):
    func = getattr(fftkit, transform)
    if pass_workers:
        return func(x, backend=backend, workers=workers)
    return func(x, backend=backend)


# ---------------------------------------------------------------------------
# 1. workers=None is a true no-op
# ---------------------------------------------------------------------------

class TestWorkersNoneIsANoOp:
    @pytest.mark.parametrize("transform", TRANSFORM_NAMES)
    @pytest.mark.parametrize("backend", all_backends_param())
    def test_workers_none_matches_omitted(self, transform, backend, request):
        if not BACKENDS[backend].supports(transform):
            pytest.skip(f"backend '{backend}' does not implement '{transform}'")
        x = _signal(transform)
        if assert_declared_limit(backend, lambda: _call(transform, x, backend, pass_workers=False), length=64):
            return
        omitted = _call(transform, x, backend, pass_workers=False)
        explicit_none = _call(transform, x, backend, workers=None)
        assert np.array_equal(omitted, explicit_none), (
            f"{backend}/{transform}: workers=None differed from omitting workers entirely"
        )


# ---------------------------------------------------------------------------
# 2. Threading must not change the answer
# ---------------------------------------------------------------------------

class TestThreadingDoesNotChangeTheAnswer:
    @pytest.mark.parametrize("workers", [1, -1])
    @pytest.mark.parametrize("transform", TRANSFORM_NAMES)
    @pytest.mark.parametrize("backend", all_backends_param())
    def test_workers_matches_none(self, transform, backend, workers):
        if not BACKENDS[backend].supports(transform):
            pytest.skip(f"backend '{backend}' does not implement '{transform}'")
        x = _signal(transform)
        if assert_declared_limit(backend, lambda: _call(transform, x, backend, workers=None), length=64):
            return
        baseline = _call(transform, x, backend, workers=None)
        threaded = _call(transform, x, backend, workers=workers)
        assert np.allclose(threaded, baseline, rtol=RTOL_F64, atol=ATOL_F64), (
            f"{backend}/{transform}: workers={workers} changed the result "
            f"(max|diff|={np.max(np.abs(threaded - baseline)):.3e})"
        )


# ---------------------------------------------------------------------------
# 3. Backends that cannot honour workers never raise for it
# ---------------------------------------------------------------------------

class TestUnsupportingBackendsIgnoreWorkersSilently:
    @pytest.mark.parametrize("workers", [1, -1, 4])
    @pytest.mark.parametrize("transform", TRANSFORM_NAMES)
    @pytest.mark.parametrize("backend", all_backends_param())
    def test_no_raise(self, transform, backend, workers):
        if not BACKENDS[backend].supports(transform):
            pytest.skip(f"backend '{backend}' does not implement '{transform}'")
        x = _signal(transform)
        if assert_declared_limit(backend, lambda: _call(transform, x, backend, workers=workers), length=64):
            return
        # Must not raise -- workers is a hint, not part of any backend's
        # documented contract, regardless of whether that backend can honour it.
        _call(transform, x, backend, workers=workers)


# ---------------------------------------------------------------------------
# 4 & 5. Proof the parameter actually reaches scipy / pyfftw
# ---------------------------------------------------------------------------

_SCIPY_AVAILABLE = "scipy" in fftkit.get_available_backends()
_PYFFTW_AVAILABLE = "pyfftw" in fftkit.get_available_backends()


@pytest.mark.skipif(not _SCIPY_AVAILABLE, reason="scipy backend not available")
class TestScipyReceivesWorkers:
    """Recorder-style proof that `workers=` reaches scipy.fft rather than
    being swallowed en route: monkeypatch the real scipy.fft.fft, capture the
    forwarded kwargs, and check the 'workers' key directly rather than
    inferring it from the (identical, by design) numeric result.
    """

    def test_workers_forwarded_when_set(self, monkeypatch):
        import scipy.fft as spfft

        calls = []

        def recorder(*args, **kwargs):
            calls.append(kwargs)
            return np.zeros(4, dtype=np.complex128)

        monkeypatch.setattr(spfft, "fft", recorder)
        x = np.arange(4, dtype=np.complex128)

        fftkit.fft(x, backend="scipy", workers=4)
        assert calls[-1].get("workers") == 4

    def test_workers_absent_when_none(self, monkeypatch):
        import scipy.fft as spfft

        calls = []

        def recorder(*args, **kwargs):
            calls.append(kwargs)
            return np.zeros(4, dtype=np.complex128)

        monkeypatch.setattr(spfft, "fft", recorder)
        x = np.arange(4, dtype=np.complex128)

        fftkit.fft(x, backend="scipy", workers=None)
        assert "workers" not in calls[-1]

        fftkit.fft(x, backend="scipy")
        assert "workers" not in calls[-1]


@pytest.mark.skipif(not _PYFFTW_AVAILABLE, reason="pyfftw backend not available")
class TestPyfftwReceivesThreads:
    """Same proof as TestScipyReceivesWorkers, for the pyfftw adapter's
    workers -> threads mapping, including the -1 -> os.cpu_count() translation
    pyfftw needs that scipy does not (scipy treats -1 as 'all cores' itself).
    """

    def test_threads_forwarded_when_set(self, monkeypatch):
        import pyfftw.interfaces.numpy_fft as pyfftw_fft

        calls = []

        def recorder(*args, **kwargs):
            calls.append(kwargs)
            return np.zeros(4, dtype=np.complex128)

        monkeypatch.setattr(pyfftw_fft, "fft", recorder)
        x = np.arange(4, dtype=np.complex128)

        fftkit.fft(x, backend="pyfftw", workers=4)
        assert calls[-1].get("threads") == 4

    def test_minus_one_becomes_cpu_count(self, monkeypatch):
        import pyfftw.interfaces.numpy_fft as pyfftw_fft

        calls = []

        def recorder(*args, **kwargs):
            calls.append(kwargs)
            return np.zeros(4, dtype=np.complex128)

        monkeypatch.setattr(pyfftw_fft, "fft", recorder)
        x = np.arange(4, dtype=np.complex128)

        fftkit.fft(x, backend="pyfftw", workers=-1)
        assert calls[-1].get("threads") == os.cpu_count()

    def test_threads_absent_when_none(self, monkeypatch):
        import pyfftw.interfaces.numpy_fft as pyfftw_fft

        calls = []

        def recorder(*args, **kwargs):
            calls.append(kwargs)
            return np.zeros(4, dtype=np.complex128)

        monkeypatch.setattr(pyfftw_fft, "fft", recorder)
        x = np.arange(4, dtype=np.complex128)

        fftkit.fft(x, backend="pyfftw", workers=None)
        assert "threads" not in calls[-1]

        fftkit.fft(x, backend="pyfftw")
        assert "threads" not in calls[-1]


# ---------------------------------------------------------------------------
# 6. All eight transforms accept the parameter
# ---------------------------------------------------------------------------

class TestAllTransformsAcceptWorkers:
    @pytest.mark.parametrize("transform", TRANSFORM_NAMES)
    @pytest.mark.parametrize("backend", all_backends_param())
    def test_accepts_workers_kwarg(self, transform, backend):
        if not BACKENDS[backend].supports(transform):
            pytest.skip(f"backend '{backend}' does not implement '{transform}'")
        x = _signal(transform)
        if assert_declared_limit(backend, lambda: _call(transform, x, backend, workers=1), length=64):
            return
        result = _call(transform, x, backend, workers=1)
        assert result is not None
