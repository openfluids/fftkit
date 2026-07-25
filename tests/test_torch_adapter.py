"""Unit tests for the torch backend adapter, using a recording stub.

Why a stub. Two parts of the torch backend are fftkit's own code and can fail
independently of torch: the ``axis``/``axes`` -> ``dim`` keyword translation
(torch.fft is the one backend that does not take ``axis``), and the deliberate
absence of a dtype cast. fftkit 0.1.0 wrote::

    x_torch.type(torch.complex64)

which silently truncated float64 input to single precision -- about 1e-7
relative error, five orders coarser than the 1e-12 cross-backend agreement
tolerance the suite asserts elsewhere. The bug survived release because torch
was never installed in CI, so the strict test simply skipped.

These tests pin that plumbing without requiring torch, by injecting a stub that
records what it was handed. They complement rather than replace the real
integration tests: the backend-parametrized suite exercises genuine torch.fft
wherever torch is installed (the CI test-backends job requires it).
"""

import sys
import types

import numpy as np
import pytest

from fftkit.backends import BACKENDS


class _FakeTensor:
    """Minimal stand-in for torch.Tensor: carries dtype, converts back to numpy."""

    def __init__(self, array):
        self._array = np.asarray(array)

    @property
    def dtype(self):
        return self._array.dtype

    def numpy(self):
        return self._array

    def type(self, dtype):  # pragma: no cover - must never be called
        raise AssertionError(
            "adapter called .type() on the tensor: this is the 0.1.0 complex64 "
            "cast that destroyed float64 precision, and it must not come back"
        )


class _Recorder:
    """Captures every call the adapter makes into the fake torch.fft namespace."""

    def __init__(self, transforms):
        self.calls = []
        self.fft = types.SimpleNamespace()
        for name in transforms:
            setattr(self.fft, name, self._make(name))

    def _make(self, name):
        def fn(tensor, **kwargs):
            self.calls.append({"transform": name, "tensor": tensor, "kwargs": kwargs})
            # Return the input unchanged; these tests are about the call, not
            # the transform. Numerical correctness of torch.fft is torch's job.
            return tensor

        return fn

    def from_numpy(self, array):
        return _FakeTensor(array)


@pytest.fixture
def stub(monkeypatch):
    rec = _Recorder(
        ["fft", "ifft", "rfft", "irfft", "fft2", "ifft2", "fftn", "ifftn"]
    )
    fake = types.ModuleType("torch")
    fake.fft = rec.fft
    fake.from_numpy = rec.from_numpy
    monkeypatch.setitem(sys.modules, "torch", fake)
    return rec


def _call(transform, *args, **kwargs):
    return BACKENDS["torch"].get(transform)(*args, **kwargs)


class TestAxisKeywordTranslation:
    """torch.fft takes dim=, not axis=. Getting this wrong is silent."""

    @pytest.mark.parametrize("transform", ["fft", "ifft", "rfft", "irfft"])
    @pytest.mark.parametrize("axis", [0, 1, -1])
    def test_axis_becomes_dim(self, stub, transform, axis):
        _call(transform, np.zeros((4, 8)), axis=axis)
        kwargs = stub.calls[-1]["kwargs"]
        assert kwargs["dim"] == axis, f"axis={axis} must be forwarded as dim={axis}"
        assert "axis" not in kwargs, "torch.fft does not accept axis=; it would TypeError"

    @pytest.mark.parametrize("transform", ["fft2", "ifft2", "fftn", "ifftn"])
    def test_axes_becomes_dim(self, stub, transform):
        _call(transform, np.zeros((3, 4, 5)), axes=(0, 2))
        kwargs = stub.calls[-1]["kwargs"]
        assert kwargs["dim"] == (0, 2)
        assert "axes" not in kwargs

    def test_n_and_norm_pass_through_unrenamed(self, stub):
        """torch keeps n= and norm= under their numpy names; only the axis
        keyword differs, so renaming these too would be a regression."""
        _call("fft", np.zeros(16), n=32, norm="ortho")
        kwargs = stub.calls[-1]["kwargs"]
        assert kwargs["n"] == 32
        assert kwargs["norm"] == "ortho"

    def test_s_passes_through_for_nd(self, stub):
        _call("fftn", np.zeros((4, 4)), s=(8, 8))
        assert stub.calls[-1]["kwargs"]["s"] == (8, 8)


class TestDtypePreservation:
    """Regression guard for the 0.1.0 complex64 truncation."""

    @pytest.mark.parametrize(
        "dtype",
        [np.float64, np.float32, np.complex128, np.complex64],
    )
    def test_input_dtype_reaches_torch_unchanged(self, stub, dtype):
        x = np.ones(8, dtype=dtype)
        _call("fft", x)
        handed_over = stub.calls[-1]["tensor"].dtype
        assert handed_over == dtype, (
            f"adapter changed dtype {dtype} -> {handed_over} before calling "
            "torch.fft; float64 input must not be truncated to single precision"
        )

    def test_float64_is_not_downcast(self, stub):
        """Stated separately because this is the exact released bug: 0.1.0 cast
        every input to complex64, so a float64 signal came back with ~1e-7
        relative error against a suite that asserts 1e-12."""
        _call("fft", np.ones(8, dtype=np.float64))
        assert stub.calls[-1]["tensor"].dtype == np.float64


class TestUnsupportedTransforms:
    def test_missing_transform_raises_not_implemented(self, monkeypatch):
        """A torch build lacking a transform must fail loudly and name itself,
        not fall through to another backend."""
        fake = types.ModuleType("torch")
        fake.fft = types.SimpleNamespace()  # empty namespace: nothing implemented
        fake.from_numpy = _FakeTensor
        monkeypatch.setitem(sys.modules, "torch", fake)
        with pytest.raises(NotImplementedError, match="torch.*fftn"):
            _call("fftn", np.zeros((2, 2)))


class TestNumpyRoundTrip:
    def test_returns_numpy_not_tensor(self, stub):
        result = _call("fft", np.zeros(8))
        assert isinstance(result, np.ndarray), (
            "the adapter must return numpy; leaking a tensor would break every "
            "caller that expects the same type from every backend"
        )
