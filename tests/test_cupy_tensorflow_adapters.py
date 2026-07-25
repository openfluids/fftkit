"""Unit tests for the cupy and tensorflow backend adapters, using recording
stubs (same rationale and pattern as test_torch_adapter.py).

Neither cupy nor tensorflow is installable in this environment (no GPU, and
tensorflow is a heavy CPU-only dependency not worth pulling in just for
tests). But the adapter logic around each -- host<->device transfer
book-keeping for cupy, axis-moving/fft_length/norm-checking for tensorflow
-- is fftkit's own code and can fail independently of whether the real
libraries are installed. These stubs pin that logic without requiring
either package, exactly as test_torch_adapter.py does for torch.
"""

import sys
import types

import numpy as np
import pytest

from fftkit.backends import BACKENDS

# =============================================================================
# cupy adapter
# =============================================================================

class _FakeCupyArray:
    """Minimal stand-in for a cupy ndarray: tags itself as GPU-resident."""

    def __init__(self, array):
        self._array = np.asarray(array)
        self.on_gpu = True


class _CupyRecorder:
    """Fake cupy module recording asarray/asnumpy/fft calls."""

    def __init__(self, transforms):
        self.asarray_calls = []
        self.asnumpy_calls = []
        self.fft_calls = []
        self.fft = types.SimpleNamespace()
        for name in transforms:
            setattr(self.fft, name, self._make_fft(name))

    def _make_fft(self, name):
        def fn(x_gpu, *args, **kwargs):
            assert isinstance(x_gpu, _FakeCupyArray), (
                f"cupy.fft.{name} must receive a GPU array (the result of "
                "cp.asarray), not the raw host input"
            )
            self.fft_calls.append({"transform": name, "args": args, "kwargs": kwargs})
            return x_gpu  # identity: correctness of cupy.fft itself is not our job

        return fn

    def asarray(self, x):
        self.asarray_calls.append(x)
        return _FakeCupyArray(x)

    def asnumpy(self, x_gpu):
        assert isinstance(x_gpu, _FakeCupyArray), "asnumpy() must receive the GPU array, not host data"
        self.asnumpy_calls.append(x_gpu)
        return x_gpu._array


@pytest.fixture
def cupy_stub(monkeypatch):
    rec = _CupyRecorder(["fft", "ifft", "rfft", "irfft", "fft2", "ifft2", "fftn", "ifftn"])
    fake = types.ModuleType("cupy")
    fake.fft = rec.fft
    fake.asarray = rec.asarray
    fake.asnumpy = rec.asnumpy
    monkeypatch.setitem(sys.modules, "cupy", fake)
    return rec


class TestCupyHostDeviceTransfer:
    """cupy_backend's fn() must transfer host -> device -> host around the
    call, and hand back plain numpy, not a lingering GPU array.
    """

    def test_input_is_transferred_to_device_before_fft(self, cupy_stub):
        x = np.arange(8, dtype=np.complex128)
        BACKENDS["cupy"].get("fft")(x)
        assert len(cupy_stub.asarray_calls) == 1
        np.testing.assert_array_equal(cupy_stub.asarray_calls[0], x)

    def test_result_is_transferred_back_to_host(self, cupy_stub):
        x = np.arange(8, dtype=np.complex128)
        result = BACKENDS["cupy"].get("fft")(x)
        assert isinstance(result, np.ndarray), "cupy adapter must return numpy, not a lingering GPU array"
        np.testing.assert_array_equal(result, x)

    def test_args_and_kwargs_forwarded_to_cupy_fft(self, cupy_stub):
        x = np.arange(16, dtype=np.complex128)
        BACKENDS["cupy"].get("fft")(x, n=8, axis=0, norm="ortho")
        call = cupy_stub.fft_calls[-1]
        assert call["kwargs"] == {"n": 8, "axis": 0, "norm": "ortho"}

    def test_missing_transform_raises_not_implemented(self, monkeypatch):
        """A cupy build lacking a transform (e.g. old cuFFT wrapper) must
        fail loudly and name itself."""
        fake = types.ModuleType("cupy")
        fake.fft = types.SimpleNamespace()  # nothing implemented
        fake.asarray = lambda x: _FakeCupyArray(x)
        fake.asnumpy = lambda x: x._array
        monkeypatch.setitem(sys.modules, "cupy", fake)
        with pytest.raises(NotImplementedError, match="cupy.*fftn"):
            BACKENDS["cupy"].get("fftn")(np.zeros((2, 2)))


# =============================================================================
# tensorflow adapter
# =============================================================================

class _FakeTFTensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    def numpy(self):
        return self.array


class _TFNumpyExperimental:
    """Records moveaxis calls; behaves like real numpy.moveaxis on the array."""

    def __init__(self):
        self.calls = []

    def moveaxis(self, x, source, destination):
        self.calls.append((source, destination))
        moved = np.moveaxis(x.array, source, destination)
        return _FakeTFTensor(moved)


class _TFSignalRecorder:
    def __init__(self):
        self.calls = []

    def _make(self, name):
        def fn(x_tf, fft_length=None):
            self.calls.append({"name": name, "fft_length": fft_length, "shape": x_tf.array.shape})
            return x_tf  # identity: correctness of tf.signal itself is not our job

        return fn

    def __getattr__(self, name):
        return self._make(name)


@pytest.fixture
def tf_stub(monkeypatch):
    fake = types.ModuleType("tensorflow")
    fake.convert_to_tensor = lambda x: _FakeTFTensor(x)
    numpy_exp = _TFNumpyExperimental()
    fake.experimental = types.SimpleNamespace(numpy=numpy_exp)
    signal = _TFSignalRecorder()
    fake.signal = signal
    monkeypatch.setitem(sys.modules, "tensorflow", fake)
    return types.SimpleNamespace(moveaxis=numpy_exp, signal=signal)


def _call(transform, *args, **kwargs):
    return BACKENDS["tensorflow"].get(transform)(*args, **kwargs)


class TestTensorflowAxisMovement:
    """tf.signal always transforms the innermost axis; the adapter must
    move the requested axis to the end before calling and back after.
    """

    @pytest.mark.parametrize("axis", [0, 1, -1])
    def test_1d_moves_axis_to_last_and_back(self, tf_stub, axis):
        x = np.zeros((4, 8))
        _call("fft", x, axis=axis)
        # Two moveaxis calls: into place before the transform, back after.
        assert tf_stub.moveaxis.calls == [(axis, -1), (-1, axis)]

    def test_2d_default_axes_use_last_two(self, tf_stub):
        x = np.zeros((3, 4, 5))
        _call("fft2", x)
        assert tf_stub.moveaxis.calls == [((-2, -1), (-2, -1)), ((-2, -1), (-2, -1))]

    def test_2d_explicit_axes_forwarded(self, tf_stub):
        x = np.zeros((3, 4, 5))
        _call("fft2", x, axes=(0, 1))
        assert tf_stub.moveaxis.calls == [((0, 1), (-2, -1)), ((-2, -1), (0, 1))]


class TestTensorflowNGuard:
    """n= for 1-D transforms must translate to fft_length=[n]; without n=,
    fft_length stays None so tf.signal applies its own default length.
    """

    def test_n_becomes_fft_length_list_for_rfft(self, tf_stub):
        _call("rfft", np.zeros(16), n=8)
        assert tf_stub.signal.calls[-1]["fft_length"] == [8]

    def test_no_n_leaves_fft_length_none(self, tf_stub):
        _call("fft", np.zeros(16))
        assert tf_stub.signal.calls[-1]["fft_length"] is None

    def test_n_ignored_for_transform_without_fft_length_kw(self, tf_stub):
        """'fft'/'ifft' (non-real) tf.signal ops take no fft_length kwarg at
        all -- the adapter must not pass n through to them as fft_length,
        which real tf.signal.fft() would reject with a TypeError.
        """
        _call("fft", np.zeros(16), n=8)
        assert tf_stub.signal.calls[-1]["fft_length"] is None


class TestTensorflowNormGuard:
    """tf.signal has no norm= keyword at all; anything other than the
    implicit default ('backward', i.e. norm=None) must raise loudly rather
    than silently compute the wrong normalization.
    """

    @pytest.mark.parametrize("norm", ["ortho", "forward"])
    def test_non_default_norm_raises_not_implemented(self, tf_stub, norm):
        with pytest.raises(NotImplementedError, match="norm"):
            _call("fft", np.zeros(16), norm=norm)

    @pytest.mark.parametrize("norm", [None, "backward"])
    def test_default_norm_is_accepted(self, tf_stub, norm):
        # Must not raise.
        _call("fft", np.zeros(16), norm=norm)

    def test_norm_guard_applies_to_2d_transforms_too(self, tf_stub):
        with pytest.raises(NotImplementedError, match="norm"):
            _call("fft2", np.zeros((4, 4)), norm="ortho")


class TestTensorflowReturnsNumpy:
    def test_returns_numpy_not_tensor(self, tf_stub):
        result = _call("fft", np.zeros(8))
        assert isinstance(result, np.ndarray), (
            "the adapter must return numpy; leaking a tf tensor would break "
            "every caller that expects the same type from every backend"
        )
