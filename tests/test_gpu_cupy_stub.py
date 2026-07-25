"""Unit tests for fftkit.gpu's own book-keeping logic (GPUBatchFFT,
gpu_available, get_gpu_info, GPUConfig.default_vram_limit), using a
recording stub for cupy -- same rationale and pattern as
test_torch_adapter.py / test_cupy_tensorflow_adapters.py.

CuPy is not installable in this environment (no GPU), so none of this logic
is otherwise reachable: allocator-install-once semantics, the
CuPy-version-compatibility branching around the plan cache API
(set_size/max_size/neither), and the GPU-available branches of
fft_batch/rfft_batch. All of that is fftkit's own code, independent of
whether cuFFT itself is correct, so a stub is the right tool: it pins the
book-keeping without asserting anything about real cuFFT numerics.
"""

import sys
import types

import numpy as np
import pytest

import fftkit.gpu as gpu_mod


class _FakeGPUArray:
    """Minimal stand-in for a cupy ndarray."""

    def __init__(self, array):
        self.array = np.asarray(array)


class _FakeDevice:
    def __init__(self, device_id=0, free=2_000_000_000, total=8_000_000_000, compute_capability="86"):
        self.id = device_id
        self._free = free
        self._total = total
        self.compute_capability = compute_capability

    @property
    def mem_info(self):
        return (self._free, self._total)


class _FakeMemoryPool:
    def __init__(self):
        self.limit = None
        self._used = 0
        self._total = 0
        self.freed = False
        self.malloc_calls = 0

    def set_limit(self, size):
        self.limit = size

    def used_bytes(self):
        return self._used

    def total_bytes(self):
        return self._total

    def free_all_blocks(self):
        self.freed = True

    def malloc(self, size):
        self.malloc_calls += 1
        return size


class _FakePlanCache:
    """Plan cache exposing only `set_size`, mirroring one real CuPy version."""

    def __init__(self):
        self.size = None
        self.cleared = False

    def set_size(self, size):
        self.size = size

    def clear(self):
        self.cleared = True


class _FakeFFTConfig:
    def __init__(self, plan_cache):
        self._plan_cache = plan_cache

    def get_plan_cache(self):
        return self._plan_cache


class _FakeFFT:
    def __init__(self, plan_cache):
        self.config = _FakeFFTConfig(plan_cache)
        self.calls = []

    def _make(self, name):
        def fn(x_gpu, axis=-1):
            assert isinstance(x_gpu, _FakeGPUArray), f"cp.fft.{name} must receive a device array"
            self.calls.append({"name": name, "axis": axis})
            return _FakeGPUArray(np.fft.__dict__[name](x_gpu.array, axis=axis))

        return fn

    def __getattr__(self, name):
        return self._make(name)


class _FakeCuda:
    def __init__(self, device, mempool):
        self._device = device
        self._mempool_factory = mempool
        self.allocator_calls = []
        self.MemoryPool = mempool

    def Device(self, device_id=0):
        return self._device

    def set_allocator(self, fn):
        self.allocator_calls.append(fn)


def _make_fake_cupy(device=None, plan_cache_cls=_FakePlanCache):
    device = device or _FakeDevice()
    plan_cache = plan_cache_cls()
    fake = types.ModuleType("cupy")
    fake.cuda = _FakeCuda(device, _FakeMemoryPool)
    fake.fft = _FakeFFT(plan_cache)
    fake.asarray = lambda x: _FakeGPUArray(x)
    fake.asnumpy = lambda x_gpu: x_gpu.array
    return fake, device, plan_cache


@pytest.fixture
def cupy_stub(monkeypatch):
    fake, device, plan_cache = _make_fake_cupy()
    monkeypatch.setitem(sys.modules, "cupy", fake)
    # Deterministic allocator-install state per test, regardless of test order.
    monkeypatch.setattr(gpu_mod, "_allocator_installed", False)
    monkeypatch.setattr(gpu_mod, "_default_processor", None)
    return types.SimpleNamespace(module=fake, device=device, plan_cache=plan_cache)


class TestGPUAvailableWithStub:
    def test_true_when_device_reachable(self, cupy_stub):
        assert gpu_mod.gpu_available() is True

    def test_false_when_device_raises(self, monkeypatch):
        fake = types.ModuleType("cupy")

        class _BrokenCuda:
            def Device(self, device_id=0):
                raise RuntimeError("no CUDA driver")

        fake.cuda = _BrokenCuda()
        monkeypatch.setitem(sys.modules, "cupy", fake)
        assert gpu_mod.gpu_available() is False


class TestGetGPUInfoWithStub:
    def test_available_shape(self, cupy_stub):
        info = gpu_mod.get_gpu_info()
        assert info == {
            "available": True,
            "device_id": 0,
            "compute_capability": "86",
            "memory_total_gb": 8_000_000_000 / 1e9,
            "memory_free_gb": 2_000_000_000 / 1e9,
        }


class TestDefaultVramLimitWithStub:
    def test_uses_fraction_of_free_memory(self, cupy_stub):
        expected = 2_000_000_000 * gpu_mod.GPUConfig.VRAM_FRACTION
        assert gpu_mod.GPUConfig.default_vram_limit() == pytest.approx(expected)


class TestShouldUseGPUBranches:
    """should_use_gpu()'s decision tree, reached via a plain monkeypatch of
    gpu_available rather than a full cupy stub (no device query needed for
    these branches).
    """

    def test_gpu_resident_large_batch_returns_true(self, monkeypatch):
        monkeypatch.setattr(gpu_mod, "gpu_available", lambda: True)
        result = gpu_mod.should_use_gpu(
            array_size=256, batch_size=gpu_mod.GPUConfig.BATCH_BREAKEVEN_RESIDENT, gpu_resident=True
        )
        assert result is True

    def test_gpu_resident_large_size_returns_true(self, monkeypatch):
        monkeypatch.setattr(gpu_mod, "gpu_available", lambda: True)
        result = gpu_mod.should_use_gpu(
            array_size=gpu_mod.GPUConfig.SIZE_GPU_OPTIMAL, batch_size=1, gpu_resident=True
        )
        assert result is True

    def test_gpu_resident_below_both_breakevens_returns_false(self, monkeypatch):
        monkeypatch.setattr(gpu_mod, "gpu_available", lambda: True)
        result = gpu_mod.should_use_gpu(array_size=256, batch_size=1, gpu_resident=True)
        assert result is False

    def test_non_resident_huge_transfer_size_returns_true(self, monkeypatch):
        monkeypatch.setattr(gpu_mod, "gpu_available", lambda: True)
        result = gpu_mod.should_use_gpu(
            array_size=gpu_mod.GPUConfig.SIZE_BREAKEVEN_WITH_TRANSFER, batch_size=1, gpu_resident=False
        )
        assert result is True

    def test_non_resident_small_size_returns_false(self, monkeypatch):
        monkeypatch.setattr(gpu_mod, "gpu_available", lambda: True)
        result = gpu_mod.should_use_gpu(array_size=1024, batch_size=1, gpu_resident=False)
        assert result is False

    def test_gpu_unavailable_always_false(self, monkeypatch):
        monkeypatch.setattr(gpu_mod, "gpu_available", lambda: False)
        result = gpu_mod.should_use_gpu(
            array_size=gpu_mod.GPUConfig.SIZE_BREAKEVEN_WITH_TRANSFER, batch_size=999, gpu_resident=True
        )
        assert result is False

    def test_prefer_gpu_false_always_false(self, monkeypatch):
        monkeypatch.setattr(gpu_mod, "gpu_available", lambda: True)
        result = gpu_mod.should_use_gpu(
            array_size=gpu_mod.GPUConfig.SIZE_BREAKEVEN_WITH_TRANSFER, batch_size=999,
            prefer_gpu=False, gpu_resident=True,
        )
        assert result is False


class TestGPUBatchFFTWithStub:
    """GPUBatchFFT's real-cupy-present code paths."""

    def test_construction_available_true(self, cupy_stub):
        processor = gpu_mod.GPUBatchFFT()
        assert processor.available is True

    def test_plan_cache_set_size_variant_used(self, cupy_stub):
        gpu_mod.GPUBatchFFT()
        assert cupy_stub.plan_cache.size == gpu_mod.GPUConfig.PLAN_CACHE_SIZE

    def test_plan_cache_max_size_variant_used(self, monkeypatch):
        """A different CuPy version exposes `max_size` instead of
        `set_size`; the adapter must fall back to setting the attribute
        directly rather than crashing with AttributeError.
        """
        class _MaxSizePlanCache:
            def __init__(self):
                self.max_size = None

        fake, device, plan_cache = _make_fake_cupy(plan_cache_cls=_MaxSizePlanCache)
        monkeypatch.setitem(sys.modules, "cupy", fake)
        monkeypatch.setattr(gpu_mod, "_allocator_installed", False)
        processor = gpu_mod.GPUBatchFFT(plan_cache_size=32)
        assert processor.available is True
        assert plan_cache.max_size == 32

    def test_plan_cache_neither_api_skips_silently(self, monkeypatch):
        """A plan cache with neither `set_size` nor `max_size` must not
        prevent construction -- the adapter simply skips configuring it."""
        class _NoAPIPlanCache:
            pass

        fake, device, plan_cache = _make_fake_cupy(plan_cache_cls=_NoAPIPlanCache)
        monkeypatch.setitem(sys.modules, "cupy", fake)
        monkeypatch.setattr(gpu_mod, "_allocator_installed", False)
        processor = gpu_mod.GPUBatchFFT()
        assert processor.available is True

    def test_plan_cache_config_raising_type_error_is_swallowed(self, monkeypatch):
        """A CuPy version whose plan-cache API exists but rejects the call
        (TypeError/AttributeError) must not abort construction -- the
        adapter's narrow except clause exists precisely for this."""
        class _RejectingPlanCache:
            def set_size(self, size):
                raise TypeError("set_size() takes no arguments in this build")

        fake, device, plan_cache = _make_fake_cupy(plan_cache_cls=_RejectingPlanCache)
        monkeypatch.setitem(sys.modules, "cupy", fake)
        monkeypatch.setattr(gpu_mod, "_allocator_installed", False)
        processor = gpu_mod.GPUBatchFFT()
        assert processor.available is True

    def test_first_instance_installs_global_allocator(self, cupy_stub):
        processor = gpu_mod.GPUBatchFFT()
        assert cupy_stub.module.cuda.allocator_calls == [processor._mempool.malloc]
        assert gpu_mod._allocator_installed is True

    def test_second_instance_does_not_reinstall_allocator(self, cupy_stub):
        gpu_mod.GPUBatchFFT()
        gpu_mod.GPUBatchFFT()
        assert len(cupy_stub.module.cuda.allocator_calls) == 1, (
            "only the first GPUBatchFFT in the process may install the global allocator"
        )

    def test_explicit_memory_limit_sets_mempool_limit(self, cupy_stub):
        processor = gpu_mod.GPUBatchFFT(memory_limit=6e9)
        assert processor._mempool.limit == int(6e9)

    def test_to_gpu_wraps_with_cupy_asarray(self, cupy_stub):
        processor = gpu_mod.GPUBatchFFT()
        x = np.arange(4)
        result = processor.to_gpu(x)
        assert isinstance(result, _FakeGPUArray)
        np.testing.assert_array_equal(result.array, x)

    def test_to_cpu_unwraps_with_cupy_asnumpy(self, cupy_stub):
        processor = gpu_mod.GPUBatchFFT()
        gpu_arr = _FakeGPUArray(np.arange(4))
        result = processor.to_cpu(gpu_arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.arange(4))

    def test_fft_batch_uses_gpu_when_available(self, cupy_stub):
        processor = gpu_mod.GPUBatchFFT()
        x = np.random.randn(4, 16) + 1j * np.random.randn(4, 16)
        result = processor.fft_batch(x, axis=1)
        # The stub's cp.fft.fft delegates to numpy.fft.fft under the hood,
        # so this also pins that axis is forwarded correctly end to end.
        np.testing.assert_allclose(result, np.fft.fft(x, axis=1))

    def test_rfft_batch_uses_gpu_when_available(self, cupy_stub):
        processor = gpu_mod.GPUBatchFFT()
        x = np.random.randn(4, 16)
        result = processor.rfft_batch(x, axis=1)
        np.testing.assert_allclose(result, np.fft.rfft(x, axis=1))

    def test_fft_gpu_resident_stays_on_device(self, cupy_stub):
        processor = gpu_mod.GPUBatchFFT()
        gpu_data = processor.to_gpu(np.arange(8) + 0j)
        result = processor.fft_gpu_resident(gpu_data, axis=-1)
        assert isinstance(result, _FakeGPUArray), "fft_gpu_resident must not transfer back to host"

    def test_ifft_gpu_resident_stays_on_device(self, cupy_stub):
        processor = gpu_mod.GPUBatchFFT()
        gpu_data = processor.to_gpu(np.arange(8) + 0j)
        result = processor.ifft_gpu_resident(gpu_data, axis=-1)
        assert isinstance(result, _FakeGPUArray)

    def test_memory_info_reports_pool_state(self, cupy_stub):
        processor = gpu_mod.GPUBatchFFT(memory_limit=1234)
        processor._mempool._used = 100
        processor._mempool._total = 200
        info = processor.memory_info()
        assert info == {"used_bytes": 100, "total_bytes": 200, "limit_bytes": 1234}

    def test_clear_cache_clears_plan_cache_and_frees_pool(self, cupy_stub):
        processor = gpu_mod.GPUBatchFFT()
        processor.clear_cache()
        assert cupy_stub.plan_cache.cleared is True
        assert processor._mempool.freed is True

    def test_clear_cache_swallows_plan_cache_api_mismatch(self, monkeypatch):
        """A plan cache lacking `.clear()` (AttributeError) must not stop
        clear_cache() from still freeing the memory pool."""
        class _NoClearPlanCache:
            pass

        fake, device, plan_cache = _make_fake_cupy(plan_cache_cls=_NoClearPlanCache)
        monkeypatch.setitem(sys.modules, "cupy", fake)
        monkeypatch.setattr(gpu_mod, "_allocator_installed", False)
        processor = gpu_mod.GPUBatchFFT()
        processor.clear_cache()  # must not raise
        assert processor._mempool.freed is True


class TestGpuFftConvenienceFunctionsWithStub:
    def test_gpu_fft_uses_cached_default_processor(self, cupy_stub):
        x = np.random.randn(8) + 1j * np.random.randn(8)
        result = gpu_mod.gpu_fft(x)
        np.testing.assert_allclose(result, np.fft.fft(x))
        assert gpu_mod._default_processor is not None

    def test_gpu_rfft_uses_cached_default_processor(self, cupy_stub):
        x = np.random.randn(8)
        result = gpu_mod.gpu_rfft(x)
        np.testing.assert_allclose(result, np.fft.rfft(x))


class TestBenchmarkCPUVsGPUGPUAvailableBranch:
    """The `if processor.available:` branch of benchmark_cpu_vs_gpu(),
    only reachable with a working GPU -- stubbed here since none exists on
    this machine.
    """

    def test_gpu_branch_populates_gpu_ms_and_speedup(self, cupy_stub):
        results = gpu_mod.benchmark_cpu_vs_gpu(sizes=[32], batch_sizes=[1], iterations=1)
        assert set(results.keys()) == {"32x1"}
        vals = results["32x1"]
        assert isinstance(vals["cpu_ms"], float)
        assert isinstance(vals["gpu_ms"], float)
        assert isinstance(vals["speedup"], float)

    def test_gpu_branch_warmup_failure_falls_back_to_na(self, monkeypatch, cupy_stub):
        """If the warmup fft_batch() call itself raises, that size/batch
        combination must degrade to 'N/A' rather than propagating the
        exception and aborting the whole sweep.
        """
        def _boom(*args, **kwargs):
            raise RuntimeError("synthetic cuFFT failure")

        monkeypatch.setattr(gpu_mod.GPUBatchFFT, "fft_batch", _boom)
        results = gpu_mod.benchmark_cpu_vs_gpu(sizes=[32], batch_sizes=[1], iterations=1)
        vals = results["32x1"]
        assert vals["gpu_ms"] == "N/A"
        assert vals["speedup"] == "N/A"
        assert isinstance(vals["cpu_ms"], float)


class TestBenchmarkCPUVsGPUDefaultArguments:
    """sizes=None / batch_sizes=None must fall back to the documented
    defaults, not raise or silently produce an empty sweep."""

    def test_sizes_none_uses_documented_default(self):
        # batch_sizes pinned to [1] and iterations=1 keeps this well under
        # budget even though the size sweep itself is the (larger) default.
        results = gpu_mod.benchmark_cpu_vs_gpu(sizes=None, batch_sizes=[1], iterations=1)
        assert set(results.keys()) == {f"{s}x1" for s in [1024, 4096, 16384, 65536]}

    def test_batch_sizes_none_uses_documented_default(self):
        results = gpu_mod.benchmark_cpu_vs_gpu(sizes=[64], batch_sizes=None, iterations=1)
        assert set(results.keys()) == {f"64x{b}" for b in [1, 16, 64, 128]}
