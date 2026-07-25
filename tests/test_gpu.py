"""Test GPU-accelerated FFT functions."""

import sys

import numpy as np
import pytest

import fftkit


class TestGPUAvailability:
    """Test GPU detection and info functions (safe to call always)."""

    def test_gpu_available_returns_bool(self):
        """gpu_available() should return a boolean."""
        result = fftkit.gpu_available()
        assert isinstance(result, (bool, np.bool_))

    def test_get_gpu_info_returns_dict(self):
        """get_gpu_info() should return a dict."""
        info = fftkit.get_gpu_info()
        assert isinstance(info, dict)
        # Should have at least an 'available' key
        assert 'available' in info

    def test_get_gpu_info_shape_matches_availability(self):
        """get_gpu_info()'s key set is a strict function of gpu_available():
        the success shape (device_id/compute_capability/memory_*) when a
        GPU is present, the failure shape ('error') when it is not -- never
        a mix of both.
        """
        info = fftkit.get_gpu_info()
        if fftkit.gpu_available():
            assert info['available'] is True
            for key in ('device_id', 'compute_capability', 'memory_total_gb', 'memory_free_gb'):
                assert key in info
            assert info['memory_total_gb'] > 0
            assert 0 <= info['memory_free_gb'] <= info['memory_total_gb']
        else:
            assert info['available'] is False
            assert 'error' in info
            assert isinstance(info['error'], str)

    def test_should_use_gpu_returns_bool(self):
        """should_use_gpu() should return a boolean."""
        result = fftkit.should_use_gpu(array_size=1024, batch_size=1)
        assert isinstance(result, (bool, np.bool_))

    def test_should_use_gpu_respects_gpu_available(self):
        """should_use_gpu() should only return True if GPU is available."""
        result = fftkit.should_use_gpu(array_size=262144, batch_size=1, prefer_gpu=True)
        if fftkit.gpu_available():
            # GPU available: might return True for large arrays
            assert isinstance(result, (bool, np.bool_))
        else:
            # GPU not available: should return False
            assert result is False


class TestAccelerateFFTPowerOfTwoGuard:
    """accelerate_fft's vDSP path requires power-of-two length; only
    reachable on macOS with the Accelerate framework actually linkable, so
    these are environment-gated rather than GPU-gated. On this Linux CI/dev
    box they SKIP (visible, named skips) rather than silently vanish.
    """

    @pytest.mark.skipif(
        sys.platform != "darwin",
        reason="Accelerate/vDSP FFT is macOS-only"
    )
    def test_non_power_of_two_length_raises_value_error(self):
        import fftkit.backends as backends
        x = np.arange(100, dtype=np.complex128)  # 100 is not a power of two
        with pytest.raises(ValueError, match="power-of-two"):
            backends.accelerate_fft(x)

    @pytest.mark.skipif(
        sys.platform != "darwin",
        reason="Accelerate/vDSP FFT is macOS-only"
    )
    def test_power_of_two_length_matches_numpy(self):
        import fftkit.backends as backends
        np.random.seed(0)
        x = np.random.randn(64) + 1j * np.random.randn(64)
        result = backends.accelerate_fft(x)
        assert np.allclose(result, np.fft.fft(x), rtol=1e-6, atol=1e-9)


class TestGPUFFTFunctions:
    """Test GPU FFT functions (skip if GPU unavailable)."""

    @pytest.mark.skipif(
        not fftkit.gpu_available(),
        reason="GPU not available"
    )
    def test_gpu_fft_small_array(self):
        """gpu_fft() should work with small arrays."""
        x = np.array([1, 2, 3, 4], dtype=np.complex128)
        result = fftkit.gpu_fft(x)

        assert result is not None
        assert len(result) == 4

    @pytest.mark.skipif(
        not fftkit.gpu_available(),
        reason="GPU not available"
    )
    def test_gpu_rfft_real_array(self):
        """gpu_rfft() should work with real arrays."""
        x = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
        result = fftkit.gpu_rfft(x)

        assert result is not None
        assert len(result) == len(x) // 2 + 1  # rfft output length

    @pytest.mark.skipif(
        not fftkit.gpu_available(),
        reason="GPU not available"
    )
    def test_gpu_fft_agrees_with_numpy(self):
        """GPU FFT should agree with numpy FFT."""
        np.random.seed(42)
        x = np.random.randn(256) + 1j * np.random.randn(256)

        result_gpu = fftkit.gpu_fft(x)
        result_numpy = np.fft.fft(x)

        assert np.allclose(result_gpu, result_numpy, rtol=1e-6, atol=1e-9), \
            "GPU FFT should match numpy FFT"


class TestGPUBatchFFT:
    """Test GPUBatchFFT class (skip if GPU unavailable)."""

    @pytest.mark.skipif(
        not fftkit.gpu_available(),
        reason="GPU not available"
    )
    def test_gpu_batch_fft_creation(self):
        """GPUBatchFFT should instantiate."""
        processor = fftkit.GPUBatchFFT()
        assert processor is not None

    @pytest.mark.skipif(
        not fftkit.gpu_available(),
        reason="GPU not available"
    )
    def test_gpu_batch_fft_available_property(self):
        """GPUBatchFFT.available should reflect GPU status."""
        processor = fftkit.GPUBatchFFT()
        assert processor.available is True  # Should be True since we got past skipif

    @pytest.mark.skipif(
        not fftkit.gpu_available(),
        reason="GPU not available"
    )
    def test_gpu_batch_fft_batch(self):
        """GPUBatchFFT.fft_batch should process batches."""
        processor = fftkit.GPUBatchFFT()
        np.random.seed(42)
        batch = np.random.randn(8, 256) + 1j * np.random.randn(8, 256)

        result = processor.fft_batch(batch, axis=1)
        assert result.shape == batch.shape

    @pytest.mark.skipif(
        not fftkit.gpu_available(),
        reason="GPU not available"
    )
    def test_gpu_batch_rfft_batch(self):
        """GPUBatchFFT.rfft_batch should process real signal batches."""
        processor = fftkit.GPUBatchFFT()
        np.random.seed(42)
        batch = np.random.randn(8, 256)  # Real signals

        result = processor.rfft_batch(batch, axis=1)
        assert result.shape[0] == batch.shape[0]  # Batch dimension preserved
        assert result.shape[1] == batch.shape[1] // 2 + 1  # rfft output length


class TestGPUBatchFFTCPUFallback:
    """Test the CPU-fallback path of fft_batch/rfft_batch when no GPU is
    available. Runs (not skipped) precisely on machines without a GPU --
    i.e. this CI/dev box -- which is the only place this path is exercised.
    """

    @pytest.mark.skipif(
        fftkit.gpu_available(),
        reason="fallback path only triggers when GPU is unavailable"
    )
    def test_fft_batch_fallback_warns_and_matches_scipy(self):
        """fallback=True (default): emits RuntimeWarning and result matches
        scipy.fft.fft on CPU, since that is exactly what the fallback calls.
        """
        np.random.seed(42)
        data = np.random.randn(4, 64) + 1j * np.random.randn(4, 64)
        processor = fftkit.GPUBatchFFT()
        assert processor.available is False

        with pytest.warns(RuntimeWarning, match="falling back to scipy.fft.fft"):
            result = processor.fft_batch(data, axis=1)

        from scipy.fft import fft as scipy_fft
        assert np.allclose(result, scipy_fft(data, axis=1))

    @pytest.mark.skipif(
        fftkit.gpu_available(),
        reason="fallback path only triggers when GPU is unavailable"
    )
    def test_rfft_batch_fallback_warns_and_matches_scipy(self):
        np.random.seed(42)
        data = np.random.randn(4, 64)
        processor = fftkit.GPUBatchFFT()

        with pytest.warns(RuntimeWarning, match="falling back to scipy.fft.rfft"):
            result = processor.rfft_batch(data, axis=1)

        from scipy.fft import rfft as scipy_rfft
        assert np.allclose(result, scipy_rfft(data, axis=1))

    @pytest.mark.skipif(
        fftkit.gpu_available(),
        reason="fallback path only triggers when GPU is unavailable"
    )
    def test_fft_batch_fallback_false_raises(self):
        """fallback=False must hard-fail instead of silently running on CPU."""
        data = np.random.randn(4, 64) + 1j * np.random.randn(4, 64)
        processor = fftkit.GPUBatchFFT()
        with pytest.raises(RuntimeError, match="GPU not available"):
            processor.fft_batch(data, axis=1, fallback=False)

    @pytest.mark.skipif(
        fftkit.gpu_available(),
        reason="fallback path only triggers when GPU is unavailable"
    )
    def test_rfft_batch_fallback_false_raises(self):
        data = np.random.randn(4, 64)
        processor = fftkit.GPUBatchFFT()
        with pytest.raises(RuntimeError, match="GPU not available"):
            processor.rfft_batch(data, axis=1, fallback=False)


class TestGPUConfig:
    """Test GPU configuration constants."""

    def test_gpu_config_constants_exist(self):
        """GPUConfig should have the device-neutral constants that replaced
        the old hardcoded VRAM_TOTAL/VRAM_LIMIT in 0.2.0 (this module is no
        longer tuned to one specific GPU's VRAM capacity).
        """
        assert hasattr(fftkit.GPUConfig, 'VRAM_FRACTION')
        assert hasattr(fftkit.GPUConfig, 'VRAM_LIMIT_FALLBACK')
        assert hasattr(fftkit.GPUConfig, 'BATCH_BREAKEVEN_RESIDENT')
        assert hasattr(fftkit.GPUConfig, 'SIZE_BREAKEVEN_RESIDENT')
        assert callable(fftkit.GPUConfig.default_vram_limit)

    def test_gpu_config_constants_reasonable(self):
        """GPUConfig values should be reasonable."""
        assert 0 < fftkit.GPUConfig.VRAM_FRACTION <= 1.0, \
            "VRAM_FRACTION is a fraction of free VRAM, must be in (0, 1]"
        assert fftkit.GPUConfig.VRAM_LIMIT_FALLBACK > 0
        assert fftkit.GPUConfig.BATCH_BREAKEVEN_RESIDENT > 0
        assert fftkit.GPUConfig.SIZE_BREAKEVEN_RESIDENT > 0

    def test_default_vram_limit_no_gpu_returns_fallback(self):
        """default_vram_limit() falls back to VRAM_LIMIT_FALLBACK when no
        CUDA device can be queried (no cupy installed / no GPU present)."""
        if fftkit.gpu_available():
            pytest.skip("GPU available; fallback path not exercised here")
        assert fftkit.GPUConfig.default_vram_limit() == fftkit.GPUConfig.VRAM_LIMIT_FALLBACK


class TestBenchmarkCPUVsGPU:
    """Test CPU vs GPU benchmarking (skip if GPU unavailable)."""

    @pytest.mark.skipif(
        not fftkit.gpu_available(),
        reason="GPU not available"
    )
    def test_benchmark_cpu_vs_gpu_callable(self):
        """benchmark_cpu_vs_gpu() should be callable."""
        # Run with small parameters to keep test fast
        results = fftkit.benchmark_cpu_vs_gpu(
            sizes=[1024],
            batch_sizes=[1],
            iterations=2
        )
        assert isinstance(results, dict)

    @pytest.mark.skipif(
        not fftkit.gpu_available(),
        reason="GPU not available"
    )
    def test_benchmark_results_structure(self):
        """Benchmark results should have expected structure."""
        results = fftkit.benchmark_cpu_vs_gpu(
            sizes=[1024],
            batch_sizes=[1],
            iterations=2
        )

        for key, val in results.items():
            assert isinstance(key, str)  # "size x batch" format
            assert isinstance(val, dict)
            assert 'cpu_ms' in val
            assert 'gpu_ms' in val
            assert 'speedup' in val
