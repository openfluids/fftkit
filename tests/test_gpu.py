"""Test GPU-accelerated FFT functions."""

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


class TestGPUConfig:
    """Test GPU configuration constants."""

    def test_gpu_config_constants_exist(self):
        """GPUConfig should have expected constants."""
        assert hasattr(fftkit.GPUConfig, 'VRAM_TOTAL')
        assert hasattr(fftkit.GPUConfig, 'VRAM_LIMIT')
        assert hasattr(fftkit.GPUConfig, 'BATCH_BREAKEVEN_RESIDENT')
        assert hasattr(fftkit.GPUConfig, 'SIZE_BREAKEVEN_RESIDENT')

    def test_gpu_config_constants_reasonable(self):
        """GPUConfig values should be reasonable."""
        assert fftkit.GPUConfig.VRAM_TOTAL > 0
        assert fftkit.GPUConfig.VRAM_LIMIT > 0
        assert fftkit.GPUConfig.VRAM_LIMIT <= fftkit.GPUConfig.VRAM_TOTAL
        assert fftkit.GPUConfig.BATCH_BREAKEVEN_RESIDENT > 0
        assert fftkit.GPUConfig.SIZE_BREAKEVEN_RESIDENT > 0


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
