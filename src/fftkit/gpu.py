"""
GPU-accelerated batch FFT processing utilities.

=============================================================================
SETUP INSTRUCTIONS
=============================================================================

STEP 1: Verify NVIDIA Driver
    nvidia-smi
    # Should show your GPU (e.g., RTX 4060) and driver version

STEP 2: Install CUDA Toolkit (if not present)
    # Ubuntu/Debian:
    sudo apt install nvidia-cuda-toolkit

    # Verify:
    nvcc --version

STEP 3: Install CuPy
    # For CUDA 12.x:
    uv pip install cupy-cuda12x

    # For CUDA 11.x:
    uv pip install cupy-cuda11x

STEP 4: Verify Installation
    python -c "
    import cupy as cp
    x = cp.array([1,2,3,4])
    y = cp.fft.fft(x)
    print(f'GPU FFT OK: {cp.asnumpy(y)[:2]}')
    "

=============================================================================
WHEN TO USE GPU vs CPU
=============================================================================

Use GPU (CuPy) when:
    - Processing batches of 16+ FFTs simultaneously (GPU-resident mode)
    - Single FFTs larger than 256K points (with data transfer)
    - Data will stay on GPU for multiple operations (pipeline)

Use CPU (MKL/scipy) when:
    - Single small FFTs (<4K points)
    - Data needs immediate transfer back to CPU
    - Memory-constrained environments

GPU-RESIDENT mode (no transfer) generally wins big over CPU FFT libraries
(e.g. MKL) once batch size or FFT length crosses the breakevens documented
on GPUConfig below; the exact speedup is device-dependent.

With data transfer (H2D + FFT + D2H):
    MKL typically wins due to PCIe transfer overhead (~0.5ms per transfer)
    GPU only wins for very large single FFTs (256K+) or when data stays on GPU

=============================================================================
USAGE
=============================================================================

    from fftkit.gpu import GPUBatchFFT, should_use_gpu

    # Check if GPU is beneficial for your workload
    if should_use_gpu(array_size=65536, batch_size=64):
        processor = GPUBatchFFT()
        result = processor.fft_batch(data)
    else:
        result = scipy.fft.fft(data)

=============================================================================
MEMORY MANAGEMENT
=============================================================================

This module is not tuned to any specific GPU. By default it limits usage to
a fraction (GPUConfig.VRAM_FRACTION, default 0.6) of the *free* VRAM
reported by the active CUDA device at construction time, leaving headroom
for the driver/context, cuFFT plan cache, and OS overhead. If no device can
be queried, it falls back to a conservative constant
(GPUConfig.VRAM_LIMIT_FALLBACK).

To adjust memory limit:
    processor = GPUBatchFFT(memory_limit=6e9)  # explicit 6GB cap

To clear memory:
    processor.clear_cache()

=============================================================================
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

ArrayResult = NDArray[Any]

# =============================================================================
# Configuration for GPU FFT processing (device-neutral)
# =============================================================================

class GPUConfig:
    """Configuration for GPU FFT processing.

    This module is deliberately not tuned to any specific GPU model. The
    memory limit is derived at runtime from the active device's *free* VRAM
    (see default_vram_limit()) rather than a hardcoded capacity, and the
    performance thresholds below are device-neutral heuristics, not
    measurements from one card: GPU-resident batch FFTs generally start
    winning over a CPU library (e.g. MKL) once the batch size or FFT length
    crosses these breakevens, but the exact crossover shifts with the GPU,
    CPU, and driver in use.

    IMPORTANT: These thresholds assume GPU-RESIDENT mode (data stays on GPU).
    With data transfer (H2D + FFT + D2H), a CPU FFT library typically wins
    due to PCIe transfer overhead.
    """

    # Fraction of free VRAM (as reported by the active device) to use as the
    # default memory_limit for GPUBatchFFT. See default_vram_limit().
    VRAM_FRACTION = 0.6

    # Conservative fallback (bytes) when no CUDA device can be queried
    # (e.g. CuPy not installed, or mem_info fails).
    VRAM_LIMIT_FALLBACK = 1e9  # 1GB

    # Performance thresholds for GPU-RESIDENT mode (no transfer)
    # Device-neutral heuristics for when GPU-resident batches generally win
    BATCH_BREAKEVEN_RESIDENT = 16  # Batch size above which GPU-resident tends to win
    SIZE_BREAKEVEN_RESIDENT = 4096  # FFT size above which GPU-resident tends to win

    # Performance thresholds WITH data transfer
    # GPU tends to only win for very large single FFTs due to PCIe overhead
    SIZE_BREAKEVEN_WITH_TRANSFER = 262144  # 256K+ for GPU to win with transfer

    SIZE_GPU_OPTIMAL = 65536  # Size above which GPU-resident clearly wins

    # cuFFT plan cache (smaller = less memory, but more recompilation)
    PLAN_CACHE_SIZE = 64

    @staticmethod
    def default_vram_limit() -> float:
        """Compute a default memory_limit (bytes) from the current device.

        Uses VRAM_FRACTION (default 0.6) of the free memory reported by
        cupy.cuda.Device(0).mem_info at call time. Falls back to
        VRAM_LIMIT_FALLBACK when no CUDA device can be queried (no CuPy, no
        GPU, or the query fails for any reason).
        """
        try:
            import cupy as cp
            free, _total = cp.cuda.Device(0).mem_info
            return float(free * GPUConfig.VRAM_FRACTION)
        except Exception:
            return GPUConfig.VRAM_LIMIT_FALLBACK


# Set to True once some GPUBatchFFT instance has installed its MemoryPool as
# CuPy's process-wide default allocator (see GPUBatchFFT.__init__).
_allocator_installed = False

# =============================================================================
# GPU Availability Check
# =============================================================================

def gpu_available() -> bool:
    """Check if CUDA GPU is available and working."""
    try:
        import cupy as cp
        # Try to access device
        device = cp.cuda.Device(0)
        _ = device.compute_capability
        return True
    except Exception:
        return False


def get_gpu_info() -> dict[str, Any]:
    """Get GPU information if available."""
    try:
        import cupy as cp
        device = cp.cuda.Device(0)
        free, total = device.mem_info
        return {
            'available': True,
            'device_id': device.id,
            'compute_capability': device.compute_capability,
            'memory_total_gb': total / 1e9,
            'memory_free_gb': free / 1e9,
        }
    except Exception as e:
        return {
            'available': False,
            'error': str(e),
        }


def should_use_gpu(array_size: int, batch_size: int = 1, prefer_gpu: bool = True,
                    gpu_resident: bool = False) -> bool:
    """Decide if GPU is beneficial for given workload.

    Args:
        array_size: Total number of elements in single FFT input
        batch_size: Number of FFTs to compute
        prefer_gpu: Whether to prefer GPU when beneficial
        gpu_resident: If True, data stays on GPU (no transfer overhead).
                      This is key - GPU wins with gpu_resident=True,
                      but MKL often wins when transfers are included.

    Returns:
        bool: True if GPU is recommended, False for CPU

    Note:
        With data transfer, MKL typically wins due to PCIe overhead.
        Set gpu_resident=True for pipelines where data stays on GPU.
    """
    if not prefer_gpu or not gpu_available():
        return False

    # GPU-resident mode: GPU wins for most batches
    if gpu_resident:
        if (batch_size >= GPUConfig.BATCH_BREAKEVEN_RESIDENT or
                array_size >= GPUConfig.SIZE_GPU_OPTIMAL):
            return True

    # With transfer: GPU only wins for very large single FFTs
    if array_size >= GPUConfig.SIZE_BREAKEVEN_WITH_TRANSFER:
        return True

    return False


# =============================================================================
# GPU Batch FFT Processor
# =============================================================================

class GPUBatchFFT:
    """Efficient batch FFT processor using CuPy.

    This class handles:
    - Efficient H2D/D2H memory transfers
    - cuFFT plan caching
    - Memory management for limited VRAM

    Example:
        processor = GPUBatchFFT()

        # Process batch of signals
        signals = np.random.randn(128, 4096)  # 128 signals, 4096 points each
        spectra = processor.fft_batch(signals, axis=1)

        # Keep data on GPU for pipeline operations
        gpu_data = processor.to_gpu(signals)
        gpu_spectra = processor.fft_gpu_resident(gpu_data, axis=1)
        # ... more GPU operations ...
        result = processor.to_cpu(gpu_spectra)
    """

    def __init__(self, memory_limit: float | None = None, plan_cache_size: int | None = None) -> None:
        """Initialize GPU processor.

        Args:
            memory_limit: Max GPU memory to use (bytes). Default: a fraction
                (GPUConfig.VRAM_FRACTION) of the current device's free VRAM,
                see GPUConfig.default_vram_limit().
            plan_cache_size: cuFFT plan cache size. Default: 64

        Allocator semantics: CuPy has a single *process-wide* default
        allocator. To avoid repeatedly resetting it (which orphans any
        memory blocks a previous MemoryPool had outstanding), only the
        first GPUBatchFFT instance created in a process installs its
        MemoryPool as CuPy's global allocator via cp.cuda.set_allocator().
        Later instances (including ones built with a custom memory_limit)
        still get their own MemoryPool for accounting (memory_info(),
        clear_cache()), but that pool will not actually back allocations
        unless it happens to be the one installed as the global allocator.
        If you need a custom memory_limit to actually govern allocation,
        create that GPUBatchFFT first, before any other instance in the
        process (including the shared default one used by gpu_fft/gpu_rfft).
        """
        self.memory_limit = memory_limit if memory_limit is not None else GPUConfig.default_vram_limit()
        self.plan_cache_size = plan_cache_size or GPUConfig.PLAN_CACHE_SIZE
        self.cp: Any = None
        self._available: bool = False
        self._mempool: Any = None

        self._init_gpu()

    def _init_gpu(self) -> None:
        """Initialize CuPy and configure settings."""
        global _allocator_installed
        try:
            import cupy as cp
            self.cp = cp

            # Configure plan cache (API varies by CuPy version)
            try:
                plan_cache = cp.fft.config.get_plan_cache()
                # Try newer API first
                if hasattr(plan_cache, 'set_size'):
                    plan_cache.set_size(self.plan_cache_size)
                elif hasattr(plan_cache, 'max_size'):
                    plan_cache.max_size = self.plan_cache_size
                # If neither exists, skip plan cache configuration
            except (AttributeError, TypeError):
                pass  # Plan cache config not available in this CuPy version

            # Configure memory pool - store as instance attribute for later access
            self._mempool = cp.cuda.MemoryPool()
            self._mempool.set_limit(size=int(self.memory_limit))

            # Only the first GPUBatchFFT in the process installs itself as
            # CuPy's global allocator; see the __init__ docstring for why.
            if not _allocator_installed:
                cp.cuda.set_allocator(self._mempool.malloc)
                _allocator_installed = True

            self._available = True
        except ImportError:
            self.cp = None
            self._available = False

    @property
    def available(self) -> bool:
        """Check if GPU is available."""
        return self._available

    def to_gpu(self, data: ArrayLike) -> Any:
        """Transfer numpy array to GPU.

        Args:
            data: numpy array

        Returns:
            CuPy array on GPU
        """
        if not self._available:
            raise RuntimeError("GPU not available")
        return self.cp.asarray(data)

    def to_cpu(self, data: Any) -> ArrayResult:
        """Transfer GPU array back to CPU.

        Args:
            data: CuPy array

        Returns:
            numpy array
        """
        if not self._available:
            raise RuntimeError("GPU not available")
        result: ArrayResult = self.cp.asnumpy(data)
        return result

    def fft_batch(self, data: ArrayLike, axis: int = -1, fallback: bool = True) -> ArrayResult:
        """Compute FFT on batch of signals (auto-transfer).

        Args:
            data: numpy array [batch, signal_length] or similar
            axis: Axis along which to compute FFT
            fallback: If True (default) and no GPU is available, silently
                degrade to scipy.fft.fft on CPU (still emits a
                RuntimeWarning so the fallback is visible, not silent). If
                False, raise RuntimeError instead of running on CPU. Default
                is True to keep this a drop-in convenience method; set
                False when you need a hard guarantee the computation ran on
                GPU.

        Returns:
            numpy array with FFT results
        """
        if not self._available:
            if not fallback:
                raise RuntimeError("GPU not available and fallback=False")
            import warnings
            warnings.warn(
                "GPUBatchFFT.fft_batch: no GPU available, falling back to scipy.fft.fft on CPU",
                RuntimeWarning,
                stacklevel=2,
            )
            from scipy.fft import fft
            return fft(np.asarray(data), axis=axis)

        gpu_data = self.to_gpu(data)
        gpu_result = self.cp.fft.fft(gpu_data, axis=axis)
        return self.to_cpu(gpu_result)

    def rfft_batch(self, data: ArrayLike, axis: int = -1, fallback: bool = True) -> ArrayResult:
        """Compute real FFT on batch of real signals (more efficient).

        Args:
            data: numpy array of real values
            axis: Axis along which to compute FFT
            fallback: If True (default) and no GPU is available, silently
                degrade to scipy.fft.rfft on CPU (still emits a
                RuntimeWarning so the fallback is visible, not silent). If
                False, raise RuntimeError instead of running on CPU.

        Returns:
            numpy array with one-sided FFT results
        """
        if not self._available:
            if not fallback:
                raise RuntimeError("GPU not available and fallback=False")
            import warnings
            warnings.warn(
                "GPUBatchFFT.rfft_batch: no GPU available, falling back to scipy.fft.rfft on CPU",
                RuntimeWarning,
                stacklevel=2,
            )
            from scipy.fft import rfft
            return rfft(np.asarray(data), axis=axis)

        gpu_data = self.to_gpu(data)
        gpu_result = self.cp.fft.rfft(gpu_data, axis=axis)
        return self.to_cpu(gpu_result)

    def fft_gpu_resident(self, gpu_data: Any, axis: int = -1) -> Any:
        """Compute FFT on data already on GPU (no transfer).

        Args:
            gpu_data: CuPy array (must already be on GPU)
            axis: Axis along which to compute FFT

        Returns:
            CuPy array (stays on GPU)
        """
        if not self._available:
            raise RuntimeError("GPU not available")
        return self.cp.fft.fft(gpu_data, axis=axis)

    def ifft_gpu_resident(self, gpu_data: Any, axis: int = -1) -> Any:
        """Compute inverse FFT on data already on GPU.

        Args:
            gpu_data: CuPy array (must already be on GPU)
            axis: Axis along which to compute IFFT

        Returns:
            CuPy array (stays on GPU)
        """
        if not self._available:
            raise RuntimeError("GPU not available")
        return self.cp.fft.ifft(gpu_data, axis=axis)

    def memory_info(self) -> dict[str, Any]:
        """Get current GPU memory usage."""
        if not self._available:
            return {'available': False}

        return {
            'used_bytes': self._mempool.used_bytes(),
            'total_bytes': self._mempool.total_bytes(),
            'limit_bytes': int(self.memory_limit),
        }

    def clear_cache(self) -> None:
        """Clear cuFFT plan cache and free unused memory."""
        if not self._available:
            return

        # Clear plan cache (API varies by CuPy version)
        try:
            plan_cache = self.cp.fft.config.get_plan_cache()
            plan_cache.clear()
        except (AttributeError, TypeError):
            pass  # Plan cache API not available in this CuPy version

        # Free unused memory from our pool
        self._mempool.free_all_blocks()


# =============================================================================
# Convenience Functions
# =============================================================================

# Module-level cached default GPUBatchFFT, lazily created on first use by
# gpu_fft()/gpu_rfft(). Reusing it (instead of constructing a fresh
# GPUBatchFFT per call) avoids repeatedly resetting CuPy's global allocator
# and orphaning previously allocated memory pool blocks (see the
# GPUBatchFFT.__init__ docstring for the allocator-install semantics).
_default_processor: GPUBatchFFT | None = None


def _get_default_processor() -> GPUBatchFFT:
    """Return the process-wide cached GPUBatchFFT used by gpu_fft/gpu_rfft."""
    global _default_processor
    if _default_processor is None:
        _default_processor = GPUBatchFFT()
    return _default_processor


def gpu_fft(data: ArrayLike, axis: int = -1) -> ArrayResult:
    """One-shot GPU FFT with auto-transfer.

    Uses a cached, lazily-created module-level GPUBatchFFT (shared across
    calls) rather than constructing a new one every call, so repeated calls
    do not thrash or leak CuPy's memory pool. If you need a processor with
    custom memory_limit/plan_cache_size, construct your own GPUBatchFFT and
    call fft_batch() directly instead of using this convenience function.

    Args:
        data: numpy array
        axis: Axis along which to compute FFT

    Returns:
        numpy array with FFT result
    """
    return _get_default_processor().fft_batch(data, axis=axis)


def gpu_rfft(data: ArrayLike, axis: int = -1) -> ArrayResult:
    """One-shot GPU real FFT with auto-transfer.

    Uses the same cached module-level GPUBatchFFT as gpu_fft(); see its
    docstring for the rationale.

    Args:
        data: numpy array of real values
        axis: Axis along which to compute FFT

    Returns:
        numpy array with one-sided FFT result
    """
    return _get_default_processor().rfft_batch(data, axis=axis)


# =============================================================================
# Benchmarking Utility
# =============================================================================

def benchmark_cpu_vs_gpu(
    sizes: list[int] | None = None, batch_sizes: list[int] | None = None, iterations: int = 10
) -> dict[str, dict[str, float | str]]:
    """Compare CPU vs GPU FFT performance.

    Args:
        sizes: List of FFT sizes to test. Default: [1024, 4096, 16384, 65536]
        batch_sizes: List of batch sizes. Default: [1, 16, 64, 128]
        iterations: Number of iterations per test

    Returns:
        dict with timing results
    """
    import time

    from scipy.fft import fft as scipy_fft

    if sizes is None:
        sizes = [1024, 4096, 16384, 65536]
    if batch_sizes is None:
        batch_sizes = [1, 16, 64, 128]

    results: dict[str, dict[str, float | str]] = {}
    processor = GPUBatchFFT()

    for size in sizes:
        for batch in batch_sizes:
            key = f'{size}x{batch}'
            data = np.random.randn(batch, size) + 1j * np.random.randn(batch, size)

            # CPU timing
            start = time.perf_counter()
            for _ in range(iterations):
                scipy_fft(data, axis=1)
            cpu_time = (time.perf_counter() - start) / iterations * 1000

            # GPU timing (if available)
            if processor.available:
                # Warmup
                try:
                    processor.fft_batch(data, axis=1)
                except Exception:
                    results[key] = {'cpu_ms': cpu_time, 'gpu_ms': 'N/A', 'speedup': 'N/A'}
                    continue

                start = time.perf_counter()
                for _ in range(iterations):
                    processor.fft_batch(data, axis=1)
                gpu_time = (time.perf_counter() - start) / iterations * 1000

                speedup = cpu_time / gpu_time if gpu_time > 0 else 0
                results[key] = {
                    'cpu_ms': round(cpu_time, 3),
                    'gpu_ms': round(gpu_time, 3),
                    'speedup': round(speedup, 2),
                }
            else:
                results[key] = {
                    'cpu_ms': round(cpu_time, 3),
                    'gpu_ms': 'N/A',
                    'speedup': 'N/A',
                }

    return results


if __name__ == '__main__':
    print('=== GPU FFT Utilities ===\n')

    # Check GPU
    print('GPU Info:')
    info = get_gpu_info()
    for k, v in info.items():
        print(f'  {k}: {v}')

    print('\n=== Quick Benchmark ===')
    if gpu_available():
        results = benchmark_cpu_vs_gpu(
            sizes=[4096, 16384, 65536],
            batch_sizes=[1, 64],
            iterations=10
        )
        print(f'{"Config":>15s} {"CPU (ms)":>10s} {"GPU (ms)":>10s} {"Speedup":>10s}')
        print('-' * 50)
        for key, vals in results.items():
            print(f'{key:>15s} {vals["cpu_ms"]:>10} {vals["gpu_ms"]:>10} {vals["speedup"]:>10}')
    else:
        print('GPU not available for benchmarking')
