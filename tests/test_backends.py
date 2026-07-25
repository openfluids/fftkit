"""Test FFT backend registration and detection."""

import sys

import numpy as np
import pytest
from conftest import all_backends_param

import fftkit


class TestBackendRegistry:
    """Test the FFT backend registry."""

    def test_backend_names_registered(self):
        """Verify expected backends are registered."""
        names = fftkit.get_backend_names()
        assert 'numpy' in names, "numpy backend should be registered"
        assert 'scipy' in names, "scipy backend should be registered"
        assert 'accelerate' in names, "accelerate backend should be registered"
        assert 'mkl' in names, "mkl backend should be registered"

    def test_available_backends_subset_of_names(self):
        """Verify available backends are subset of registered names."""
        available = fftkit.get_available_backends()
        names = fftkit.get_backend_names()
        assert set(available).issubset(set(names)), \
            f"Available {available} should be subset of {names}"

    def test_available_backends_not_empty(self):
        """At least one backend should be available."""
        available = fftkit.get_available_backends()
        assert len(available) > 0, "At least one backend must be available"
        # numpy should always be available
        assert 'numpy' in available, "numpy should always be available"


class TestGetFFTFunc:
    """Test get_fft_func behavior."""

    def test_get_fft_func_no_argument(self):
        """get_fft_func() uses DEFAULT_BACKEND and returns callable."""
        func = fftkit.get_fft_func()
        assert callable(func), "Should return a callable"

        # Verify it works
        x = np.array([1, 2, 3, 4], dtype=np.complex128)
        result = func(x)
        assert result is not None and len(result) == 4

    def test_get_fft_func_with_default_backend(self):
        """get_fft_func with DEFAULT_BACKEND name works."""
        func = fftkit.get_fft_func(fftkit.DEFAULT_BACKEND)
        assert callable(func)

    def test_get_fft_func_nonexistent_backend(self):
        """Requesting nonexistent backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown.*backend"):
            fftkit.get_fft_func('nonexistent-backend-xyz')

    @pytest.mark.parametrize("backend", all_backends_param())
    def test_get_fft_func_available_backends(self, backend):
        """All available backends should return working functions."""
        func = fftkit.get_fft_func(backend)
        x = np.array([1, 2, 3, 4], dtype=np.complex128)
        result = func(x)
        assert result is not None and len(result) == 4


class TestUnavailableBackends:
    """Test that unavailable backends raise when called."""

    def test_accelerate_off_macos(self):
        """Accelerate should fail off macOS."""
        if sys.platform == 'darwin':
            pytest.skip("Accelerate should work on macOS")

        func = fftkit.get_fft_func('accelerate')
        x = np.array([1, 2, 3, 4], dtype=np.complex128)

        with pytest.raises((NotImplementedError, RuntimeError)):
            func(x)

    def test_mkl_when_unavailable(self):
        """MKL should fail when not installed."""
        try:
            import mkl_fft  # noqa: F401
            pytest.skip("MKL is installed, skipping unavailable test")
        except ImportError:
            pass

        func = fftkit.get_fft_func('mkl')
        x = np.array([1, 2, 3, 4], dtype=np.complex128)

        with pytest.raises((ImportError, Exception)):
            func(x)


class TestDetectBackend:
    """Test backend auto-detection."""

    def test_detect_backend_returns_string(self):
        """detect_backend should return a string."""
        backend = fftkit.detect_backend()
        assert isinstance(backend, str)
        assert len(backend) > 0

    def test_detect_backend_lowercase(self):
        """detect_backend returns lowercase name."""
        backend = fftkit.detect_backend()
        assert backend == backend.lower(), f"Backend name should be lowercase: {backend}"

    def test_fftkit_backend_env_var_precedence(self, monkeypatch):
        """FFTKIT_BACKEND takes precedence over PYMODAL_FFT_BACKEND."""
        # Need to re-import config module to pick up new env vars
        monkeypatch.setenv("FFTKIT_BACKEND", "NUMPY")
        monkeypatch.setenv("PYMODAL_FFT_BACKEND", "SCIPY")

        # Import fresh config
        import importlib

        import fftkit.config
        importlib.reload(fftkit.config)

        assert fftkit.config.detect_backend() == "numpy"

    def test_pymodal_backend_env_var_fallback(self, monkeypatch):
        """PYMODAL_FFT_BACKEND used if FFTKIT_BACKEND not set."""
        monkeypatch.delenv("FFTKIT_BACKEND", raising=False)
        monkeypatch.setenv("PYMODAL_FFT_BACKEND", "SCIPY")

        import importlib

        import fftkit.config
        importlib.reload(fftkit.config)

        assert fftkit.config.detect_backend() == "scipy"

    def test_env_var_lowercasing(self, monkeypatch):
        """Environment variables are lowercased."""
        monkeypatch.setenv("FFTKIT_BACKEND", "NUMPY")
        monkeypatch.delenv("PYMODAL_FFT_BACKEND", raising=False)

        import importlib

        import fftkit.config
        importlib.reload(fftkit.config)

        assert fftkit.config.detect_backend() == "numpy"


class TestBackendProperties:
    """Test properties of available backends."""

    def test_gpu_available_callable(self):
        """gpu_available() should be callable and return bool."""
        result = fftkit.gpu_available()
        assert isinstance(result, (bool, np.bool_))

    def test_mkl_available_callable(self):
        """mkl_available() should be callable and return bool."""
        result = fftkit.mkl_available()
        assert isinstance(result, (bool, np.bool_))


class TestGetOptimalBackend:
    """Test get_optimal_backend's workload-based selection heuristic."""

    def test_small_array_no_mkl_no_gpu_returns_scipy(self, monkeypatch):
        """Below the MKL size threshold, and with no GPU, must fall through
        to the scipy default -- this is the only backend guaranteed present.
        """
        monkeypatch.setattr(fftkit.backends, "mkl_available", lambda: False)
        monkeypatch.setattr(fftkit.backends, "gpu_available", lambda: False)
        result = fftkit.get_optimal_backend(array_size=512, prefer_gpu=True)
        assert result == 'scipy'

    def test_large_array_with_mkl_returns_mkl(self, monkeypatch):
        """array_size >= 1024 with MKL installed and no GPU should pick mkl."""
        monkeypatch.setattr(fftkit.backends, "mkl_available", lambda: True)
        monkeypatch.setattr(fftkit.backends, "gpu_available", lambda: False)
        result = fftkit.get_optimal_backend(array_size=2048, prefer_gpu=True)
        assert result == 'mkl'

    def test_huge_transfer_array_with_gpu_returns_cupy(self, monkeypatch):
        """array_size >= 256K with GPU available (non gpu_resident) should
        prefer cupy despite transfer overhead, per the documented threshold.
        """
        monkeypatch.setattr(fftkit.backends, "gpu_available", lambda: True)
        result = fftkit.get_optimal_backend(array_size=300_000, prefer_gpu=True, gpu_resident=False)
        assert result == 'cupy'

    def test_gpu_resident_large_batch_returns_cupy(self, monkeypatch):
        """gpu_resident=True with batch_size >= BATCH_BREAKEVEN_RESIDENT (16)
        should pick cupy even for a small per-FFT array_size.
        """
        monkeypatch.setattr(fftkit.backends, "gpu_available", lambda: True)
        result = fftkit.get_optimal_backend(array_size=256, batch_size=32, prefer_gpu=True, gpu_resident=True)
        assert result == 'cupy'

    def test_prefer_gpu_false_never_returns_cupy(self, monkeypatch):
        """prefer_gpu=False must never select a GPU backend, regardless of size."""
        monkeypatch.setattr(fftkit.backends, "gpu_available", lambda: True)
        monkeypatch.setattr(fftkit.backends, "mkl_available", lambda: False)
        result = fftkit.get_optimal_backend(array_size=300_000, prefer_gpu=False)
        assert result != 'cupy'


class TestBenchmarkBackends:
    """Test benchmark_backends() over the actually-available backends."""

    def test_returns_dict_keyed_by_available_backends(self):
        # Keep iterations tiny: this measures the *shape* of the result,
        # not performance, so runtime should stay well under the suite's
        # fast-suite budget.
        results = fftkit.benchmark_backends(size=64, iterations=2)
        assert isinstance(results, dict)
        assert set(results.keys()) == set(fftkit.get_available_backends())

    def test_timings_are_positive_floats(self):
        results = fftkit.benchmark_backends(size=64, iterations=2)
        for name, val in results.items():
            # benchmark_backends() records a string "Error: ..." per-backend
            # on failure instead of raising; on this machine (scipy+numpy
            # only, both always working) every entry must be a positive
            # float, not an error string.
            assert isinstance(val, float), f"{name}: expected float timing, got {val!r}"
            assert val > 0


class TestRegisterMklScipyBackend:
    """Test register_mkl_scipy_backend's honest success/failure return."""

    def test_returns_false_when_mkl_unavailable(self):
        """Without mkl_fft installed, must return False, not raise."""
        try:
            import mkl_fft  # noqa: F401
            pytest.skip("mkl_fft is installed on this machine")
        except ImportError:
            pass
        assert fftkit.register_mkl_scipy_backend() is False


class TestSuiteNotSilentlyEmpty:
    """Guard against the collection-time parametrize trap this suite used to
    have: @pytest.mark.parametrize("backend", fftkit.get_available_backends())
    silently drops to zero test cases for any backend not importable on the
    machine running pytest, with no skip, no failure, nothing in the report.
    Pin down that scipy and numpy -- always-available backends -- are
    actually present and actually exercised, so a future regression back to
    that pattern (which would still pass this exact assertion by accident
    only if scipy/numpy happened to be probed) has at least one contract to
    violate.
    """

    def test_scipy_and_numpy_always_available_and_exercised(self):
        available = fftkit.get_available_backends()
        assert "scipy" in available, "scipy must always be available"
        assert "numpy" in available, "numpy must always be available"

        # "exercised" = the registered callable actually runs and produces
        # a correctly-shaped result, not merely that the name is listed.
        x = np.array([1, 2, 3, 4], dtype=np.complex128)
        for name in ("scipy", "numpy"):
            result = fftkit.get_fft_func(name)(x)
            assert len(result) == len(x), f"{name} backend did not run"

    def test_all_backends_param_covers_full_registry(self):
        """all_backends_param() must enumerate every registered backend name,
        not just the available ones -- this is the whole point of the fix.
        """
        from conftest import all_backends_param

        param_ids = {p.id for p in all_backends_param()}
        assert param_ids == set(fftkit.get_backend_names())
