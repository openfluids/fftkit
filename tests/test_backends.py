"""Test FFT backend registration and detection."""

import sys

import numpy as np
import pytest

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

    @pytest.mark.parametrize("backend", fftkit.get_available_backends())
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
