"""Test FFT backend registration and detection."""

import sys

import numpy as np
import pytest
from conftest import all_backends_param

import fftkit
import fftkit.config


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

    def test_no_env_vars_mkl_present_returns_mkl(self, monkeypatch):
        """With no env override and mkl_fft importable, detect_backend must
        pick 'mkl' (step 3 of the documented precedence).

        mkl_fft is injected rather than assumed. An earlier version of this
        test asserted the real import and passed only on machines where MKL
        happened to be installed; it failed on a clean scipy+numpy
        environment, which is precisely what CI's default job runs. A test
        for a documented precedence rule must not depend on which optional
        package the runner happens to have.
        """
        import sys
        import types

        monkeypatch.delenv("FFTKIT_BACKEND", raising=False)
        monkeypatch.delenv("PYMODAL_FFT_BACKEND", raising=False)
        monkeypatch.setitem(sys.modules, "mkl_fft", types.ModuleType("mkl_fft"))
        assert fftkit.config.detect_backend() == "mkl"

    def test_no_env_vars_mkl_absent_falls_back_to_scipy(self, monkeypatch):
        """When 'import mkl_fft' fails (simulated here since mkl_fft is
        actually installed on this machine) and no env var is set,
        detect_backend must fall through to the always-available 'scipy'
        default rather than raising or returning a stale value.
        """
        monkeypatch.delenv("FFTKIT_BACKEND", raising=False)
        monkeypatch.delenv("PYMODAL_FFT_BACKEND", raising=False)
        # Force `import mkl_fft` to raise ImportError without touching the
        # real installed package: inserting None under its name makes the
        # import machinery raise ImportError for that name specifically.
        monkeypatch.setitem(sys.modules, "mkl_fft", None)
        assert fftkit.config.detect_backend() == "scipy"


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

    def test_mkl_available_true_when_importable(self):
        """mkl_fft is installed on this machine, so mkl_available() must
        report True (the try branch), not just "some bool"."""
        try:
            import mkl_fft  # noqa: F401
        except ImportError:
            pytest.skip("mkl_fft not installed on this machine")
        assert fftkit.mkl_available() is True

    def test_mkl_available_false_when_import_fails(self, monkeypatch):
        """When `import mkl_fft` fails, mkl_available() must return False,
        not raise. Simulated via sys.modules rather than uninstalling the
        real (installed) package.
        """
        monkeypatch.setitem(sys.modules, "mkl_fft", None)
        assert fftkit.mkl_available() is False


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

    def test_gpu_resident_below_breakeven_falls_through_to_mkl(self, monkeypatch):
        """gpu_resident=True with GPU available but BOTH batch_size and
        array_size below their breakevens must fall through the
        gpu_resident branch entirely (neither `if` under it fires) rather
        than short-circuiting to cupy just because gpu_resident was
        requested.
        """
        monkeypatch.setattr(fftkit.backends, "gpu_available", lambda: True)
        monkeypatch.setattr(fftkit.backends, "mkl_available", lambda: True)
        result = fftkit.get_optimal_backend(
            array_size=2048, batch_size=1, prefer_gpu=True, gpu_resident=True
        )
        assert result == 'mkl'

    def test_no_mkl_no_gpu_small_transfer_array_returns_scipy(self, monkeypatch):
        """Non-resident path, GPU available but array_size below the
        256K transfer breakeven, and no MKL: must land on the scipy
        default, not cupy or mkl.
        """
        monkeypatch.setattr(fftkit.backends, "gpu_available", lambda: True)
        monkeypatch.setattr(fftkit.backends, "mkl_available", lambda: False)
        result = fftkit.get_optimal_backend(
            array_size=2048, batch_size=1, prefer_gpu=True, gpu_resident=False
        )
        assert result == 'scipy'


class TestBenchmarkBackendsErrorHandling:
    """benchmark_backends() must record a per-backend failure as a string,
    not let one broken backend abort the whole sweep.
    """

    def test_broken_backend_records_error_string_others_still_time(self, monkeypatch):
        def _boom(*args, **kwargs):
            raise RuntimeError("synthetic failure")

        broken = fftkit.backends.Backend("scipy", {"fft": _boom})
        monkeypatch.setitem(fftkit.backends.BACKENDS, "scipy", broken)
        # Restrict to backends guaranteed present so the assertion below is
        # deterministic regardless of what else is installed on this machine.
        monkeypatch.setattr(
            fftkit.backends, "get_available_backends", lambda: ["scipy", "numpy"]
        )
        results = fftkit.benchmark_backends(size=64, iterations=2)
        assert results["scipy"] == "Error: synthetic failure"
        assert isinstance(results["numpy"], float) and results["numpy"] > 0


class TestBackendMethods:
    """Backend.supports/.get/.__call__ error-path contracts, exercised
    directly on a minimal Backend rather than through a real module."""

    def test_supports_true_for_registered_transform(self):
        b = fftkit.backends.Backend("dummy", {"fft": lambda x: x})
        assert b.supports("fft") is True

    def test_supports_false_for_unregistered_transform(self):
        b = fftkit.backends.Backend("dummy", {"fft": lambda x: x})
        assert b.supports("ifft") is False

    def test_get_missing_transform_raises_named_not_implemented(self):
        b = fftkit.backends.Backend("dummy", {"fft": lambda x: x})
        with pytest.raises(NotImplementedError, match="dummy.*ifft"):
            b.get("ifft")

    def test_call_dispatches_to_registered_transform(self):
        b = fftkit.backends.Backend("dummy", {"fft": lambda x: x * 2})
        assert b("fft", 21) == 42

    def test_call_missing_transform_raises(self):
        b = fftkit.backends.Backend("dummy", {"fft": lambda x: x})
        with pytest.raises(NotImplementedError):
            b("ifft", 1)


class TestGetAvailableBackendsCache:
    """get_available_backends() caches at module level; refresh=True must
    re-probe instead of returning the stale cached list.
    """

    def test_refresh_true_reprobes_and_reflects_registry_change(self, monkeypatch):
        # Prime the cache with the real registry first.
        first = fftkit.get_available_backends()
        assert "scipy" in first

        # Swap in a broken 'scipy' entry: without refresh, the cache must
        # still report the old (cached) result.
        def _boom(*args, **kwargs):
            raise RuntimeError("synthetic failure")

        monkeypatch.setitem(
            fftkit.backends.BACKENDS, "scipy", fftkit.backends.Backend("scipy", {"fft": _boom})
        )
        cached = fftkit.get_available_backends()
        assert "scipy" in cached, "cached call must not re-probe"

        refreshed = fftkit.get_available_backends(refresh=True)
        assert "scipy" not in refreshed, "refresh=True must re-probe and drop the now-broken backend"

        # Restore the cache to the real registry state so later tests in
        # this process see a correct, working 'scipy' entry again.
        monkeypatch.undo()
        fftkit.get_available_backends(refresh=True)


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
        """Without mkl_fft installed, must return False (and warn), not raise."""
        try:
            import mkl_fft  # noqa: F401
            pytest.skip("mkl_fft is installed on this machine")
        except ImportError:
            pass
        # Warns as well as returning False: a bare False is not actionable.
        with pytest.warns(fftkit.MklBackendWarning, match="fftkit\\[mkl\\]"):
            assert fftkit.register_mkl_scipy_backend() is False

    def test_mkl_present_but_scipy_interface_broken_still_returns_false(self):
        """On this machine mkl_fft is installed but its scipy.fft interface
        additionally imports a separate ``mkl`` package that is NOT
        installed here, so ``import mkl_fft.interfaces.scipy_fft`` itself
        raises ModuleNotFoundError (a subclass of ImportError). This is a
        genuinely different failure point than "mkl_fft absent" (which the
        sibling test above covers, and which SKIPS on this machine since
        mkl_fft *is* importable) -- both still land on the honest `return
        False`, which is the contract under test.

        The return value alone is not the whole contract: because
        ``mkl_available()`` is True and ``get_fft_func('mkl')`` works in this
        state, a bare False is actively misleading. So the warning must name
        the module that failed AND the fix, and the two failure modes must not
        produce the same message.
        """
        try:
            import mkl_fft  # noqa: F401
        except ImportError:
            pytest.skip("mkl_fft not installed on this machine")
        try:
            import mkl_fft.interfaces.scipy_fft  # noqa: F401
            pytest.skip("mkl_fft.interfaces.scipy_fft is fully importable on this machine")
        except ImportError:
            pass

        with pytest.warns(fftkit.MklBackendWarning) as record:
            assert fftkit.register_mkl_scipy_backend() is False

        message = str(record[0].message)
        # Names the actual missing module, not a generic failure.
        assert "'mkl'" in message
        # Points at the right fix. The wrong fix here would be reinstalling
        # mkl_fft, which is already present and working.
        assert "pip install mkl" in message
        # And must not tell the user to install what they already have.
        assert 'fftkit[mkl]' not in message

    def test_missing_mkl_fft_gives_a_different_message_than_missing_mkl(self, monkeypatch):
        """The two failure modes need different remedies, so they must not
        share a message. Simulates mkl_fft being absent entirely."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("mkl_fft"):
                raise ImportError("No module named 'mkl_fft'", name="mkl_fft")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.warns(fftkit.MklBackendWarning) as record:
            assert fftkit.register_mkl_scipy_backend() is False

        message = str(record[0].message)
        assert "mkl_fft" in message
        assert 'fftkit[mkl]' in message, "absent mkl_fft should point at the fftkit extra"

    def test_returns_true_when_scipy_interface_available(self, monkeypatch):
        """Success path: when ``mkl_fft.interfaces.scipy_fft`` imports
        cleanly, register_mkl_scipy_backend() must return True.

        Stubs both mkl_fft.interfaces.scipy_fft and
        scipy.fft.set_global_backend so this is deterministic regardless of
        whether the real `mkl` package happens to be installed, and so it
        never performs the real (process-wide, one-way) global backend
        mutation that set_global_backend's own docstring warns about
        ("permanent use" / "overwrite the previously set global backend").
        """
        import types

        fake_mkl_fft = types.ModuleType("mkl_fft")
        fake_interfaces = types.ModuleType("mkl_fft.interfaces")
        fake_scipy_fft_iface = types.ModuleType("mkl_fft.interfaces.scipy_fft")
        fake_mkl_fft.interfaces = fake_interfaces
        fake_interfaces.scipy_fft = fake_scipy_fft_iface

        monkeypatch.setitem(sys.modules, "mkl_fft", fake_mkl_fft)
        monkeypatch.setitem(sys.modules, "mkl_fft.interfaces", fake_interfaces)
        monkeypatch.setitem(sys.modules, "mkl_fft.interfaces.scipy_fft", fake_scipy_fft_iface)

        calls = []
        import scipy.fft as scipy_fft_module
        monkeypatch.setattr(scipy_fft_module, "set_global_backend", lambda backend: calls.append(backend))

        assert fftkit.register_mkl_scipy_backend() is True
        assert calls == [fake_scipy_fft_iface]


class TestBackwardsCompatShims:
    """The 0.1.0-era standalone shim functions (scipy_fft/numpy_fft/etc.)
    still delegate to BACKENDS[...] correctly.
    """

    def test_scipy_fft_shim_matches_numpy_reference(self):
        x = np.array([1, 2, 3, 4], dtype=np.complex128)
        result = fftkit.scipy_fft(x)
        assert np.allclose(result, np.fft.fft(x), rtol=1e-12, atol=1e-14)

    def test_numpy_fft_shim_matches_numpy_reference(self):
        x = np.array([1, 2, 3, 4], dtype=np.complex128)
        result = fftkit.numpy_fft(x)
        assert np.allclose(result, np.fft.fft(x), rtol=1e-12, atol=1e-14)

    def test_accelerate_fft_shim_off_macos_raises(self):
        if sys.platform == 'darwin':
            pytest.skip("Accelerate should work on macOS")
        x = np.array([1, 2, 3, 4], dtype=np.complex128)
        with pytest.raises(NotImplementedError, match="macOS"):
            fftkit.accelerate_fft(x)

    def test_mkl_fft_transform_shim_when_available(self):
        try:
            import mkl_fft  # noqa: F401
        except ImportError:
            pytest.skip("mkl_fft not installed on this machine")
        x = np.array([1, 2, 3, 4], dtype=np.complex128)
        result = fftkit.mkl_fft_transform(x)
        assert np.allclose(result, np.fft.fft(x), rtol=1e-10, atol=1e-12)

    def test_mkl_fft_transform_shim_wraps_missing_module(self, monkeypatch):
        """When mkl_fft is not importable, the shim must translate
        ModuleNotFoundError into an ImportError carrying install
        instructions, not leak the raw exception.
        """
        def _raise_module_not_found(*args, **kwargs):
            raise ModuleNotFoundError("No module named 'mkl_fft'")

        broken = fftkit.backends.Backend("mkl", {"fft": _raise_module_not_found})
        monkeypatch.setitem(fftkit.backends.BACKENDS, "mkl", broken)
        x = np.array([1, 2, 3, 4], dtype=np.complex128)
        with pytest.raises(ImportError, match="mkl_fft not installed"):
            fftkit.mkl_fft_transform(x)

    def test_cupy_fft_shim_wraps_missing_module(self):
        """cupy is not installed in this environment, so the shim's
        ModuleNotFoundError -> ImportError translation is genuinely
        exercised here, not simulated.
        """
        try:
            import cupy  # noqa: F401
            pytest.skip("cupy is installed on this machine")
        except ImportError:
            pass
        x = np.array([1, 2, 3, 4], dtype=np.complex128)
        with pytest.raises(ImportError, match="CuPy not installed"):
            fftkit.cupy_fft(x)


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
