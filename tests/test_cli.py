"""Test the fftkit CLI (python -m fftkit / fftkit.__main__.main)."""

import json

import pytest

import fftkit.__main__ as cli_main
from fftkit.__main__ import main
from fftkit.backends import BACKENDS, Backend


class TestCLIBasics:
    """Exit codes and dispatch."""

    def test_no_subcommand_returns_nonzero(self, capsys):
        """Running with no subcommand must fail loudly (non-zero exit),
        not silently succeed with no output.
        """
        result = main([])
        assert result != 0
        captured = capsys.readouterr()
        assert "usage" in captured.out.lower()

    def test_info_returns_zero(self, capsys):
        result = main(["info"])
        assert result == 0
        captured = capsys.readouterr()
        assert "fftkit info" in captured.out

    def test_bench_returns_zero(self, capsys):
        # Small size/iters to keep this fast; only exercises the code path.
        result = main(["bench", "--size", "1024", "--iters", "2"])
        assert result == 0
        captured = capsys.readouterr()
        assert "fftkit bench" in captured.out

    def test_unknown_subcommand_raises_systemexit(self):
        """argparse itself rejects unregistered subcommands via SystemExit
        (its documented behavior), not a return code.
        """
        with pytest.raises(SystemExit):
            main(["not-a-real-command"])


class TestCLIJSONOutput:
    """--json output must be valid, parseable JSON with the expected shape."""

    def test_bench_json_parses(self, capsys):
        result = main(["bench", "--size", "1024", "--iters", "2", "--json"])
        assert result == 0
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload["size"] == 1024
        assert payload["iterations"] == 2
        assert "results" in payload
        # scipy/numpy are always available, so the results dict is not empty.
        assert set(payload["results"]).issuperset({"scipy", "numpy"})

    def test_bench_json_result_values_are_numeric_or_error(self, capsys):
        result = main(["bench", "--size", "1024", "--iters", "2", "--json"])
        assert result == 0
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        for name, val in payload["results"].items():
            # Each backend's timing is either a numeric ms/FFT figure or an
            # {"error": "..."} dict -- never null/missing.
            assert isinstance(val, (int, float, dict)), f"{name}: unexpected type {type(val)}"
            if isinstance(val, dict):
                assert "error" in val

    def test_info_output_is_not_json_by_default(self, capsys):
        """info has no --json flag; sanity check it stays human-readable."""
        result = main(["info"])
        assert result == 0
        captured = capsys.readouterr()
        with pytest.raises(json.JSONDecodeError):
            json.loads(captured.out)


class TestInfoEnvVarSelection:
    """cmd_info's "(selected via $VAR)" annotation, which only fires when
    FFTKIT_BACKEND or PYMODAL_FFT_BACKEND is actually set.
    """

    def test_fftkit_backend_env_var_shown(self, capsys, monkeypatch):
        monkeypatch.setenv("FFTKIT_BACKEND", "scipy")
        monkeypatch.delenv("PYMODAL_FFT_BACKEND", raising=False)
        result = main(["info"])
        assert result == 0
        captured = capsys.readouterr()
        assert "(selected via $FFTKIT_BACKEND)" in captured.out

    def test_pymodal_backend_env_var_shown(self, capsys, monkeypatch):
        monkeypatch.delenv("FFTKIT_BACKEND", raising=False)
        monkeypatch.setenv("PYMODAL_FFT_BACKEND", "scipy")
        result = main(["info"])
        assert result == 0
        captured = capsys.readouterr()
        assert "(selected via $PYMODAL_FFT_BACKEND)" in captured.out

    def test_no_env_var_shows_auto_detected(self, capsys, monkeypatch):
        monkeypatch.delenv("FFTKIT_BACKEND", raising=False)
        monkeypatch.delenv("PYMODAL_FFT_BACKEND", raising=False)
        result = main(["info"])
        assert result == 0
        captured = capsys.readouterr()
        assert "(auto-detected)" in captured.out


class TestInfoGPUSection:
    """cmd_info prints a different block depending on GPU availability;
    monkeypatch get_gpu_info() to reach both without real hardware.
    """

    def test_gpu_available_prints_device_details(self, capsys, monkeypatch):
        fake_info = {
            "available": True,
            "device_id": 0,
            "compute_capability": "8.6",
            "memory_total_gb": 24.0,
            "memory_free_gb": 12.5,
        }
        monkeypatch.setattr(cli_main, "get_gpu_info", lambda: fake_info)
        result = main(["info"])
        assert result == 0
        captured = capsys.readouterr()
        assert "available            : yes" in captured.out
        assert "device_id            : 0" in captured.out
        assert "compute_capability   : 8.6" in captured.out
        assert "memory_total_gb      : 24.00" in captured.out
        assert "memory_free_gb       : 12.50" in captured.out

    def test_gpu_unavailable_prints_no_and_reason(self, capsys, monkeypatch):
        monkeypatch.setattr(
            cli_main, "get_gpu_info", lambda: {"available": False, "error": "no CUDA device"}
        )
        result = main(["info"])
        assert result == 0
        captured = capsys.readouterr()
        assert "available            : no (no CUDA device)" in captured.out

    def test_gpu_unavailable_no_error_key_falls_back_to_unknown(self, capsys, monkeypatch):
        """get_gpu_info() is documented to include 'error' on failure, but
        the CLI's .get('error', 'unknown') default must still not crash if
        it were ever missing."""
        monkeypatch.setattr(cli_main, "get_gpu_info", lambda: {"available": False})
        result = main(["info"])
        assert result == 0
        captured = capsys.readouterr()
        assert "available            : no (unknown)" in captured.out


class TestBenchErrorPath:
    """_run_backend_bench must record a per-backend exception instead of
    aborting the whole `bench` run.
    """

    def test_broken_backend_recorded_as_error_others_still_timed(self, capsys, monkeypatch):
        def _boom(*args, **kwargs):
            raise RuntimeError("synthetic failure")

        monkeypatch.setitem(BACKENDS, "scipy", Backend("scipy", {"fft": _boom}))
        monkeypatch.setattr(cli_main, "get_available_backends", lambda: ["scipy", "numpy"])

        result = main(["bench", "--size", "64", "--iters", "2"])
        assert result == 0
        captured = capsys.readouterr()
        assert "Errors:" in captured.out
        assert "scipy: synthetic failure" in captured.out
        # numpy is unaffected and still produces a normal timing row.
        assert "numpy" in captured.out

    def test_broken_backend_recorded_as_error_in_json(self, capsys, monkeypatch):
        def _boom(*args, **kwargs):
            raise RuntimeError("synthetic failure")

        monkeypatch.setitem(BACKENDS, "scipy", Backend("scipy", {"fft": _boom}))
        monkeypatch.setattr(cli_main, "get_available_backends", lambda: ["scipy", "numpy"])

        result = main(["bench", "--size", "64", "--iters", "2", "--json"])
        assert result == 0
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload["results"]["scipy"] == {"error": "synthetic failure"}
        assert isinstance(payload["results"]["numpy"], float)

    def test_all_backends_broken_prints_no_valid_timing(self, capsys, monkeypatch):
        def _boom(*args, **kwargs):
            raise RuntimeError("synthetic failure")

        monkeypatch.setitem(BACKENDS, "scipy", Backend("scipy", {"fft": _boom}))
        monkeypatch.setitem(BACKENDS, "numpy", Backend("numpy", {"fft": _boom}))
        monkeypatch.setattr(cli_main, "get_available_backends", lambda: ["scipy", "numpy"])

        result = main(["bench", "--size", "64", "--iters", "2"])
        assert result == 0
        captured = capsys.readouterr()
        assert "No backend produced a valid timing." in captured.out
        assert "Errors:" in captured.out


class TestBenchGPUFlag:
    """`bench --gpu` dispatches to benchmark_cpu_vs_gpu; exercised here
    without real hardware since benchmark_cpu_vs_gpu degrades cleanly (GPU
    columns become 'N/A') when no GPU is present -- the actual state of
    this dev box.
    """

    def test_gpu_bench_table_output(self, capsys):
        result = main(["bench", "--gpu", "--size", "64", "--batch", "1", "--iters", "2"])
        assert result == 0
        captured = capsys.readouterr()
        assert "fftkit bench --gpu" in captured.out
        assert "CPU (ms)" in captured.out
        assert "GPU (ms)" in captured.out

    def test_gpu_bench_no_gpu_message_shown(self, capsys, monkeypatch):
        monkeypatch.setattr(
            cli_main, "get_gpu_info", lambda: {"available": False, "error": "no CUDA device"}
        )
        result = main(["bench", "--gpu", "--size", "64", "--batch", "1", "--iters", "2"])
        assert result == 0
        captured = capsys.readouterr()
        assert "No GPU available (no CUDA device); showing CPU timings only." in captured.out

    def test_gpu_bench_json(self, capsys):
        result = main(["bench", "--gpu", "--size", "64", "--batch", "1", "--iters", "2", "--json"])
        assert result == 0
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert isinstance(payload, dict)
        # size x batch is the only combination requested via --size/--batch.
        assert set(payload.keys()) == {"64x1"}
        assert set(payload["64x1"].keys()) == {"cpu_ms", "gpu_ms", "speedup"}

    def test_gpu_bench_json_size_and_batch_together(self, capsys):
        """--size and --batch together restrict the sweep to exactly one
        (size, batch) combination instead of the default cartesian sweep."""
        result = main(["bench", "--gpu", "--size", "128", "--batch", "4", "--iters", "1", "--json"])
        assert result == 0
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert list(payload.keys()) == ["128x4"]


class TestBenchArgumentCombinations:
    def test_batch_without_gpu_is_ignored(self, capsys):
        """--batch is documented as "Ignored without --gpu"; passing it
        alongside a plain (non-gpu) bench must not error or change
        behavior."""
        result = main(["bench", "--size", "64", "--batch", "16", "--iters", "2"])
        assert result == 0
        captured = capsys.readouterr()
        assert "fftkit bench (size=64" in captured.out

    def test_size_default_when_omitted(self, capsys):
        result = main(["bench", "--iters", "2", "--json"])
        assert result == 0
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload["size"] == 65536
