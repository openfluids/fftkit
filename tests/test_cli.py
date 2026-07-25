"""Test the fftkit CLI (python -m fftkit / fftkit.__main__.main)."""

import json

import pytest

from fftkit.__main__ import main


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
