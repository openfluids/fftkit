"""Tests for fftkit.__version__ resolution.

__version__ is read from installed package metadata rather than hardcoded,
so it can't drift from pyproject.toml.
"""

import importlib
import importlib.metadata

import fftkit


def test_version_reads_installed_package_metadata():
    """The normal case: fftkit is installed (editable) in this venv, so
    __version__ must come from real package metadata, not the
    PackageNotFoundError fallback string.
    """
    assert fftkit.__version__ != "0.0.0+unknown", "fftkit is installed in this venv; version must resolve from metadata"
    # Loose sanity check: a real version string, not garbage.
    assert fftkit.__version__[0].isdigit()


def test_version_falls_back_when_package_metadata_missing(monkeypatch):
    """When importlib.metadata.version() can't find the distribution
    (e.g. running from a source tree with no install record),
    __version__ must fall back to the documented sentinel instead of
    propagating PackageNotFoundError to every `import fftkit`.

    Reloads fftkit's top-level module in place (not a fresh process) so
    the assertion is on the exact code path `import fftkit` runs; the
    reload only re-executes fftkit/__init__.py itself (its submodules stay
    cached in sys.modules and are merely re-bound), and the fixture reloads
    again at teardown to restore the real version for every later test in
    this session.
    """
    def _raise(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _raise)
    try:
        importlib.reload(fftkit)
        assert fftkit.__version__ == "0.0.0+unknown"
    finally:
        # monkeypatch restores importlib.metadata.version at the end of
        # this fixture's teardown, but that happens *after* this function
        # returns; reload now, before undo, then reload once more after
        # undo so both this test and everything downstream see the real
        # version restored under the *real* importlib.metadata.version.
        monkeypatch.undo()
        importlib.reload(fftkit)
        assert fftkit.__version__ != "0.0.0+unknown"
