"""Tests for the 0.1.x -> 0.2.0 axis-default migration guard.

fftkit 0.1.x defaulted the 1-D transforms to ``axis=0``; 0.2.0 moved them to
``axis=-1`` to match numpy and scipy. The dangerous property of that change is
that it raises no error: a caller passing a 2-D array gets a different answer
than before, silently. ``AxisDefaultWarning`` exists to make that one case
audible, and these tests pin exactly when it fires.

The rule under test:

    warn  <=>  caller omitted axis  AND  input has ndim > 1  AND  transform is 1-D

Every other combination must stay quiet, because warning on calls that cannot
have changed meaning would train users to filter the warning away.
"""

import warnings

import numpy as np
import pytest

import fftkit

ONE_D_TRANSFORMS = ["fft", "ifft", "rfft", "irfft"]


@pytest.fixture
def x1d():
    rng = np.random.default_rng(11)
    return rng.standard_normal(32)


@pytest.fixture
def x2d():
    rng = np.random.default_rng(12)
    return rng.standard_normal((4, 8))


class TestWarnsOnlyWhenBehaviourChanged:
    """The warning must fire on the ambiguous case and nowhere else."""

    @pytest.mark.parametrize("transform", ONE_D_TRANSFORMS)
    def test_warns_on_multidim_without_explicit_axis(self, transform, x2d):
        with pytest.warns(fftkit.AxisDefaultWarning, match="axis=0 to axis=-1"):
            getattr(fftkit, transform)(x2d)

    @pytest.mark.parametrize("transform", ONE_D_TRANSFORMS)
    def test_silent_on_1d_input(self, transform, x1d):
        """A 1-D signal gives identical results for axis 0 and -1, so there is
        nothing to warn about. This is the overwhelming majority of real calls;
        warning here would make the guard worthless through sheer noise."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", fftkit.AxisDefaultWarning)
            getattr(fftkit, transform)(x1d)

    @pytest.mark.parametrize("transform", ONE_D_TRANSFORMS)
    @pytest.mark.parametrize("axis", [0, 1, -1, -2])
    def test_silent_when_axis_is_explicit(self, transform, axis, x2d):
        """An explicit axis means the caller has already made the choice the
        warning is trying to prompt -- including axis=-1, which must not be
        confused with the default."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", fftkit.AxisDefaultWarning)
            getattr(fftkit, transform)(x2d, axis=axis)

    @pytest.mark.parametrize("transform", ["fft2", "ifft2", "fftn", "ifftn"])
    def test_silent_for_nd_transforms(self, transform):
        """fft2/fftn are new in 0.2.0 and have no 0.1.x default to have changed,
        so they must never warn even on N-D input without axes=."""
        rng = np.random.default_rng(13)
        x = rng.standard_normal((3, 4, 5))
        with warnings.catch_warnings():
            warnings.simplefilter("error", fftkit.AxisDefaultWarning)
            getattr(fftkit, transform)(x)


class TestGetFFTFuncPath:
    """get_fft_func()(x) is the 0.1.x idiom, so it needs the same guard.

    Code written against 0.1.0 most likely calls the registry function directly
    rather than the (new) top-level fftkit.fft, which makes this the path most
    likely to be holding a 2-D array that used to transform along axis 0.
    """

    def test_warns_on_multidim(self, x2d):
        func = fftkit.get_fft_func("scipy")
        with pytest.warns(fftkit.AxisDefaultWarning):
            func(x2d)

    def test_silent_on_1d(self, x1d):
        func = fftkit.get_fft_func("scipy")
        with warnings.catch_warnings():
            warnings.simplefilter("error", fftkit.AxisDefaultWarning)
            func(x1d)

    def test_silent_with_explicit_axis(self, x2d):
        func = fftkit.get_fft_func("scipy")
        with warnings.catch_warnings():
            warnings.simplefilter("error", fftkit.AxisDefaultWarning)
            func(x2d, axis=0)

    def test_explicit_axis_is_honoured(self, x2d):
        """The guard must not quietly override an axis the caller asked for."""
        func = fftkit.get_fft_func("scipy")
        assert np.allclose(func(x2d, axis=0), np.fft.fft(x2d, axis=0))


class TestGuardDoesNotChangeResults:
    """A migration warning that altered numerics would be worse than none."""

    @pytest.mark.parametrize("transform", ONE_D_TRANSFORMS)
    def test_default_equals_explicit_minus_one(self, transform, x2d):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", fftkit.AxisDefaultWarning)
            default = getattr(fftkit, transform)(x2d)
        explicit = getattr(fftkit, transform)(x2d, axis=-1)
        assert np.allclose(default, explicit), (
            f"{transform}: the effective default must still be axis=-1"
        )

    def test_axis_zero_reproduces_old_behaviour(self, x2d):
        """The migration path the warning recommends must actually work: passing
        axis=0 has to give what 0.1.x gave."""
        assert np.allclose(fftkit.fft(x2d, axis=0), np.fft.fft(x2d, axis=0))


class TestWarningIsUsable:
    def test_is_userwarning_subclass(self):
        """DeprecationWarning is hidden by default outside __main__, so a
        library user -- precisely who is affected -- would never see it."""
        assert issubclass(fftkit.AxisDefaultWarning, UserWarning)

    def test_can_be_silenced_by_category(self, x2d):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.filterwarnings("ignore", category=fftkit.AxisDefaultWarning)
            fftkit.fft(x2d)
        assert not [w for w in caught if w.category is fftkit.AxisDefaultWarning]

    def test_message_names_both_directions(self, x2d):
        """A migration warning is only useful if it says how to get either
        behaviour, not merely that something changed."""
        with pytest.warns(fftkit.AxisDefaultWarning) as record:
            fftkit.fft(x2d)
        message = str(record[0].message)
        assert "axis=-1" in message and "axis=0" in message
        assert "0.2.0" in message

    def test_points_at_caller_not_library(self, x2d):
        """stacklevel must blame the user's call site; a warning pointing into
        fftkit's own source tells the reader nothing actionable."""
        with pytest.warns(fftkit.AxisDefaultWarning) as record:
            fftkit.fft(x2d)
        assert record[0].filename == __file__, (
            f"warning blamed {record[0].filename}, expected this test file"
        )
