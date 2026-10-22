"""Full (backend, transform) matrix tests, using scipy.fft as reference.

Covers: fft/ifft, rfft/irfft, fft2/ifft2, fftn/ifftn across every
registered backend (parametrized via conftest.all_backends_param /
partial_backends_param so unavailable-on-this-machine backends show up as
named skips, not silent absences -- see conftest.py for the rationale),
across shapes (power-of-two, non-power-of-two, prime, length-1, 2-D, 3-D),
explicit axis selection, n= padding/truncation, and norm= conventions.
"""

import numpy as np
import pytest
import scipy.fft as spfft
from conftest import all_backends_param, assert_declared_limit, partial_backends_param

import fftkit

# ---------------------------------------------------------------------------
# Tolerance tiers
# ---------------------------------------------------------------------------
# float64/complex128: backends implementing the same algorithm family
# (radix FFTs) should agree with scipy to within a couple of orders of
# float64 eps (2.2e-16); 1e-9/1e-10 leaves headroom for summation-order
# differences between libraries without hiding a genuine bug.
RTOL_F64 = 1e-9
ATOL_F64 = 1e-10

# float32/complex64: eps ~= 1.2e-7, and FFT butterfly summation accumulates
# error growing with log2(N); this tier is deliberately ~1000x looser than
# the float64 tier and must NOT be used to relax the float64 assertions.
RTOL_F32 = 1e-3
ATOL_F32 = 1e-4

# 1-D shapes: power-of-two (best case for every FFT algorithm), a
# non-power-of-two composite (100 = 2^2*5^2), and two primes (97, 101) --
# the worst case for radix/mixed-radix planners, which must fall back to
# Bluestein's algorithm or a direct DFT.
SHAPES_1D = {
    "pow2": 128,
    "non_pow2": 100,
    "prime_97": 97,
    "prime_101": 101,
    "length1": 1,
}


def _complex_signal(n, dtype=np.complex128):
    rng = np.random.default_rng(0)
    real = rng.standard_normal(n)
    imag = rng.standard_normal(n)
    return (real + 1j * imag).astype(dtype)


def _real_signal(n, dtype=np.float64):
    rng = np.random.default_rng(0)
    return rng.standard_normal(n).astype(dtype)


# ---------------------------------------------------------------------------
# Task B: 1-D fft/ifft and rfft/irfft vs scipy, across shapes and dtypes
# ---------------------------------------------------------------------------

class TestForward1DVsScipy:
    """fft/rfft forward transforms across shapes, backends, and dtype tiers."""

    @pytest.mark.parametrize("backend", all_backends_param("fft"))
    @pytest.mark.parametrize("shape_name", sorted(SHAPES_1D))
    def test_fft_complex128(self, backend, shape_name):
        n = SHAPES_1D[shape_name]
        x = _complex_signal(n, np.complex128)
        if assert_declared_limit(backend, lambda: fftkit.fft(x, backend=backend), length=n):
            return
        result = fftkit.fft(x, backend=backend)
        expected = spfft.fft(x)
        assert np.allclose(result, expected, rtol=RTOL_F64, atol=ATOL_F64), (
            f"{backend}/{shape_name}: max|diff|={np.max(np.abs(result - expected)):.3e}"
        )

    @pytest.mark.parametrize("backend", all_backends_param("fft"))
    @pytest.mark.parametrize("shape_name", sorted(SHAPES_1D))
    def test_fft_complex64_looser_tolerance(self, backend, shape_name):
        n = SHAPES_1D[shape_name]
        x = _complex_signal(n, np.complex64)
        if assert_declared_limit(backend, lambda: fftkit.fft(x, backend=backend), length=n):
            return
        result = fftkit.fft(x, backend=backend)
        expected = spfft.fft(x)
        assert np.allclose(result, expected, rtol=RTOL_F32, atol=ATOL_F32), (
            f"{backend}/{shape_name} (complex64): max|diff|={np.max(np.abs(result - expected)):.3e}"
        )

    @pytest.mark.parametrize("backend", all_backends_param("rfft"))
    @pytest.mark.parametrize("shape_name", sorted(SHAPES_1D))
    def test_rfft_float64(self, backend, shape_name):
        n = SHAPES_1D[shape_name]
        x = _real_signal(n, np.float64)
        result = fftkit.rfft(x, backend=backend)
        expected = spfft.rfft(x)
        assert np.allclose(result, expected, rtol=RTOL_F64, atol=ATOL_F64), (
            f"{backend}/{shape_name}: max|diff|={np.max(np.abs(result - expected)):.3e}"
        )

    @pytest.mark.parametrize("backend", all_backends_param("rfft"))
    @pytest.mark.parametrize("shape_name", sorted(SHAPES_1D))
    def test_rfft_float32_looser_tolerance(self, backend, shape_name):
        n = SHAPES_1D[shape_name]
        x = _real_signal(n, np.float32)
        result = fftkit.rfft(x, backend=backend)
        expected = spfft.rfft(x)
        assert np.allclose(result, expected, rtol=RTOL_F32, atol=ATOL_F32), (
            f"{backend}/{shape_name} (float32): max|diff|={np.max(np.abs(result - expected)):.3e}"
        )


class TestInverse1DVsScipy:
    """ifft/irfft round-trips and agreement with scipy's inverse."""

    @pytest.mark.parametrize("backend", all_backends_param("ifft"))
    @pytest.mark.parametrize("shape_name", sorted(SHAPES_1D))
    def test_ifft_matches_scipy(self, backend, shape_name):
        n = SHAPES_1D[shape_name]
        x = _complex_signal(n, np.complex128)
        X = spfft.fft(x)
        result = fftkit.ifft(X, backend=backend)
        expected = spfft.ifft(X)
        assert np.allclose(result, expected, rtol=RTOL_F64, atol=ATOL_F64)

    @pytest.mark.parametrize("backend", all_backends_param("ifft"))
    def test_ifft_fft_round_trip(self, backend):
        x = _complex_signal(64, np.complex128)
        X = fftkit.fft(x, backend=backend)
        recovered = fftkit.ifft(X, backend=backend)
        assert np.allclose(recovered, x, rtol=RTOL_F64, atol=ATOL_F64)

    @pytest.mark.parametrize("backend", all_backends_param("irfft"))
    @pytest.mark.parametrize("shape_name", sorted(SHAPES_1D))
    def test_irfft_matches_scipy(self, backend, shape_name):
        n = SHAPES_1D[shape_name]
        x = _real_signal(n, np.float64)
        X = spfft.rfft(x)
        result = fftkit.irfft(X, n=n, backend=backend)
        expected = spfft.irfft(X, n=n)
        assert np.allclose(result, expected, rtol=RTOL_F64, atol=ATOL_F64)


# ---------------------------------------------------------------------------
# Task B: 2-D / 3-D fft2/ifft2, fftn/ifftn
# ---------------------------------------------------------------------------

class TestNDTransformsVsScipy:
    """fft2/ifft2 and fftn/ifftn on genuinely multi-dimensional input."""

    @pytest.mark.parametrize("backend", all_backends_param("fft2"))
    def test_fft2_2d_input(self, backend):
        rng = np.random.default_rng(1)
        x = (rng.standard_normal((12, 20)) + 1j * rng.standard_normal((12, 20)))
        result = fftkit.fft2(x, backend=backend)
        expected = spfft.fft2(x)
        assert np.allclose(result, expected, rtol=RTOL_F64, atol=ATOL_F64)

    @pytest.mark.parametrize("backend", all_backends_param("fft2"))
    def test_ifft2_round_trip(self, backend):
        rng = np.random.default_rng(2)
        x = (rng.standard_normal((10, 14)) + 1j * rng.standard_normal((10, 14)))
        X = fftkit.fft2(x, backend=backend)
        recovered = fftkit.ifft2(X, backend=backend)
        assert np.allclose(recovered, x, rtol=RTOL_F64, atol=ATOL_F64)

    @pytest.mark.parametrize("backend", all_backends_param("fftn"))
    def test_fftn_3d_input(self, backend):
        rng = np.random.default_rng(3)
        x = (rng.standard_normal((4, 6, 5)) + 1j * rng.standard_normal((4, 6, 5)))
        result = fftkit.fftn(x, backend=backend)
        expected = spfft.fftn(x)
        assert np.allclose(result, expected, rtol=RTOL_F64, atol=ATOL_F64)

    @pytest.mark.parametrize("backend", all_backends_param("fftn"))
    def test_ifftn_3d_round_trip(self, backend):
        rng = np.random.default_rng(4)
        x = (rng.standard_normal((4, 6, 5)) + 1j * rng.standard_normal((4, 6, 5)))
        X = fftkit.fftn(x, backend=backend)
        recovered = fftkit.ifftn(X, backend=backend)
        assert np.allclose(recovered, x, rtol=RTOL_F64, atol=ATOL_F64)


# ---------------------------------------------------------------------------
# Task B: explicit axis, incl. the 0.1.0 -> 0.2.0 axis=0 -> axis=-1 regression
# ---------------------------------------------------------------------------

class TestAxisSelection:
    """Explicit axis in {0, 1, -1} on 2-D input.

    fftkit 0.1.0 defaulted 1-D transforms to axis=0; 0.2.0 switched to the
    NumPy/SciPy convention axis=-1 (last axis). test_default_axis_is_last_axis
    is the regression guard for that change: silently reverting the default
    would make this comparison fail against np.fft.fft's default (also -1).
    """

    @pytest.mark.parametrize("backend", all_backends_param("fft"))
    def test_default_axis_is_last_axis(self, backend):
        rng = np.random.default_rng(5)
        x = rng.standard_normal((6, 9)) + 1j * rng.standard_normal((6, 9))
        # Omitting axis= on 2-D input is exactly the case that changed meaning
        # between 0.1.x and 0.2.0, so it also raises AxisDefaultWarning. Asserted
        # here rather than filtered: if the warning stops firing, the migration
        # guard has silently regressed and this is where that should surface.
        if assert_declared_limit(
            backend, lambda: fftkit.fft(x, axis=-1, backend=backend), length=x.shape[-1]
        ):
            return
        with pytest.warns(fftkit.AxisDefaultWarning):
            result = fftkit.fft(x, backend=backend)  # no axis= given
        expected = np.fft.fft(x)  # numpy default axis=-1
        assert np.allclose(result, expected, rtol=RTOL_F64, atol=ATOL_F64), (
            "fftkit.fft(x) without axis= must transform the LAST axis "
            "(0.2.0 convention), not axis=0 (0.1.0 convention)"
        )

    @pytest.mark.parametrize("backend", all_backends_param("fft"))
    @pytest.mark.parametrize("axis", [0, 1, -1])
    def test_explicit_axis_matches_scipy(self, backend, axis):
        rng = np.random.default_rng(6)
        x = rng.standard_normal((7, 11)) + 1j * rng.standard_normal((7, 11))
        if assert_declared_limit(
            backend, lambda: fftkit.fft(x, axis=axis, backend=backend), length=x.shape[axis]
        ):
            return
        result = fftkit.fft(x, axis=axis, backend=backend)
        expected = spfft.fft(x, axis=axis)
        assert np.allclose(result, expected, rtol=RTOL_F64, atol=ATOL_F64), (
            f"{backend}, axis={axis}: mismatch vs scipy"
        )
        # axis=1 and axis=-1 must be the literal same result on a 2-D array.
        if axis in (1, -1):
            other = -1 if axis == 1 else 1
            result_other = fftkit.fft(x, axis=other, backend=backend)
            assert np.allclose(result, result_other, rtol=RTOL_F64, atol=ATOL_F64)


# ---------------------------------------------------------------------------
# Task B: n= zero-padding and truncation
# ---------------------------------------------------------------------------

class TestNParameter:
    """n= padding (n > len) and truncation (n < len), forward and inverse."""

    @pytest.mark.parametrize("backend", all_backends_param("fft"))
    @pytest.mark.parametrize("n_delta", [-10, 10])  # truncate / zero-pad
    def test_fft_n_padding_and_truncation(self, backend, n_delta):
        x = _complex_signal(64, np.complex128)
        n = 64 + n_delta
        if assert_declared_limit(backend, lambda: fftkit.fft(x, n=n, backend=backend), n=n):
            return
        result = fftkit.fft(x, n=n, backend=backend)
        expected = spfft.fft(x, n=n)
        assert result.shape[-1] == n
        assert np.allclose(result, expected, rtol=RTOL_F64, atol=ATOL_F64)

    @pytest.mark.parametrize("backend", all_backends_param("ifft"))
    @pytest.mark.parametrize("n_delta", [-10, 10])
    def test_ifft_n_padding_and_truncation(self, backend, n_delta):
        x = _complex_signal(64, np.complex128)
        X = spfft.fft(x)
        n = 64 + n_delta
        result = fftkit.ifft(X, n=n, backend=backend)
        expected = spfft.ifft(X, n=n)
        assert result.shape[-1] == n
        assert np.allclose(result, expected, rtol=RTOL_F64, atol=ATOL_F64)

    @pytest.mark.parametrize("backend", all_backends_param("rfft"))
    @pytest.mark.parametrize("n_delta", [-10, 10])
    def test_rfft_n_padding_and_truncation(self, backend, n_delta):
        x = _real_signal(64, np.float64)
        n = 64 + n_delta
        result = fftkit.rfft(x, n=n, backend=backend)
        expected = spfft.rfft(x, n=n)
        assert result.shape[-1] == n // 2 + 1
        assert np.allclose(result, expected, rtol=RTOL_F64, atol=ATOL_F64)


# ---------------------------------------------------------------------------
# Task B: norm='backward' | 'ortho' | 'forward' -- verified via Parseval
# ---------------------------------------------------------------------------

class TestNormConventions:
    """Assert the correct Parseval constant for each norm mode, not merely
    that the call succeeds. For a length-N complex FFT X = fft(x, norm=...):

        norm='backward' (default): sum|x|^2 = (1/N) * sum|X|^2
        norm='ortho'   : sum|x|^2 =            sum|X|^2   (unitary)
        norm='forward' : sum|x|^2 =    N     * sum|X|^2

    Verified numerically against scipy to 1e-12 relative in a scratch
    calculation before writing these assertions.
    """

    @pytest.mark.parametrize("backend", all_backends_param("fft"))
    @pytest.mark.parametrize("norm,scale_energy", [
        ("backward", lambda E, N: E / N),
        ("ortho", lambda E, N: E),
        ("forward", lambda E, N: E * N),
    ])
    def test_parseval_per_norm(self, backend, norm, scale_energy):
        x = _complex_signal(97, np.complex128)  # prime length: no radix shortcuts
        N = len(x)
        if assert_declared_limit(
            backend, lambda: fftkit.fft(x, norm=norm, backend=backend), length=N, norm=norm
        ):
            return
        X = fftkit.fft(x, norm=norm, backend=backend)

        E_time = np.sum(np.abs(x) ** 2)
        E_freq = scale_energy(np.sum(np.abs(X) ** 2), N)

        # 1e-9 relative: same justification as RTOL_F64 above (a couple of
        # orders above float64 eps to absorb summation-order differences).
        assert np.isclose(E_time, E_freq, rtol=1e-9), (
            f"{backend}/{norm}: time energy {E_time:.6f} vs scaled freq energy {E_freq:.6f}"
        )

    # Requires ifft as well as fft: parametrizing on "fft" alone ran this
    # against accelerate, which implements no ifft.
    @pytest.mark.parametrize("backend", all_backends_param(("fft", "ifft")))
    @pytest.mark.parametrize("norm", ["backward", "ortho", "forward"])
    def test_ifft_fft_round_trip_per_norm(self, backend, norm):
        x = _complex_signal(50, np.complex128)
        if assert_declared_limit(
            backend, lambda: fftkit.fft(x, norm=norm, backend=backend), length=len(x), norm=norm
        ):
            return
        X = fftkit.fft(x, norm=norm, backend=backend)
        recovered = fftkit.ifft(X, norm=norm, backend=backend)
        assert np.allclose(recovered, x, rtol=RTOL_F64, atol=ATOL_F64), (
            f"{backend}/{norm}: ifft(fft(x, norm=norm), norm=norm) != x"
        )


# ---------------------------------------------------------------------------
# Partial backends must raise NotImplementedError for transforms they lack,
# never silently skip or silently succeed with a wrong result.
# ---------------------------------------------------------------------------

TRANSFORM_NAMES = ("fft", "ifft", "rfft", "irfft", "fft2", "ifft2", "fftn", "ifftn")


# For every backend that IS available on this machine but does NOT implement
# a given transform (per its registered Backend.transforms map -- e.g.
# accelerate only has 'fft'; tensorflow has no fftn/ifftn), calling it
# through the top-level dispatch must raise NotImplementedError, not
# silently fall back to a different backend and not raise some other
# exception type. Uses partial_backends_param so backends that ARE available
# and DO support the transform, or aren't available at all, show up as
# named skips instead of being silently absent.
#
# Each transform needs its own skip set (partial_backends_param(transform)),
# which a plain double @pytest.mark.parametrize can't express cleanly, so
# the (backend, transform) matrix is built explicitly instead.
def _partial_backend_transform_cases():
    cases = []
    for transform in TRANSFORM_NAMES:
        for param in partial_backends_param(transform):
            cases.append(pytest.param(param.values[0], transform, marks=param.marks,
                                       id=f"{param.id}-{transform}"))
    return cases


@pytest.mark.parametrize("backend,transform", _partial_backend_transform_cases())
def test_unsupported_transform_raises_not_implemented(backend, transform):
    func = getattr(fftkit, transform)
    x = np.ones(8) if transform == "rfft" else np.ones(8, dtype=np.complex128)
    with pytest.raises(NotImplementedError):
        func(x, backend=backend)


class TestNextFastLen:
    """fftkit.next_fast_len: the padding-for-speed helper.

    Its value is a performance property, which a unit test cannot assert
    reliably (timings are machine- and load-dependent). What IS testable is the
    contract the speedup rests on: the returned length is at least n, is built
    only from small prime factors, and composes with the n= argument to give
    exactly the transform scipy would give at that length. The measured
    speedups live in the function's docstring and benchmarks/study_padding.py.
    """

    @pytest.mark.parametrize("n", [1, 2, 7, 100, 10007, 65521, 262139])
    def test_never_shorter_than_requested(self, n):
        assert fftkit.next_fast_len(n) >= n

    @pytest.mark.parametrize("n", [2, 4, 8, 1024, 4096, 65536])
    def test_already_fast_lengths_are_unchanged(self, n):
        """Powers of two are optimal for every radix-2 implementation, so
        padding them would be pure waste."""
        assert fftkit.next_fast_len(n) == n

    @pytest.mark.parametrize("n", [10007, 65521, 262139, 1000003])
    def test_result_factors_into_small_primes(self, n):
        """The whole point: a length whose factors are small primes takes the
        radix path instead of Bluestein/direct-DFT. scipy's preferred radices
        are 2, 3, 5, 7 and 11, so the result must contain no larger factor."""
        m = fftkit.next_fast_len(n)
        remaining = m
        for prime in (2, 3, 5, 7, 11):
            while remaining % prime == 0:
                remaining //= prime
        assert remaining == 1, (
            f"next_fast_len({n}) = {m} still has a factor {remaining} outside "
            "{2,3,5,7,11}, which would defeat the radix path"
        )

    @pytest.mark.parametrize("n", [97, 10007, 65521])
    def test_agrees_with_scipy(self, n):
        """Documented as delegating to scipy; if that changes, the docstring's
        measured speedups no longer describe this function."""
        assert fftkit.next_fast_len(n) == spfft.next_fast_len(n)
        assert fftkit.next_fast_len(n, real=True) == spfft.next_fast_len(n, real=True)

    @pytest.mark.parametrize("backend", all_backends_param("fft"))
    def test_composes_with_n_argument(self, backend):
        """The documented idiom, fft(x, n=next_fast_len(len(x))), must give
        exactly scipy's transform at the padded length -- padding for speed
        must not change the answer."""
        rng = np.random.default_rng(21)
        n = 1013  # prime, so next_fast_len actually moves
        x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        target = fftkit.next_fast_len(n)
        assert target > n, "test is pointless if the length does not change"
        if assert_declared_limit(
            backend, lambda: fftkit.fft(x, n=target, backend=backend), n=target
        ):
            return
        result = fftkit.fft(x, n=target, backend=backend)
        assert result.shape[-1] == target
        assert np.allclose(result, spfft.fft(x, n=target), rtol=RTOL_F64, atol=ATOL_F64)

    def test_padding_interpolates_rather_than_resolves(self):
        """Padding refines the frequency grid but adds no resolution. Guards the
        docstring claim, since users reach for padding expecting the opposite:
        the padded spectrum must be finer-grained yet peak at the same
        frequency, not reveal a second tone that was not separable before.
        """
        fs, n = 1000.0, 512
        t = np.arange(n) / fs
        x = np.sin(2 * np.pi * 100.0 * t)
        target = 4 * n

        plain = np.abs(fftkit.rfft(x))
        padded = np.abs(fftkit.rfft(x, n=target))
        f_plain = fftkit.rfftfreq(n, 1 / fs)
        f_padded = fftkit.rfftfreq(target, 1 / fs)

        # Finer grid...
        assert f_padded[1] - f_padded[0] < f_plain[1] - f_plain[0]
        # ...but the same peak, within one bin of the padded grid.
        assert abs(f_padded[np.argmax(padded)] - f_plain[np.argmax(plain)]) <= (
            f_padded[1] - f_padded[0]
        )


class TestWorkersDoesNotInterfereWithOtherKeywords:
    """`workers=` is the newest keyword on the matrix; the risk worth a
    regression test is not threading itself (see test_workers.py) but that
    adding it broke the existing n=/axis=/norm= keywords it now sits next
    to -- e.g. an argument-order slip that shifted a positional binding.
    """

    @pytest.mark.parametrize("backend", all_backends_param("fft"))
    def test_workers_alongside_n_axis_norm_on_fft(self, backend):
        x = _complex_signal(100, np.complex128)
        if assert_declared_limit(backend, lambda: fftkit.fft(x, backend=backend), length=100):
            return
        with_workers = fftkit.fft(x, n=128, axis=-1, norm="ortho", backend=backend, workers=None)
        without_workers = fftkit.fft(x, n=128, axis=-1, norm="ortho", backend=backend)
        assert np.array_equal(with_workers, without_workers)
        expected = spfft.fft(x, n=128, axis=-1, norm="ortho")
        assert np.allclose(with_workers, expected, rtol=RTOL_F64, atol=ATOL_F64)
