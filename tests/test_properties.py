"""Property-based tests for fftkit's core FFT identities, using hypothesis
over generated signals rather than one fixed 50 Hz sine.

Signals are constrained to finite, bounded-magnitude floats: NaN/inf would
make every downstream assertion vacuously fail, and unbounded magnitude
(e.g. 1e300) would blow float64 cancellation error past any fixed tolerance
regardless of correctness -- neither is a meaningful test of FFT correctness,
just of float edge-case handling this module doesn't claim to guarantee.

max_examples / deadline are tuned to keep the whole property suite under
roughly a couple of seconds, per the "keep the whole suite fast" constraint.
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import fftkit

# Bounded, finite float64 element strategy: magnitude capped at 1e3 keeps
# FFT sums (which grow like N * max|x|) well inside float64's exact-integer
# range (2^53 ~ 9e15) even for the largest N used below (64), so tolerance
# failures reflect real FFT bugs, not accumulated rounding from huge inputs.
_finite_floats = st.floats(
    min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False, width=64
)

# Signal lengths kept small (2..64) so hypothesis can explore many shapes
# per second; FFT correctness at length 64 generalizes to larger N (same
# algorithm path), and small N maximizes the number of distinct examples
# hypothesis can try within the time budget.
_lengths = st.integers(min_value=2, max_value=64)


def _real_arrays(length):
    return hnp.arrays(dtype=np.float64, shape=length, elements=_finite_floats)


def _complex_arrays(length):
    return hnp.arrays(
        dtype=np.complex128,
        shape=length,
        elements=st.complex_numbers(
            min_magnitude=0, max_magnitude=1e3, allow_nan=False, allow_infinity=False
        ),
    )


# Relative/absolute tolerance: same justification as test_invariants.py's
# TOL_INVARIANTS -- a few orders above float64 eps (2.2e-16) to absorb
# summation-order differences, tight enough to catch a real defect.
RTOL = 1e-8
ATOL = 1e-6  # slightly looser atol: near-zero bins after cancellation


@st.composite
def _length_and_two_real_arrays(draw):
    n = draw(_lengths)
    x = draw(_real_arrays(n))
    y = draw(_real_arrays(n))
    return n, x, y


@st.composite
def _length_and_two_complex_arrays(draw):
    n = draw(_lengths)
    x = draw(_complex_arrays(n))
    y = draw(_complex_arrays(n))
    return n, x, y


@st.composite
def _length_array_and_shift(draw):
    n = draw(_lengths)
    x = draw(_real_arrays(n))
    k = draw(st.integers(min_value=0, max_value=n - 1))
    return n, x, k


class TestLinearity:
    """fft(a*x + b*y) == a*fft(x) + b*fft(y) for any scalars a, b."""

    @settings(max_examples=50, deadline=None)
    @given(
        data=_length_and_two_complex_arrays(),
        a=st.complex_numbers(min_magnitude=0, max_magnitude=10, allow_nan=False, allow_infinity=False),
        b=st.complex_numbers(min_magnitude=0, max_magnitude=10, allow_nan=False, allow_infinity=False),
    )
    def test_fft_is_linear(self, data, a, b):
        n, x, y = data
        lhs = fftkit.fft(a * x + b * y)
        rhs = a * fftkit.fft(x) + b * fftkit.fft(y)
        assert np.allclose(lhs, rhs, rtol=RTOL, atol=ATOL)


class TestShiftTheorem:
    """A circular shift in time is a linear phase ramp in frequency:
    fft(roll(x, k))[m] == fft(x)[m] * exp(-2j*pi*m*k/N).
    """

    @settings(max_examples=50, deadline=None)
    @given(data=_length_array_and_shift())
    def test_time_shift_is_frequency_phase_ramp(self, data):
        n, x, k = data
        X = fftkit.fft(x)
        x_shifted = np.roll(x, k)
        X_shifted = fftkit.fft(x_shifted)

        m = np.arange(n)
        predicted = X * np.exp(-2j * np.pi * m * k / n)
        assert np.allclose(X_shifted, predicted, rtol=RTOL, atol=ATOL)


class TestConjugateSymmetryOfRfft:
    """rfft(x) on real x holds exactly the non-redundant half of fft(x);
    reconstructing the full spectrum via conjugate symmetry
    (X[N-k] = conj(X[k])) must reproduce fft(x) exactly.
    """

    @settings(max_examples=50, deadline=None)
    @given(n=_lengths, x=st.data())
    def test_rfft_reconstructs_full_spectrum_via_conjugate_symmetry(self, n, x):
        signal = x.draw(_real_arrays(n))
        X_full = fftkit.fft(signal)
        X_r = fftkit.rfft(signal)

        half = n // 2 + 1
        assert len(X_r) == half
        # rfft's own bins must literally equal fft's first `half` bins.
        assert np.allclose(X_r, X_full[:half], rtol=RTOL, atol=ATOL)

        # Reconstruct the mirrored half and compare to fft(x) directly.
        reconstructed = np.empty(n, dtype=np.complex128)
        reconstructed[:half] = X_r
        for k in range(1, n - half + 1):
            reconstructed[n - k] = np.conj(X_r[k])
        assert np.allclose(reconstructed, X_full, rtol=RTOL, atol=ATOL)


class TestRoundTrip:
    """ifft(fft(x)) == x."""

    @settings(max_examples=50, deadline=None)
    @given(n=_lengths, x=st.data())
    def test_ifft_fft_round_trip_complex(self, n, x):
        signal = x.draw(_complex_arrays(n))
        recovered = fftkit.ifft(fftkit.fft(signal))
        assert np.allclose(recovered, signal, rtol=RTOL, atol=ATOL)

    @settings(max_examples=50, deadline=None)
    @given(n=_lengths, x=st.data())
    def test_irfft_rfft_round_trip_real(self, n, x):
        signal = x.draw(_real_arrays(n))
        recovered = fftkit.irfft(fftkit.rfft(signal), n=n)
        assert np.allclose(recovered, signal, rtol=RTOL, atol=ATOL)
