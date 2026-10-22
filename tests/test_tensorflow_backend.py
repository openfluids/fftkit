"""Test suite for the TensorFlow FFT backend.

This module tests both fixes to the TensorFlow backend:
1. Support for real (float32/float64) input to fft/ifft
2. Support for norm='ortho' and norm='forward' in all transforms
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import fft as scipy_fft

import fftkit
from fftkit.backends import BACKENDS

# Skip entire module if TensorFlow is not available
tensorflow = pytest.importorskip("tensorflow")


class TestTensorflowRealInput:
    """Test that real input is properly cast to complex (Defect 1)."""

    def test_float64_to_complex128(self):
        """float64 real input should produce complex128 output."""
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        result = fftkit.fft(x, backend='tensorflow')
        assert result.dtype == np.complex128
        # Verify correctness against scipy
        expected = scipy_fft.fft(x)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_float32_to_complex64(self):
        """float32 real input should produce complex64 output."""
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = fftkit.fft(x, backend='tensorflow')
        assert result.dtype == np.complex64
        expected = scipy_fft.fft(x)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_complex64_passthrough(self):
        """complex64 input should remain complex64."""
        x = np.array([1.0+1.0j, 2.0+1.0j], dtype=np.complex64)
        result = fftkit.fft(x, backend='tensorflow')
        assert result.dtype == np.complex64

    def test_complex128_passthrough(self):
        """complex128 input should remain complex128."""
        x = np.array([1.0+1.0j, 2.0+1.0j], dtype=np.complex128)
        result = fftkit.fft(x, backend='tensorflow')
        assert result.dtype == np.complex128

    def test_float64_real_input_all_transforms(self):
        """All transforms should accept float64 real input."""
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

        # Test fft
        result_fft = fftkit.fft(x, backend='tensorflow')
        assert result_fft.dtype == np.complex128

        # Test ifft (on complex input)
        x_complex = result_fft
        result_ifft = fftkit.ifft(x_complex, backend='tensorflow')
        assert result_ifft.dtype == np.complex128

        # Test rfft
        result_rfft = fftkit.rfft(x, backend='tensorflow')
        assert result_rfft.dtype == np.complex128

        # Test irfft (on complex input, returns real)
        result_irfft = fftkit.irfft(result_rfft, backend='tensorflow')
        assert result_irfft.dtype == np.float64


class TestTensorflowNormalization:
    """Test norm='ortho' and norm='forward' support (Defect 2)."""

    def test_fft_all_norms_match_scipy(self):
        """fft with all norms should match scipy."""
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.complex128)

        for norm in [None, 'backward', 'ortho', 'forward']:
            tf_result = fftkit.fft(x, backend='tensorflow', norm=norm)
            scipy_result = scipy_fft.fft(x, norm=norm)
            np.testing.assert_allclose(tf_result, scipy_result, rtol=1e-6,
                                      err_msg=f"fft norm={norm} mismatch")

    def test_ifft_all_norms_match_scipy(self):
        """ifft with all norms should match scipy."""
        x = np.array([1.0+1.0j, 2.0+1.0j, 3.0+1.0j, 4.0+1.0j], dtype=np.complex128)

        for norm in [None, 'backward', 'ortho', 'forward']:
            tf_result = fftkit.ifft(x, backend='tensorflow', norm=norm)
            scipy_result = scipy_fft.ifft(x, norm=norm)
            np.testing.assert_allclose(tf_result, scipy_result, rtol=1e-6,
                                      err_msg=f"ifft norm={norm} mismatch")

    def test_rfft_all_norms_match_scipy(self):
        """rfft with all norms should match scipy."""
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

        for norm in [None, 'backward', 'ortho', 'forward']:
            tf_result = fftkit.rfft(x, backend='tensorflow', norm=norm)
            scipy_result = scipy_fft.rfft(x, norm=norm)
            np.testing.assert_allclose(tf_result, scipy_result, rtol=1e-6,
                                      err_msg=f"rfft norm={norm} mismatch")

    def test_irfft_all_norms_match_scipy(self):
        """irfft with all norms should match scipy."""
        # Use rfft to generate proper complex input
        x_real = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        x_complex = scipy_fft.rfft(x_real)

        for norm in [None, 'backward', 'ortho', 'forward']:
            tf_result = fftkit.irfft(x_complex, n=len(x_real), backend='tensorflow', norm=norm)
            scipy_result = scipy_fft.irfft(x_complex, n=len(x_real), norm=norm)
            np.testing.assert_allclose(tf_result, scipy_result, rtol=1e-6,
                                      err_msg=f"irfft norm={norm} mismatch")

    def test_fft2_all_norms_match_scipy(self):
        """fft2 with all norms should match scipy."""
        x = np.random.randn(4, 4) + 1j*np.random.randn(4, 4)

        for norm in [None, 'backward', 'ortho', 'forward']:
            tf_result = fftkit.fft2(x, backend='tensorflow', norm=norm)
            scipy_result = scipy_fft.fft2(x, norm=norm)
            np.testing.assert_allclose(tf_result, scipy_result, rtol=1e-6,
                                      err_msg=f"fft2 norm={norm} mismatch")

    def test_ifft2_all_norms_match_scipy(self):
        """ifft2 with all norms should match scipy."""
        x = np.random.randn(4, 4) + 1j*np.random.randn(4, 4)

        for norm in [None, 'backward', 'ortho', 'forward']:
            tf_result = fftkit.ifft2(x, backend='tensorflow', norm=norm)
            scipy_result = scipy_fft.ifft2(x, norm=norm)
            np.testing.assert_allclose(tf_result, scipy_result, rtol=1e-6,
                                      err_msg=f"ifft2 norm={norm} mismatch")

    def test_norm_none_equals_backward_fft(self):
        """norm=None should behave identically to norm='backward' for fft."""
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.complex128)

        result_none = fftkit.fft(x, backend='tensorflow', norm=None)
        result_backward = fftkit.fft(x, backend='tensorflow', norm='backward')
        np.testing.assert_array_equal(result_none, result_backward)

    def test_norm_none_equals_backward_ifft(self):
        """norm=None should behave identically to norm='backward' for ifft."""
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.complex128)

        result_none = fftkit.ifft(x, backend='tensorflow', norm=None)
        result_backward = fftkit.ifft(x, backend='tensorflow', norm='backward')
        np.testing.assert_array_equal(result_none, result_backward)

    def test_norm_none_equals_backward_rfft(self):
        """norm=None should behave identically to norm='backward' for rfft."""
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

        result_none = fftkit.rfft(x, backend='tensorflow', norm=None)
        result_backward = fftkit.rfft(x, backend='tensorflow', norm='backward')
        np.testing.assert_array_equal(result_none, result_backward)

    def test_norm_none_equals_backward_irfft(self):
        """norm=None should behave identically to norm='backward' for irfft."""
        x_real = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        x_complex = scipy_fft.rfft(x_real)

        result_none = fftkit.irfft(x_complex, n=len(x_real), backend='tensorflow', norm=None)
        result_backward = fftkit.irfft(x_complex, n=len(x_real), backend='tensorflow', norm='backward')
        np.testing.assert_array_equal(result_none, result_backward)


class TestTensorflowDtypeMapping:
    """Regression tests to ensure dtype mapping is preserved (Defect 1)."""

    @pytest.mark.parametrize("input_dtype,expected_output_dtype,transform", [
        (np.float64, np.complex128, 'fft'),
        (np.float32, np.complex64, 'fft'),
        (np.complex64, np.complex64, 'fft'),
        (np.complex128, np.complex128, 'fft'),
        (np.float64, np.complex128, 'ifft'),
        (np.float32, np.complex64, 'ifft'),
        (np.float64, np.complex128, 'rfft'),
        (np.float32, np.complex64, 'rfft'),
    ])
    def test_dtype_preservation(self, input_dtype, expected_output_dtype, transform):
        """Each input dtype should map to the expected output dtype."""
        if input_dtype in (np.float32, np.float64):
            x = np.array([1.0, 2.0, 3.0, 4.0], dtype=input_dtype)
        else:
            x = np.array([1.0+1.0j, 2.0+1.0j, 3.0+1.0j, 4.0+1.0j], dtype=input_dtype)

        result = fftkit.__dict__[transform](x, backend='tensorflow')
        assert result.dtype == expected_output_dtype, \
            f"{transform}({input_dtype}) produced {result.dtype}, expected {expected_output_dtype}"


class TestTensorflowRoundTrip:
    """Test round-trip invariance: ifft(fft(x)) == x for all norms."""

    def test_complex_roundtrip_all_norms(self):
        """Complex fft -> ifft should recover input for all norms."""
        x = np.array([1.0+0.5j, 2.0+0.3j, 3.0+0.2j, 4.0+0.1j], dtype=np.complex128)

        for norm in [None, 'backward', 'ortho', 'forward']:
            X = fftkit.fft(x, backend='tensorflow', norm=norm)
            x_recovered = fftkit.ifft(X, backend='tensorflow', norm=norm)
            np.testing.assert_allclose(x_recovered, x, rtol=1e-6, atol=1e-14,
                                      err_msg=f"Round-trip failed for norm={norm}")

    def test_real_roundtrip_all_norms(self):
        """Real rfft -> irfft should recover input for all norms."""
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

        for norm in [None, 'backward', 'ortho', 'forward']:
            X = fftkit.rfft(x, backend='tensorflow', norm=norm)
            x_recovered = fftkit.irfft(X, n=len(x), backend='tensorflow', norm=norm)
            np.testing.assert_allclose(x_recovered, x, rtol=1e-6, atol=1e-14,
                                      err_msg=f"Real round-trip failed for norm={norm}")


class TestTensorflowParseval:
    """Test Parseval's theorem holds with correct normalization."""

    def test_parseval_backward_norm(self):
        """Parseval: sum(|x|^2) == (1/N)*sum(|X|^2) for backward norm."""
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.complex128)
        N = len(x)

        X = fftkit.fft(x, backend='tensorflow', norm='backward')

        energy_time = np.sum(np.abs(x)**2)
        energy_freq = np.sum(np.abs(X)**2) / N

        np.testing.assert_allclose(energy_time, energy_freq, rtol=1e-6)

    def test_parseval_ortho_norm(self):
        """Parseval: sum(|x|^2) == sum(|X|^2) for ortho norm."""
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.complex128)

        X = fftkit.fft(x, backend='tensorflow', norm='ortho')

        energy_time = np.sum(np.abs(x)**2)
        energy_freq = np.sum(np.abs(X)**2)

        np.testing.assert_allclose(energy_time, energy_freq, rtol=1e-6)

    def test_parseval_forward_norm(self):
        """Parseval: sum(|x|^2) == N*sum(|X|^2) for forward norm."""
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.complex128)
        N = len(x)

        X = fftkit.fft(x, backend='tensorflow', norm='forward')

        energy_time = np.sum(np.abs(x)**2)
        energy_freq = N * np.sum(np.abs(X)**2)

        np.testing.assert_allclose(energy_time, energy_freq, rtol=1e-6)


class TestTensorflowWithPaddingAndTruncation:
    """Test that n=/s= works correctly with all norms."""

    def test_fft_with_n_padding(self):
        """fft with n= padding should match scipy for all norms."""
        x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        n = 8  # Pad to 8

        for norm in [None, 'backward', 'ortho', 'forward']:
            tf_result = fftkit.fft(x, n=n, backend='tensorflow', norm=norm)
            scipy_result = scipy_fft.fft(x, n=n, norm=norm)
            np.testing.assert_allclose(tf_result, scipy_result, rtol=1e-6,
                                      err_msg=f"fft(n={n}) norm={norm} mismatch")

    def test_fft_with_n_truncation(self):
        """fft with n= truncation should match scipy for all norms."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        n = 3  # Truncate to 3

        for norm in [None, 'backward', 'ortho', 'forward']:
            tf_result = fftkit.fft(x, n=n, backend='tensorflow', norm=norm)
            scipy_result = scipy_fft.fft(x, n=n, norm=norm)
            np.testing.assert_allclose(tf_result, scipy_result, rtol=1e-6,
                                      err_msg=f"fft(n={n}) norm={norm} mismatch")

    def test_rfft_with_n(self):
        """rfft with n= should match scipy for all norms."""
        x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        n = 8

        for norm in [None, 'backward', 'ortho', 'forward']:
            tf_result = fftkit.rfft(x, n=n, backend='tensorflow', norm=norm)
            scipy_result = scipy_fft.rfft(x, n=n, norm=norm)
            np.testing.assert_allclose(tf_result, scipy_result, rtol=1e-6,
                                      err_msg=f"rfft(n={n}) norm={norm} mismatch")

    def test_fft2_with_s_padding(self):
        """fft2 with s= padding should match scipy for all norms."""
        x = np.random.randn(3, 3)
        s = (8, 8)

        for norm in [None, 'backward', 'ortho', 'forward']:
            tf_result = fftkit.fft2(x, s=s, backend='tensorflow', norm=norm)
            scipy_result = scipy_fft.fft2(x, s=s, norm=norm)
            np.testing.assert_allclose(tf_result, scipy_result, rtol=1e-6,
                                      err_msg=f"fft2(s={s}) norm={norm} mismatch")


class TestTensorflowLimitationsPreserved:
    """Verify that declared limitations are still enforced."""

    def test_fftn_not_implemented(self):
        """tensorflow backend should not support fftn."""
        assert not BACKENDS['tensorflow'].supports('fftn')

        x = np.random.randn(2, 3, 4)
        with pytest.raises(NotImplementedError, match="does not implement"):
            fftkit.fftn(x, backend='tensorflow')

    def test_ifftn_not_implemented(self):
        """tensorflow backend should not support ifftn."""
        assert not BACKENDS['tensorflow'].supports('ifftn')

        x = np.random.randn(2, 3, 4) + 1j*np.random.randn(2, 3, 4)
        with pytest.raises(NotImplementedError, match="does not implement"):
            fftkit.ifftn(x, backend='tensorflow')
