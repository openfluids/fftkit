"""
Shared FFT backend selection and wrapper utilities for FFT operations.

fftkit supports multiple FFT backends with different performance
characteristics, each exposed through the same transform matrix:
``fft, ifft, rfft, irfft, fft2, ifft2, fftn, ifftn`` at ``axis=-1``
(NumPy/SciPy convention).

Backends:

- ``scipy``: default, always available
- ``numpy``: fallback, always available
- ``mkl``: Intel MKL via ``mkl_fft`` (2-20x faster than scipy for large FFTs)
- ``pyfftw``: FFTW via ``pyfftw``
- ``cupy``: NVIDIA GPU via ``cupy`` (host<->device transfer handled internally)
- ``torch``: PyTorch (CPU or CUDA tensor, depending on install)
- ``tensorflow``: TensorFlow ``tf.signal`` (partial transform matrix)
- ``accelerate``: Apple Accelerate/vDSP via ctypes, macOS only
  (1-D complex forward FFT, power-of-two lengths only)

Optional backends are imported lazily, so ``import fftkit`` only requires
numpy and scipy. Use ``get_available_backends()`` to see what actually works
on this machine.

Environment variables:

``FFTKIT_BACKEND``: Override the default FFT backend (takes precedence)
    export FFTKIT_BACKEND=mkl

Threading (``workers=``):

Every transform in the matrix accepts an optional ``workers`` keyword, the
last argument in each signature. It is a performance hint, not part of the
numerical contract:

- ``workers=None`` (the default): pass nothing to the backend, i.e. a true
  no-op that leaves the backend's own default threading untouched.
- ``workers=-1``: use all available cores (``scipy.fft`` interprets this
  natively; the ``pyfftw`` adapter maps it to ``os.cpu_count()``).
- ``workers=N``: use ``N`` threads, where the backend supports it.

Only ``scipy`` (``workers=``) and ``pyfftw`` (mapped to ``threads=``) honour
this. Every other backend -- ``numpy`` (no threading support), ``mkl``
(threads via its own environment/runtime, not a per-call argument), ``cupy``,
``torch``, ``tensorflow`` (inherently parallel, no per-call equivalent) and
``accelerate`` (a single-threaded ctypes path) -- silently ignores it rather
than raising, so code written against scipy's ``workers=`` keyword stays
portable across every backend in the matrix. This is safe specifically
because ``workers`` only ever changes how many threads compute a result, never
the result itself.

**MPI / multi-process users:** this library is commonly run one process per
MPI rank during DNS/LES post-processing. ``workers=-1`` inside N ranks on an
N-core node spawns up to N threads *per rank*, oversubscribing the node by a
factor of N. The default (``workers=None``) is deliberately conservative for
exactly this reason -- opt into ``workers=-1``/``workers=N`` only in
single-process, single-rank contexts (e.g. interactive post-processing on a
workstation), never inside an MPI-parallel run.
"""

from __future__ import annotations

import importlib
import os
import sys
import warnings
from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import DEFAULT_BACKEND
from .gpu import gpu_available as gpu_available  # re-exported for backwards compat

TRANSFORM_NAMES = ("fft", "ifft", "rfft", "irfft", "fft2", "ifft2", "fftn", "ifftn")

#: 1-D transforms whose default ``axis`` moved from 0 to -1 in 0.2.0.
_AXIS_MOVED_TRANSFORMS = ("fft", "ifft", "rfft", "irfft")


class AxisDefaultWarning(UserWarning):
    """The default ``axis`` changed from 0 to -1 in fftkit 0.2.0.

    Raised when a multi-dimensional array reaches a 1-D transform without an
    explicit ``axis``, because that is the only case where the change alters
    results. Such a call transformed columns under 0.1.x and transforms rows
    now, with no error either way -- a silently different answer, which is
    the failure mode worth a warning.

    Silence it by passing ``axis`` explicitly (the fix, not a workaround), or
    globally::

        warnings.filterwarnings("ignore", category=fftkit.AxisDefaultWarning)

    A UserWarning subclass rather than DeprecationWarning on purpose:
    DeprecationWarning is hidden by default outside ``__main__``, so library
    users -- exactly the people affected -- would never see it. Slated for
    removal in 0.3.0.
    """


class MklBackendWarning(UserWarning):
    """``register_mkl_scipy_backend()`` could not install the MKL scipy backend.

    Carries the name of the module that failed to import, because the function
    returns a bare ``False`` and the two failure modes need different fixes.
    See :func:`register_mkl_scipy_backend`.
    """


class _Unset:
    """Sentinel distinguishing 'caller omitted axis' from 'caller passed -1'."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<unset>"


UNSET_AXIS: Any = _Unset()


def resolve_axis(x: ArrayLike, axis: Any, transform: str) -> int:
    """Return the effective ``axis``, warning if the 0.2.0 default change bites.

    Only warns when the caller omitted ``axis`` AND the input has more than one
    dimension. A 1-D signal gives the same result for axis 0 and -1, so warning
    there would be noise for the overwhelming majority of calls.
    """
    if not isinstance(axis, _Unset):
        return int(axis)

    if transform in _AXIS_MOVED_TRANSFORMS and np.ndim(x) > 1:
        warnings.warn(
            f"fftkit.{transform}() received a {np.ndim(x)}-D array without an explicit "
            "'axis'. The default changed from axis=0 to axis=-1 in fftkit 0.2.0 to match "
            "numpy and scipy, so this call now transforms the LAST axis where fftkit "
            "0.1.x transformed the FIRST one. Pass axis=-1 to keep the new behaviour "
            "(and silence this warning), or axis=0 to restore the old one.",
            AxisDefaultWarning,
            stacklevel=3,
        )
    return -1

# Result of any registered transform. rfft/irfft can return real or complex
# depending on direction, so this stays a plain ndarray of unspecified dtype.
ArrayResult = NDArray[Any]

# Call signature of a single registered transform. The transform matrix has
# two genuinely different call conventions in practice -- 1-D transforms take
# (x, n=None, axis=-1, norm=None), N-D transforms take (x, s=None, axes=None,
# norm=None) -- and several backends (e.g. torch's 1-D wrappers) only accept
# their own family's keywords, not both. A structural Protocol strict enough
# to name every keyword would therefore reject some of the very callables it
# needs to describe, so this alias fixes the shared leading positional
# argument (``x``, the input array) and leaves the rest as keyword arguments,
# rather than falling back to a bare, unparameterized ``Callable``.
TransformFunc = Callable[..., ArrayResult]


class Backend:
    """Describes one FFT backend: its name and the transforms it implements.

    ``transforms`` maps a subset (or all) of ``TRANSFORM_NAMES`` to callables.
    Transforms the backend does not support are simply absent; requesting one
    via :meth:`get` raises a ``NotImplementedError`` naming the backend and
    the transform.
    """

    def __init__(self, name: str, transforms: dict[str, TransformFunc]) -> None:
        self.name = name
        self.transforms = transforms

    def supports(self, transform: str) -> bool:
        return transform in self.transforms

    def get(self, transform: str) -> TransformFunc:
        try:
            return self.transforms[transform]
        except KeyError:
            raise NotImplementedError(f"Backend '{self.name}' does not implement '{transform}'.") from None

    def __call__(self, transform: str, *args: Any, **kwargs: Any) -> ArrayResult:
        return self.get(transform)(*args, **kwargs)


def _lazy_module_backend(
    name: str,
    module_path: str,
    transforms: tuple[str, ...] = TRANSFORM_NAMES,
    setup: Callable[[Any], None] | None = None,
    workers_kwarg: str | None = None,
) -> Backend:
    """Build a Backend that imports ``module_path`` lazily and delegates directly.

    Works for any namespace that already implements the NumPy/SciPy FFT
    calling convention: ``scipy.fft``, ``numpy.fft``, ``mkl_fft.interfaces.numpy_fft``,
    ``pyfftw.interfaces.numpy_fft``.

    ``setup``, if given, runs once with the imported module the first time any
    transform on this backend is called. It exists for namespaces whose defaults
    suit one-shot scripts rather than a library: see ``_pyfftw_setup``.

    ``workers_kwarg``, if given, is the name the underlying module uses for its
    own thread-count argument (``"workers"`` for scipy, ``"threads"`` for
    pyfftw). The caller-facing ``workers`` keyword is always popped out of
    ``kwargs`` here, so a module with ``workers_kwarg=None`` (numpy, mkl)
    silently drops it instead of forwarding an argument the underlying
    function does not accept. When ``workers_kwarg="threads"``,
    ``workers=-1`` is translated to ``os.cpu_count()`` because pyfftw, unlike
    scipy, does not treat -1 as "all cores" itself. ``workers=None`` never
    reaches this branch at all: nothing is added to ``kwargs``, which is the
    true no-op the default is supposed to be.
    """
    done = False

    def _make(transform: str) -> TransformFunc:
        def fn(*args: Any, **kwargs: Any) -> ArrayResult:
            nonlocal done
            workers = kwargs.pop("workers", None)
            module = importlib.import_module(module_path)
            if setup is not None and not done:
                setup(module)
                done = True
            func = getattr(module, transform, None)
            if func is None:
                raise NotImplementedError(f"Backend '{name}' does not implement '{transform}'.")
            if workers is not None and workers_kwarg is not None:
                if workers == -1 and workers_kwarg == "threads":
                    workers = os.cpu_count()
                kwargs[workers_kwarg] = workers
            result: ArrayResult = func(*args, **kwargs)
            return result

        return fn

    return Backend(name, {t: _make(t) for t in transforms})


def _pyfftw_setup(module: Any) -> None:
    """Enable pyfftw's plan cache.

    ``pyfftw.interfaces`` ships with its cache DISABLED, so every call re-plans
    the transform from scratch. FFTW's advantage is an expensive plan amortised
    over many transforms, which is exactly the library case, so re-planning
    per call gives away the reason to use it at all.

    Measured on an AMD Ryzen 9 9900X, ms/FFT, cache off -> on:

        N=1024     0.540 -> 0.216
        N=65536    6.527 -> 2.846
        N=262144  18.560 -> 11.238

    At 65536 and above this moves pyfftw from slowest of four backends to
    second only to MKL. At N=1024 it remains the slowest even cached (numpy
    0.177) -- per-call overhead still dominates a transform that small, so
    the cache narrows the gap rather than closing it.
    """
    import pyfftw

    if not pyfftw.interfaces.cache.is_enabled():
        pyfftw.interfaces.cache.enable()


def _cupy_backend() -> Backend:
    """Backend wrapping cupy.fft with automatic host<->device transfer."""
    name = "cupy"

    def _make(transform: str) -> TransformFunc:
        def fn(x: ArrayLike, *args: Any, **kwargs: Any) -> ArrayResult:
            import cupy as cp

            # cupy.fft has no per-call thread-count argument -- cuFFT already
            # parallelises across the GPU's own cores -- so `workers` is a
            # no-op hint here, not something to forward. Popped rather than
            # left in kwargs so it never reaches cp.fft, which would raise.
            kwargs.pop("workers", None)
            func = getattr(cp.fft, transform, None)
            if func is None:
                raise NotImplementedError(f"Backend '{name}' does not implement '{transform}'.")
            x_gpu = cp.asarray(x)
            result_gpu = func(x_gpu, *args, **kwargs)
            result: ArrayResult = cp.asnumpy(result_gpu)
            return result

        return fn

    return Backend(name, {t: _make(t) for t in TRANSFORM_NAMES})


def _torch_backend() -> Backend:
    """Backend wrapping torch.fft: axis/axes -> dim, numpy in / numpy out.

    torch.fft preserves input precision (float64/complex128 in -> complex128
    out, float32/complex64 in -> complex64 out), so no explicit casting is
    needed here.
    """
    name = "torch"

    def _make_1d(transform: str) -> TransformFunc:
        def fn(
            x: ArrayLike, n: int | None = None, axis: int = -1, norm: str | None = None,
            workers: int | None = None,
        ) -> ArrayResult:
            import torch

            # torch.fft has no thread-count argument of its own -- it is
            # already inherently parallel (CPU: intra-op threads; CUDA: the
            # GPU) -- so `workers` is accepted only to satisfy the shared
            # call signature and is otherwise a no-op here.
            del workers
            func = getattr(torch.fft, transform, None)
            if func is None:
                raise NotImplementedError(f"Backend '{name}' does not implement '{transform}'.")
            x_t = torch.from_numpy(np.asarray(x))
            result = func(x_t, n=n, dim=axis, norm=norm)
            arr: ArrayResult = result.numpy()
            return arr

        return fn

    def _make_nd(
        transform: str,
    ) -> TransformFunc:
        def fn(
            x: ArrayLike, s: tuple[int, ...] | None = None, axes: tuple[int, ...] | None = None,
            norm: str | None = None, workers: int | None = None,
        ) -> ArrayResult:
            import torch

            # See _make_1d: no per-call thread-count equivalent, ignored.
            del workers

            func = getattr(torch.fft, transform, None)
            if func is None:
                raise NotImplementedError(f"Backend '{name}' does not implement '{transform}'.")
            x_t = torch.from_numpy(np.asarray(x))
            # Omit s=/dim= entirely when unset rather than forwarding None.
            # scipy and numpy read axes=None as "use the default axes", but
            # torch.fft.fft2 rejects it outright:
            #   TypeError: fft_fft2(): argument 'dim' must be tuple of ints,
            #              not NoneType
            # Leaving the keyword out lets torch apply its own default, which
            # matches scipy's (last two axes for *2, all axes for *n).
            kwargs: dict[str, Any] = {"norm": norm}
            if s is not None:
                kwargs["s"] = s
            if axes is not None:
                kwargs["dim"] = axes
            result = func(x_t, **kwargs)
            arr: ArrayResult = result.numpy()
            return arr

        return fn

    funcs = {t: _make_1d(t) for t in ("fft", "ifft", "rfft", "irfft")}
    funcs.update({t: _make_nd(t) for t in ("fft2", "ifft2", "fftn", "ifftn")})
    return Backend(name, funcs)


def next_fast_len(n: int, real: bool = False) -> int:
    """Smallest length >= ``n`` that FFT implementations transform quickly.

    A prime-length transform falls back to Bluestein's algorithm or a direct
    DFT; a length whose factors are small primes gets the radix path. The gap
    is not marginal. Measured on an AMD Ryzen 9 9900X with scipy 1.18:

        N = 10007  (prime)  0.294 ms  ->  10080   0.068 ms    4.3x
        N = 65521  (prime)  3.440 ms  ->  65536   0.542 ms    6.4x
        N = 262139 (prime) 15.886 ms  -> 262144   1.411 ms   11.3x

    The last case adds five zeros for an eleven-fold speedup. Pair this with
    the ``n=`` argument, which does the zero-padding:

        X = fftkit.fft(x, n=fftkit.next_fast_len(len(x)))

    Args:
        n: Minimum length required.
        real: Pass True when the transform is ``rfft``/``irfft``; some sizes are
            fast for real input but not complex, and vice versa.

    Returns:
        The next fast length, or ``n`` itself if it is already fast.

    Note:
        Delegates to :func:`scipy.fft.next_fast_len`, whose preferred radices
        are 2, 3, 5, 7 and 11. FFTW agrees with it on every length tested here.
        cuFFT prefers products of 2, 3, 5 and 7 only, so for the ``cupy``
        backend an 11-smooth result from this function may not be optimal --
        measure with ``fftkit bench`` if that matters for your sizes.

        Padding changes the transform: it interpolates the spectrum onto a
        finer frequency grid without adding resolution, and it alters power
        normalisation. Use it for speed and for peak-location refinement, not
        to separate two tones that were unresolved at the original length.
    """
    from scipy.fft import next_fast_len as _scipy_next_fast_len

    result: int = _scipy_next_fast_len(n, real=real)
    return result


def _resize_axis(arr: NDArray[Any], n: int, axis: int) -> NDArray[Any]:
    """Truncate or zero-pad ``arr`` along ``axis`` to length ``n``.

    This is what numpy's and scipy's ``n=``/``s=`` mean for a forward
    transform, and it lets backends whose native API cannot express a
    transform length still honour the argument instead of ignoring it.
    """
    current = arr.shape[axis]
    if n == current:
        return arr
    if n < current:
        index: list[slice] = [slice(None)] * arr.ndim
        index[axis] = slice(0, n)
        return arr[tuple(index)]
    pad = [(0, 0)] * arr.ndim
    pad[axis] = (0, n - current)
    return np.pad(arr, pad)


def _tensorflow_backend() -> Backend:
    """Backend wrapping tf.signal: partial matrix, last-axis/last-2-axes only.

    tf.signal has no ``axis``/``norm`` keyword; it always transforms the
    innermost dimension(s). We move the requested axis(es) to the end,
    transform, and move them back. Norm scaling ('ortho' and 'forward')
    is implemented by post-multiplying the result.
    tf.signal has no general n-dimensional transform, so fftn/ifftn are
    intentionally left unimplemented.
    """
    name = "tensorflow"

    def _cast_to_complex(arr: NDArray[Any]) -> NDArray[Any]:
        """Cast real input to complex while preserving precision.

        float64 -> complex128, float32 -> complex64.
        Complex types are left unchanged.
        """
        dtype = arr.dtype
        if dtype == np.float64:
            return arr.astype(np.complex128)
        elif dtype == np.float32:
            return arr.astype(np.complex64)
        # Complex types pass through unchanged
        return arr

    def _get_norm_scale(norm: str | None, transform: str, logical_length: int) -> float | None:
        """Compute the norm scaling factor to apply after TensorFlow's computation.

        TensorFlow computes according to 'backward' norm (1/N applied in ifft).
        This returns the additional scaling needed for 'ortho' or 'forward'.

        Args:
            norm: The requested norm ('backward', 'ortho', 'forward', or None).
            transform: The transform name ('fft', 'ifft', 'rfft', 'irfft', 'fft2d', 'ifft2d').
            logical_length: The length N (after n=/s= padding/truncation).

        Returns:
            The scaling factor to multiply the result by, or None if no scaling needed.
        """
        if norm is None or norm == 'backward':
            return None

        N = float(logical_length)
        sqrt_N = np.sqrt(N)

        # Forward transforms: fft, rfft, fft2d
        if transform in ('fft', 'rfft', 'fft2d'):
            if norm == 'ortho':
                return float(1.0 / sqrt_N)
            elif norm == 'forward':
                return 1.0 / N
        # Inverse transforms: ifft, irfft, ifft2d
        # Note: TensorFlow's ifft already includes 1/N, so we multiply on top
        elif transform in ('ifft', 'irfft', 'ifft2d'):
            if norm == 'ortho':
                return float(sqrt_N)
            elif norm == 'forward':
                return N

        return None

    def _make_1d(tf_name: str) -> TransformFunc:
        def fn(
            x: ArrayLike, n: int | None = None, axis: int = -1, norm: str | None = None,
            workers: int | None = None,
        ) -> ArrayResult:
            import tensorflow as tf

            # tf.signal has no per-call thread-count argument -- TensorFlow's
            # op-level parallelism is controlled process-wide, not per FFT --
            # so `workers` is accepted for signature compatibility and ignored.
            del workers
            x_arr = np.asarray(x)

            # Cast real input to complex (tf.signal.fft/ifft only accept complex types)
            # rfft/irfft accept real input natively, so skip for those.
            if tf_name in ('fft', 'ifft'):
                x_arr = _cast_to_complex(x_arr)

            # tf.signal.fft/ifft take no fft_length, so n= has to be applied by
            # resizing the input first -- which is exactly what numpy's n= means
            # for a forward transform: truncate to n, or zero-pad up to n.
            # Silently dropping n here (the previous behaviour) returned an array
            # of the wrong length with no error.
            #
            # rfft's n= is also an INPUT length, but tf.signal.rfft accepts
            # fft_length directly, so that path stays as-is. irfft's n= is an
            # OUTPUT length, which only fft_length can express -- resizing its
            # input would mean something different.

            # Determine the logical length N that the norm scaling divides by.
            # For every transform except irfft that is the length along the
            # transformed axis. irfft is the exception: its input is a
            # half-spectrum of m bins describing a real signal of length
            # 2*(m-1), and the normalization constant follows the real signal,
            # not the half-spectrum. Using m here scales `ortho` by
            # sqrt(m/N) and `forward` by m/N -- a 28% and 48% error
            # respectively at m=33, silently wrong rather than raising.
            logical_length = x_arr.shape[axis]
            if tf_name == "irfft" and n is None:
                logical_length = 2 * (x_arr.shape[axis] - 1)
            if n is not None and tf_name in ("fft", "ifft"):
                x_arr = _resize_axis(x_arr, n, axis)
                logical_length = n

            x_tf = tf.convert_to_tensor(x_arr)
            x_tf = tf.experimental.numpy.moveaxis(x_tf, axis, -1)
            func = getattr(tf.signal, tf_name)

            if n is not None and tf_name in ("rfft", "irfft"):
                result = func(x_tf, fft_length=[n])
                logical_length = n
            else:
                result = func(x_tf)

            result = tf.experimental.numpy.moveaxis(result, -1, axis)
            arr: ArrayResult = result.numpy()

            # Apply norm scaling if needed
            scale = _get_norm_scale(norm, tf_name, logical_length)
            if scale is not None:
                arr = arr * scale

            return arr

        return fn

    def _make_2d(tf_name: str) -> TransformFunc:
        def fn(
            x: ArrayLike, s: tuple[int, ...] | None = None, axes: tuple[int, ...] | None = None,
            norm: str | None = None, workers: int | None = None,
        ) -> ArrayResult:
            import tensorflow as tf

            # See _make_1d: no per-call thread-count equivalent, ignored.
            del workers
            resolved_axes = (-2, -1) if axes is None else tuple(axes)
            x_arr = np.asarray(x)

            # For fft2d/ifft2d (which work on 2-D complex input),
            # cast real input to complex. Note: we're here for fft2/ifft2,
            # not for rfft2/irfft2 (which don't exist in the registered matrix).
            x_arr = _cast_to_complex(x_arr)

            # Same defect as the 1-D path: fft2d/ifft2d take no fft_length, and
            # the registry only ever maps fft2/ifft2 onto them, so the old
            # `fft_length=s` branch was dead code and s= was silently discarded.
            # Apply it by resizing each transformed axis, matching numpy's s=.
            logical_lengths = list(x_arr.shape[ax] for ax in resolved_axes)
            if s is not None:
                if len(s) != len(resolved_axes):
                    raise ValueError(
                        f"Backend '{name}': s={s!r} has {len(s)} entries but "
                        f"{len(resolved_axes)} axes were selected ({resolved_axes!r})."
                    )
                for length, ax in zip(s, resolved_axes):
                    x_arr = _resize_axis(x_arr, length, ax)
                logical_lengths = list(s)

            x_tf = tf.convert_to_tensor(x_arr)
            x_tf = tf.experimental.numpy.moveaxis(x_tf, resolved_axes, (-2, -1))
            func = getattr(tf.signal, tf_name)
            result = func(x_tf)
            result = tf.experimental.numpy.moveaxis(result, (-2, -1), resolved_axes)
            arr: ArrayResult = result.numpy()

            # Apply norm scaling if needed
            # For 2-D transforms, N = product of the two transformed axis lengths
            logical_length = int(np.prod(logical_lengths))
            scale = _get_norm_scale(norm, tf_name, logical_length)
            if scale is not None:
                arr = arr * scale

            return arr

        return fn

    funcs = {
        "fft": _make_1d("fft"),
        "ifft": _make_1d("ifft"),
        "rfft": _make_1d("rfft"),
        "irfft": _make_1d("irfft"),
        "fft2": _make_2d("fft2d"),
        "ifft2": _make_2d("ifft2d"),
    }
    return Backend(name, funcs)


def _accelerate_backend() -> Backend:
    """Backend wrapping Apple's Accelerate/vDSP framework via ctypes.

    Only a 1-D complex forward FFT of power-of-two length is implemented;
    every other transform raises NotImplementedError.
    """
    name = "accelerate"

    def _fft(
        x: ArrayLike, n: int | None = None, axis: int = -1, norm: str | None = None,
        workers: int | None = None,
    ) -> ArrayResult:
        import ctypes
        import ctypes.util

        # vDSP is called here through a single-threaded ctypes path with no
        # thread-count argument, so `workers` is accepted for signature
        # compatibility and otherwise ignored.
        del workers
        if sys.platform != "darwin":
            raise NotImplementedError("Accelerate FFT is only available on macOS.")
        if n is not None:
            raise NotImplementedError("Backend 'accelerate' does not support n= padding/truncation.")
        if norm not in (None, "backward"):
            raise NotImplementedError(f"Backend 'accelerate' does not support norm={norm!r}.")

        lib_path = ctypes.util.find_library("Accelerate")
        if lib_path is None:
            raise RuntimeError("Accelerate framework not found.")
        accel = ctypes.cdll.LoadLibrary(lib_path)

        class DSPDoubleSplitComplex(ctypes.Structure):
            _fields_ = [
                ("realp", ctypes.POINTER(ctypes.c_double)),
                ("imagp", ctypes.POINTER(ctypes.c_double)),
            ]

        x_arr = np.asarray(x, dtype=np.complex128)
        length = x_arr.shape[axis]
        log2n = int(np.log2(length))
        if 2**log2n != length:
            raise ValueError("vDSP FFT requires power-of-two length.")

        real = np.ascontiguousarray(np.real(x_arr), dtype=np.float64)
        imag = np.ascontiguousarray(np.imag(x_arr), dtype=np.float64)
        split = DSPDoubleSplitComplex(
            real.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            imag.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )

        accel.vDSP_create_fftsetupD.restype = ctypes.c_void_p
        setup = accel.vDSP_create_fftsetupD(ctypes.c_uint(log2n), 2)
        if not setup:
            raise RuntimeError("Failed to create FFT setup.")
        accel.vDSP_fft_zipD(ctypes.c_void_p(setup), ctypes.byref(split), 1, ctypes.c_uint(log2n), 1)
        accel.vDSP_destroy_fftsetupD(ctypes.c_void_p(setup))
        return real + 1j * imag

    return Backend(name, {"fft": _fft})


# =============================================================================
# Backend Registry
# =============================================================================
# All backends are registered unconditionally; availability probing (not
# registration) decides what can actually run on this machine.

BACKENDS: dict[str, Backend] = {
    "scipy": _lazy_module_backend("scipy", "scipy.fft", workers_kwarg="workers"),
    "numpy": _lazy_module_backend("numpy", "numpy.fft"),
    "mkl": _lazy_module_backend("mkl", "mkl_fft.interfaces.numpy_fft"),
    "pyfftw": _lazy_module_backend(
        "pyfftw", "pyfftw.interfaces.numpy_fft", setup=_pyfftw_setup, workers_kwarg="threads"
    ),
    "cupy": _cupy_backend(),
    "torch": _torch_backend(),
    "tensorflow": _tensorflow_backend(),
    "accelerate": _accelerate_backend(),
}

# Cache of probe results for get_available_backends(); populated on first call.
_available_backends_cache: list[str] | None = None


# =============================================================================
# Backend Selection Utilities
# =============================================================================

def get_fft_func(backend: str | None = None) -> TransformFunc:
    """Get the forward complex FFT function (axis=-1) for the specified backend.

    The returned callable carries the same axis-migration guard as
    ``fftkit.fft``: this is the 0.1.x idiom (``get_fft_func('scipy')(x)``), so
    it is the path most likely to be holding a 2-D array that used to be
    transformed along axis 0. See :class:`AxisDefaultWarning`.
    """
    backend = backend or DEFAULT_BACKEND
    if backend not in BACKENDS:
        raise ValueError(f"Unknown FFT backend: {backend}. Available: {list(BACKENDS.keys())}")
    inner = BACKENDS[backend].get("fft")

    def fft_with_axis_guard(x: ArrayLike, *args: Any, **kwargs: Any) -> ArrayResult:
        if not args and "axis" not in kwargs:
            kwargs["axis"] = resolve_axis(x, UNSET_AXIS, "fft")
        return inner(x, *args, **kwargs)

    return fft_with_axis_guard


def get_backend_names() -> list[str]:
    """Return list of all registered backend names."""
    return list(BACKENDS.keys())


def get_available_backends(refresh: bool = False) -> list[str]:
    """Return list of backends that are actually importable/working.

    Results are cached at module level after the first call; pass
    ``refresh=True`` to re-probe (e.g. after installing a new backend).
    """
    global _available_backends_cache
    if _available_backends_cache is not None and not refresh:
        return _available_backends_cache

    available = []
    test_signal = np.array([1, 2, 3, 4], dtype=np.complex128)

    for name, backend in BACKENDS.items():
        try:
            func = backend.get("fft")
            result = func(test_signal)
            if result is not None and len(result) == len(test_signal):
                available.append(name)
        except Exception:
            pass

    _available_backends_cache = available
    return available


def mkl_available() -> bool:
    """Check if Intel MKL FFT is available."""
    try:
        import mkl_fft  # noqa: F401
        return True
    except ImportError:
        return False


def get_optimal_backend(array_size: int, batch_size: int = 1, prefer_gpu: bool = True,
                         gpu_resident: bool = False) -> str:
    """Select optimal backend based on workload characteristics.

    Args:
        array_size: Total number of elements in FFT input
        batch_size: Number of FFTs to compute (for batching decisions)
        prefer_gpu: Whether to prefer GPU when beneficial
        gpu_resident: If True, assume data stays on GPU (no transfer overhead)
                      This dramatically changes GPU vs CPU tradeoffs.

    Returns:
        str: Name of recommended backend ('scipy', 'mkl', 'cupy', etc.)

    Note:
        With data transfer included (gpu_resident=False), MKL typically wins
        due to PCIe overhead (~0.5ms per transfer). GPU only wins when:
        - Data stays on GPU (gpu_resident=True), OR
        - Very large single FFTs (256K+), OR
        - Multiple FFT operations in a pipeline
    """
    # GPU-resident mode: GPU wins for batches
    if gpu_resident and prefer_gpu and gpu_available():
        if batch_size >= 16 or array_size >= 16384:
            return 'cupy'

    # With transfer: GPU only wins for very large FFTs
    if prefer_gpu and gpu_available() and not gpu_resident:
        if array_size >= 262144:  # 256K+ single FFTs
            return 'cupy'

    # MKL is beneficial for most CPU FFTs (any size >= 1K)
    if mkl_available() and array_size >= 1024:
        return 'mkl'

    # Default to scipy
    return 'scipy'


def benchmark_backends(size: int = 8192, iterations: int = 100) -> dict[str, float | str]:
    """Quick benchmark of available backends.

    Returns dict of {backend_name: time_per_fft_ms}
    """
    import time

    results: dict[str, float | str] = {}
    test_signal = np.random.randn(size) + 1j * np.random.randn(size)

    for name in get_available_backends():
        try:
            func = BACKENDS[name].get("fft")
            # Warmup
            func(test_signal)

            start = time.perf_counter()
            for _ in range(iterations):
                func(test_signal)
            elapsed = time.perf_counter() - start

            results[name] = (elapsed / iterations) * 1000  # ms per FFT
        except Exception as e:
            results[name] = f"Error: {e}"

    return results


# =============================================================================
# Backwards-compatible standalone shims (0.1.0 API)
# =============================================================================

def accelerate_fft(x: ArrayLike, axis: int = -1) -> ArrayResult:
    return BACKENDS["accelerate"].get("fft")(x, axis=axis)


def scipy_fft(x: ArrayLike, axis: int = -1) -> ArrayResult:
    return BACKENDS["scipy"].get("fft")(x, axis=axis)


def numpy_fft(x: ArrayLike, axis: int = -1) -> ArrayResult:
    return BACKENDS["numpy"].get("fft")(x, axis=axis)


def mkl_fft_transform(x: ArrayLike, axis: int = -1) -> ArrayResult:
    """FFT via Intel MKL (2-20x faster for large arrays).

    Requires mkl_fft package. Install with:
        uv pip install mkl_fft --index-url https://software.repos.intel.com/python/pypi
    """
    try:
        return BACKENDS["mkl"].get("fft")(x, axis=axis)
    except ModuleNotFoundError as e:
        raise ImportError(
            "mkl_fft not installed. Install with:\n"
            "uv pip install mkl_fft --index-url https://software.repos.intel.com/python/pypi "
            "--extra-index-url https://pypi.org/simple"
        ) from e


def cupy_fft(x: ArrayLike, axis: int = -1) -> ArrayResult:
    """FFT using NVIDIA GPU via CuPy (auto-transfer mode).

    Accepts numpy array, transfers to GPU, computes FFT, returns numpy array.

    Requires: pip install cupy-cuda12x (for CUDA 12.x)
    """
    try:
        return BACKENDS["cupy"].get("fft")(x, axis=axis)
    except ModuleNotFoundError as e:
        raise ImportError(
            "CuPy not installed. Install with:\n"
            "uv pip install cupy-cuda12x"
        ) from e


def register_mkl_scipy_backend() -> bool:
    """Set MKL as the global scipy.fft backend for all scipy.fft calls.

    After calling this, all scipy.fft operations use MKL automatically.

    Returns True on success, False if the MKL scipy interface is unavailable.
    On failure it also emits :class:`MklBackendWarning` naming the missing
    module, because "False" alone is not actionable and the two ways this
    fails need different fixes:

    - ``mkl_fft`` is not installed at all -> ``pip install "fftkit[mkl]"``.
    - ``mkl_fft`` IS installed and fftkit's own ``mkl`` backend works, but
      ``mkl_fft.interfaces.scipy_fft`` additionally imports the separate
      ``mkl`` package, which ``mkl-fft`` does not depend on -> ``pip install mkl``.

    The second case is the confusing one: ``mkl_available()`` returns True and
    ``get_fft_func('mkl')`` works, while this function returns False.
    """
    try:
        import mkl_fft.interfaces.scipy_fft
        from scipy.fft import set_global_backend
        set_global_backend(mkl_fft.interfaces.scipy_fft)
        return True
    except ImportError as exc:
        missing = getattr(exc, "name", None) or "a required module"
        if missing == "mkl":
            hint = (
                "mkl_fft is installed and fftkit's own 'mkl' backend works, but "
                "mkl_fft.interfaces.scipy_fft also imports the separate 'mkl' "
                "package, which mkl-fft does not declare as a dependency. "
                "Install it with: pip install mkl"
            )
        else:
            hint = 'Install the MKL FFT bindings with: pip install "fftkit[mkl]"'
        warnings.warn(
            f"register_mkl_scipy_backend() left scipy.fft unchanged: cannot import "
            f"{missing!r} ({exc}). {hint}",
            MklBackendWarning,
            stacklevel=2,
        )
        return False


if __name__ == '__main__':
    print('=== FFT Backends ===\n')
    print(f"Available: {get_available_backends()}")
    results = benchmark_backends(size=65536, iterations=50)
    for name, time_ms in sorted(results.items(), key=lambda x: x[1] if isinstance(x[1], float) else 999):
        print(f"  {name}: {time_ms:.3f} ms" if isinstance(time_ms, float) else f"  {name}: {time_ms}")
