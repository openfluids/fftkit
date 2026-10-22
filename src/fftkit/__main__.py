"""Command-line interface for fftkit.

Run ``python -m fftkit <command>`` or, once installed, ``fftkit <command>``.

Commands:

- ``info``: report the detected backend, the full backend availability
  matrix, and GPU status for this machine.
- ``bench``: benchmark available FFT backends (or CPU vs GPU with ``--gpu``).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys

import numpy
import scipy

import fftkit

from .backends import BACKENDS, TRANSFORM_NAMES, get_available_backends, get_backend_names
from .gpu import benchmark_cpu_vs_gpu, get_gpu_info

# Short, human-readable reasons why a backend probe failed, keyed by backend
# name. Anything not listed here falls back to a generic "not available"
# message built from the probe exception.
_UNAVAILABLE_HINTS: dict[str, str] = {
    "mkl": "not installed (pip install mkl_fft)",
    "pyfftw": "not installed (pip install pyfftw)",
    "cupy": "not installed / no CUDA GPU",
    "torch": "not installed (pip install torch)",
    "tensorflow": "not installed (pip install tensorflow)",
    "accelerate": "macOS only" if sys.platform != "darwin" else "Accelerate framework not found",
}


def _unavailable_reason(name: str) -> str:
    return _UNAVAILABLE_HINTS.get(name, "not available")


def _selecting_env_var() -> str | None:
    import os

    if os.environ.get("FFTKIT_BACKEND"):
        return "FFTKIT_BACKEND"
    if os.environ.get("PYMODAL_FFT_BACKEND"):
        return "PYMODAL_FFT_BACKEND"
    return None


def cmd_info(args: argparse.Namespace) -> int:
    available = set(get_available_backends())
    env_var = _selecting_env_var()

    print("=== fftkit info ===\n")
    print(f"fftkit version : {fftkit.__version__}")
    print(f"Python version : {platform.python_version()}")
    print(f"numpy version  : {numpy.__version__}")
    print(f"scipy version  : {scipy.__version__}")
    print()
    if env_var:
        print(f"Default backend: {fftkit.DEFAULT_BACKEND}  (selected via ${env_var})")
    else:
        print(f"Default backend: {fftkit.DEFAULT_BACKEND}  (auto-detected)")

    print("\nBackends:")
    name_w = max(len(n) for n in get_backend_names())
    header = f"  {'name':<{name_w}}  {'available':<9}  transforms / reason"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name in get_backend_names():
        backend = BACKENDS[name]
        if name in available:
            supported = [t for t in TRANSFORM_NAMES if backend.supports(t)]
            detail = ", ".join(supported)
            print(f"  {name:<{name_w}}  {'yes':<9}  {detail}")
        else:
            print(f"  {name:<{name_w}}  {'no':<9}  {_unavailable_reason(name)}")

    print("\nGPU:")
    gpu_info = get_gpu_info()
    if gpu_info.get("available"):
        print("  available            : yes")
        print(f"  device_id            : {gpu_info['device_id']}")
        print(f"  compute_capability   : {gpu_info['compute_capability']}")
        print(f"  memory_total_gb      : {gpu_info['memory_total_gb']:.2f}")
        print(f"  memory_free_gb       : {gpu_info['memory_free_gb']:.2f}")
    else:
        print(f"  available            : no ({gpu_info.get('error', 'unknown')})")

    return 0


def _run_backend_bench(size: int, iterations: int) -> dict[str, float | dict[str, str]]:
    """Benchmark all available backends at ``size``, catching per-backend errors."""
    import time

    import numpy as np

    results: dict[str, float | dict[str, str]] = {}
    test_signal = np.random.randn(size) + 1j * np.random.randn(size)

    for name in get_available_backends():
        try:
            func = BACKENDS[name].get("fft")
            func(test_signal)  # warmup

            start = time.perf_counter()
            for _ in range(iterations):
                func(test_signal)
            elapsed = time.perf_counter() - start
            results[name] = (elapsed / iterations) * 1000  # ms per FFT
        except Exception as e:
            results[name] = {"error": str(e)}

    return results


def cmd_bench(args: argparse.Namespace) -> int:
    if args.gpu:
        sizes = [args.size] if args.size is not None else None
        batch_sizes = [args.batch] if args.batch is not None else None
        results = benchmark_cpu_vs_gpu(sizes=sizes, batch_sizes=batch_sizes, iterations=args.iters)
        if args.json:
            print(json.dumps(results, indent=2))
            return 0

        print(f"=== fftkit bench --gpu (iterations={args.iters}) ===\n")
        gpu_info = get_gpu_info()
        if not gpu_info.get("available"):
            print(f"No GPU available ({gpu_info.get('error', 'unknown')}); showing CPU timings only.\n")
        print(f"{'config':>15s} {'CPU (ms)':>10s} {'GPU (ms)':>10s} {'speedup':>10s}")
        print("-" * 50)
        for key, vals in results.items():
            print(f"{key:>15s} {vals['cpu_ms']:>10} {vals['gpu_ms']:>10} {vals['speedup']:>10}")
        return 0

    size = args.size if args.size is not None else 65536
    bench_results = _run_backend_bench(size, args.iters)

    if args.json:
        print(json.dumps({"size": size, "iterations": args.iters, "results": bench_results}, indent=2))
        return 0

    print(f"=== fftkit bench (size={size}, iterations={args.iters}) ===")
    print("Speedup is relative to the slowest backend in this run.\n")

    timings: dict[str, float] = {
        name: val for name, val in bench_results.items() if isinstance(val, (int, float))
    }
    errors: dict[str, dict[str, str]] = {
        name: val for name, val in bench_results.items() if isinstance(val, dict)
    }

    if timings:
        slowest = max(timings.values())
        name_w = max(len(n) for n in timings)
        print(f"  {'backend':<{name_w}}  {'time (ms)':>10}  {'speedup':>8}")
        for name, ms in sorted(timings.items(), key=lambda kv: kv[1]):
            speedup = slowest / ms if ms > 0 else float("inf")
            print(f"  {name:<{name_w}}  {ms:>10.4f}  {speedup:>7.2f}x")
    else:
        print("  No backend produced a valid timing.")

    if errors:
        print("\n  Errors:")
        for name, err in errors.items():
            print(f"    {name}: {err['error']}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fftkit", description="Inspect and benchmark fftkit FFT backends.")
    subparsers = parser.add_subparsers(dest="command")

    info_parser = subparsers.add_parser("info", help="report detected backend, availability, and GPU status")
    info_parser.set_defaults(func=cmd_info)

    bench_parser = subparsers.add_parser("bench", help="benchmark available FFT backends")
    bench_parser.add_argument(
        "--size", type=int, default=None,
        help="FFT size. Non-GPU default: 65536. With --gpu: single size to test "
             "instead of the default sweep ([1024, 4096, 16384, 65536]).",
    )
    bench_parser.add_argument(
        "--batch", type=int, default=None,
        help="Batch size to test with --gpu, instead of the default sweep ([1, 16, 64, 128]). "
             "Ignored without --gpu.",
    )
    bench_parser.add_argument("--iters", type=int, default=50, help="iterations per backend (default: 50)")
    bench_parser.add_argument("--gpu", action="store_true", help="compare CPU vs GPU instead of backend-vs-backend")
    bench_parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of a table")
    bench_parser.set_defaults(func=cmd_bench)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        parser.print_help()
        return 1

    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
